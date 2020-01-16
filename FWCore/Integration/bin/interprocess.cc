#include "boost/program_options.hpp"
#include "boost/interprocess/shared_memory_object.hpp"

#include "boost/interprocess/managed_shared_memory.hpp"
#include <boost/interprocess/sync/named_mutex.hpp>
#include <boost/interprocess/sync/named_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>

#include "TClass.h"
#include "TBufferFile.h"

#include <string>
#include <iostream>
#include <atomic>
#include <thread>
#include <signal.h>

#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "DataFormats/TestObjects/interface/ThingCollection.h"

static char const* const kMemoryNameOpt = "memory-name";
static char const* const kMemoryNameCommandOpt = "memory-name,m";
static char const* const kUniqueIDOpt = "unique-id";
static char const* const kUniqueIDCommandOpt = "unique-id,i";
static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt = "help,h";

//NOTE: Can use TestProcessor as the harness for the worker

namespace {
  std::string unique_name(std::string iBase, std::string_view ID) {
    iBase.append(ID);
    return iBase;
  }

  template <typename T>
  class Serializer {
  public:
    Serializer(boost::interprocess::managed_shared_memory& iSM,
               std::string iSMName,
               std::string const& iBase,
               std::string_view ID)
        : managed_shm_(&iSM),
          smName_{std::move(iSMName)},
          name_{unique_name(iBase, ID)},
          class_{TClass::GetClass(typeid(T))},
          bufferFile_{TBuffer::kWrite} {
      buffer_ = managed_shm_->find<char>(name_.c_str());
      assert(buffer_.first);
      std::pair<bool*, std::size_t> sm_buffer_resized =
          managed_shm_->find<bool>(unique_name(iBase + "_resize", ID).c_str());
      buffer_resized_ = sm_buffer_resized.first;
      assert(buffer_resized_);
    }

    void serialize(T& iValue) {
      bufferFile_.Reset();
      class_->WriteBuffer(bufferFile_, &iValue);

      if (static_cast<unsigned long>(bufferFile_.Length()) > buffer_.second) {
        managed_shm_->destroy<char>(name_.c_str());
        //auto diff = bufferFile_.Length() - buffer_.second;
        // can not grow an existing shared memory segment that is already mapped
        //auto success = managed_shm_->grow(smName_.c_str(), diff);
        //assert(success);

        buffer_.first = managed_shm_->construct<char>(name_.c_str())[bufferFile_.Length()](0);
        buffer_.second = bufferFile_.Length();
        *buffer_resized_ = true;
      }
      std::copy(bufferFile_.Buffer(), bufferFile_.Buffer() + bufferFile_.Length(), buffer_.first);
    }

  private:
    boost::interprocess::managed_shared_memory* const managed_shm_;
    std::string smName_;
    std::string name_;
    std::pair<char*, std::size_t> buffer_;
    bool* buffer_resized_;
    TClass* const class_;
    TBufferFile bufferFile_;
  };
}  // namespace

class Harness {
public:
  Harness(std::string const& iConfig) : tester_(edm::test::TestProcessor::Config{iConfig}) {}

  edmtest::ThingCollection getBeginRunValue(unsigned int iRun) {
    auto run = tester_.testBeginRun(iRun);
    return *run.get<edmtest::ThingCollection>("beginRun");
  }

  edmtest::ThingCollection getBeginLumiValue(unsigned int iLumi) {
    auto lumi = tester_.testBeginLuminosityBlock(iLumi);
    return *lumi.get<edmtest::ThingCollection>("beginLumi");
  }

  edmtest::ThingCollection getEventValue() {
    auto event = tester_.test();
    return *event.get<edmtest::ThingCollection>();
  }

  edmtest::ThingCollection getEndLumiValue() {
    auto lumi = tester_.testEndLuminosityBlock();
    return *lumi.get<edmtest::ThingCollection>("endLumi");
  }

  edmtest::ThingCollection getEndRunValue() {
    auto run = tester_.testEndRun();
    return *run.get<edmtest::ThingCollection>("endRun");
  }

private:
  edm::test::TestProcessor tester_;
};

namespace {
  std::atomic<bool> s_stopRequested = false;
  std::atomic<bool> s_signal_happened = false;
  std::atomic<bool> s_helperReady = false;
  std::atomic<boost::interprocess::scoped_lock<boost::interprocess::named_mutex>*> s_lock = nullptr;
  std::atomic<bool> s_helperThreadDone = false;
  int s_sig = 0;

  void sig_handler(int sig, siginfo_t*, void*) {
    s_sig = sig;
    s_signal_happened = true;
    while (not s_helperThreadDone) {
    };
    signal(sig, SIG_DFL);
    raise(sig);
  }

  void cleanupThread() {
    std::cerr << "Started cleanup thread\n";
    sigset_t ensemble;

    sigemptyset(&ensemble);
    sigaddset(&ensemble, SIGABRT);
    sigaddset(&ensemble, SIGILL);
    sigaddset(&ensemble, SIGBUS);
    sigaddset(&ensemble, SIGSEGV);
    sigaddset(&ensemble, SIGTERM);
    pthread_sigmask(SIG_BLOCK, &ensemble, NULL);

    std::cerr << "Start loop\n";
    s_helperReady = true;
    while (not s_stopRequested.load()) {
      sleep(30);
      if (s_signal_happened) {
        auto v = s_lock.load();
        if (v) {
          std::cerr << "SIGNAL CAUGHT: unlock\n";
          v->unlock();
        }
        s_helperThreadDone = true;
        std::cerr << "SIGNAL CAUGHT\n";
        break;
      } else {
        std::cerr << "SIGNAL woke\n";
      }
    }
    std::cerr << "Ending cleanup thread\n";
  }
}  // namespace

int main(int argc, char* argv[]) {
  std::string descString(argv[0]);
  descString += " [--";
  descString += kMemoryNameOpt;
  descString += "] memory_name";
  boost::program_options::options_description desc(descString);

  desc.add_options()(kHelpCommandOpt, "produce help message")(
      kMemoryNameCommandOpt, boost::program_options::value<std::string>(), "memory name")(
      kUniqueIDCommandOpt, boost::program_options::value<std::string>(), "unique id");

  boost::program_options::positional_options_description p;
  p.add(kMemoryNameOpt, 1);
  p.add(kUniqueIDOpt, 2);

  boost::program_options::options_description all_options("All Options");
  all_options.add(desc);

  boost::program_options::variables_map vm;
  try {
    store(boost::program_options::command_line_parser(argc, argv).options(all_options).positional(p).run(), vm);
    notify(vm);
  } catch (boost::program_options::error const& iException) {
    std::cout << argv[0] << ": Error while trying to process command line arguments:\n"
              << iException.what() << "\nFor usage and an options list, please do 'cmsRun --help'.";
    return 1;
  }

  if (vm.count(kHelpOpt)) {
    std::cout << desc << std::endl;
    return 0;
  }

  if (!vm.count(kMemoryNameOpt)) {
    std::cout << " no argument given" << std::endl;
    return 1;
  }

  if (!vm.count(kUniqueIDOpt)) {
    std::cout << " no second argument given" << std::endl;
    return 1;
  }

  std::thread helperThread;
  {
    //Setup watchdog thread for crashing signals

    struct sigaction act;
    act.sa_sigaction = sig_handler;
    act.sa_flags = 0;
    sigemptyset(&act.sa_mask);
    sigaction(SIGABRT, &act, nullptr);
    sigaction(SIGILL, &act, nullptr);
    sigaction(SIGBUS, &act, nullptr);
    sigaction(SIGSEGV, &act, nullptr);
    sigaction(SIGTERM, &act, nullptr);

    std::thread t(cleanupThread);
    t.detach();
    helperThread = std::move(t);
  }
  while (s_helperReady.load() == false) {
  }
  try {
    std::string const memoryName(vm[kMemoryNameOpt].as<std::string>());
    std::string const uniqueID(vm[kUniqueIDOpt].as<std::string>());
    {
      using namespace boost::interprocess;
      auto memoryNameUnique = unique_name(memoryName, uniqueID);
      managed_shared_memory managed_shm{open_only, memoryNameUnique.c_str()};

      named_mutex named_mtx{open_or_create, unique_name("mtx", uniqueID).c_str()};
      named_condition named_cndFromMain{open_or_create, unique_name("cndFromMain", uniqueID).c_str()};
      std::pair<bool*, std::size_t> sm_stop = managed_shm.find<bool>(unique_name("stop", uniqueID).c_str());
      std::pair<edm::Transition*, std::size_t> sm_transitionType =
          managed_shm.find<edm::Transition>(unique_name("transitionType", uniqueID).c_str());
      assert(sm_transitionType.first);
      std::pair<unsigned long long*, std::size_t> sm_transitionID =
          managed_shm.find<unsigned long long>(unique_name("transitionID", uniqueID).c_str());

      named_condition named_cndToMain{open_or_create, unique_name("cndToMain", uniqueID).c_str()};

      int counter = 0;

      scoped_lock<named_mutex> lock(named_mtx);
      s_lock.store(&lock);
      std::cerr << uniqueID << " process: initializing " << std::endl;
      int nlines;
      std::cin >> nlines;

      std::string configuration;
      for (int i = 0; i < nlines; ++i) {
        std::string c;
        std::getline(std::cin, c);
        std::cerr << c << "\n";
        configuration += c + "\n";
      }

      Harness harness(configuration);

      {
        //Make sure TestProcessor did not reset the signal handlers
        sigset_t ensemble;
        sigemptyset(&ensemble);
        sigaddset(&ensemble, SIGABRT);
        sigaddset(&ensemble, SIGILL);
        sigaddset(&ensemble, SIGBUS);
        sigaddset(&ensemble, SIGSEGV);
        sigaddset(&ensemble, SIGTERM);
        sigprocmask(SIG_BLOCK, &ensemble, NULL);
      }

      Serializer<edmtest::ThingCollection> serializer(managed_shm, memoryNameUnique, "buffer", uniqueID);
      Serializer<edmtest::ThingCollection> br_serializer(managed_shm, memoryNameUnique, "brbuffer", uniqueID);
      Serializer<edmtest::ThingCollection> bl_serializer(managed_shm, memoryNameUnique, "blbuffer", uniqueID);
      Serializer<edmtest::ThingCollection> el_serializer(managed_shm, memoryNameUnique, "elbuffer", uniqueID);
      Serializer<edmtest::ThingCollection> er_serializer(managed_shm, memoryNameUnique, "erbuffer", uniqueID);

      std::cerr << uniqueID << " process: done initializing" << std::endl;
      named_cndToMain.notify_all();
      while (true) {
        {
          ++counter;
          std::cerr << uniqueID << " process: waiting " << counter << std::endl;
          named_cndFromMain.wait(lock);
          if (*sm_stop.first) {
            break;
          }
        }

        switch (*sm_transitionType.first) {
          case edm::Transition::BeginRun: {
            std::cerr << uniqueID << " process: start beginRun " << std::endl;
            auto value = harness.getBeginRunValue(*sm_transitionID.first);

            br_serializer.serialize(value);
            std::cerr << uniqueID << " process: end beginRun " << value.size() << std::endl;

            break;
          }
          case edm::Transition::BeginLuminosityBlock: {
            std::cerr << uniqueID << " process: start beginLumi " << std::endl;
            auto value = harness.getBeginLumiValue(*sm_transitionID.first);

            bl_serializer.serialize(value);
            std::cerr << uniqueID << " process: end beginLumi " << value.size() << std::endl;

            break;
          }
          case edm::Transition::Event: {
            std::cerr << uniqueID << " process: integrating " << counter << std::endl;
            auto value = harness.getEventValue();

            std::cerr << uniqueID << " process: integrated " << counter << std::endl;

            serializer.serialize(value);
            std::cerr << uniqueID << " process: " << value.size() << " " << counter << std::endl;
            //usleep(10000000);
            break;
          }
          case edm::Transition::EndLuminosityBlock: {
            std::cerr << uniqueID << " process: start endLumi " << std::endl;
            auto value = harness.getEndLumiValue();

            el_serializer.serialize(value);
            std::cerr << uniqueID << " process: end endLumi " << value.size() << std::endl;

            break;
          }
          case edm::Transition::EndRun: {
            std::cerr << uniqueID << " process: start endRun " << std::endl;
            auto value = harness.getEndRunValue();

            er_serializer.serialize(value);
            std::cerr << uniqueID << " process: end endRun " << value.size() << std::endl;

            break;
          }
          default: {
            assert(false);
          }
        }
        std::cerr << uniqueID << " process: notifying " << counter << std::endl;
        named_cndToMain.notify_all();
      }
    }
  } catch (std::exception const& iExcept) {
    std::cerr << iExcept.what();
    s_stopRequested = true;
    return 1;
  } catch (...) {
    std::cerr << "caught unknown exception";
    s_stopRequested = true;
    return 1;
  }
  s_stopRequested = true;
  return 0;
}