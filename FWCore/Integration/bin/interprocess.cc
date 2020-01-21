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

  using namespace boost::interprocess;

  class SMWriteBuffer {
    std::size_t bufferSize_;
    char* buffer_;
    char* bufferIndex_;
    std::array<std::string, 2> bufferNames_;
    std::unique_ptr<managed_shared_memory> sm_;

  public:
    SMWriteBuffer(std::string const& iUniqueName, char* iBufferIndex)
        : bufferSize_{0}, buffer_{nullptr}, bufferIndex_{iBufferIndex} {
      bufferNames_[0] = iUniqueName + "buffer0";
      bufferNames_[1] = iUniqueName + "buffer1";
      assert(bufferIndex_);
    }

    ~SMWriteBuffer() {
      if (sm_) {
        sm_->destroy<char>("buffer");
        sm_.reset();
        shared_memory_object::remove(bufferNames_[*bufferIndex_].c_str());
      }
    }

    void copyToBuffer(const char* iStart, std::size_t iLength) {
      if (iLength > bufferSize_) {
        growBuffer(iLength);
      }
      std::copy(iStart, iStart + iLength, buffer_);
    }

  private:
    void growBuffer(std::size_t iLength) {
      int newBuffer = (*bufferIndex_ + 1) % 2;
      std::cerr << "growing buffer " << iLength << " new index " << newBuffer << "\n";
      if (sm_) {
        sm_->destroy<char>("buffer");
        sm_.reset();
        shared_memory_object::remove(bufferNames_[*bufferIndex_].c_str());
      }
      sm_ = std::make_unique<managed_shared_memory>(open_or_create, bufferNames_[newBuffer].c_str(), iLength + 1024);
      assert(sm_.get());
      bufferSize_ = iLength;
      *bufferIndex_ = newBuffer;
      buffer_ = sm_->construct<char>("buffer")[iLength](0);
      assert(buffer_);
    }
  };

  template <typename T>
  class Serializer {
  public:
    Serializer(SMWriteBuffer& iBuffer)
        : buffer_(iBuffer), class_{TClass::GetClass(typeid(T))}, bufferFile_{TBuffer::kWrite} {}

    void serialize(T& iValue) {
      bufferFile_.Reset();
      class_->WriteBuffer(bufferFile_, &iValue);

      buffer_.copyToBuffer(bufferFile_.Buffer(), bufferFile_.Length());
    }

  private:
    SMWriteBuffer& buffer_;
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
      sleep(60);
      if (s_signal_happened) {
        auto v = s_lock.load();
        if (v) {
          std::cerr << "SIGNAL CAUGHT: unlock\n";
          v->unlock();
        }
        s_helperThreadDone = true;
        std::cerr << "SIGNAL CAUGHT " << s_sig << "\n";
        break;
      } else {
        std::cerr << "SIGNAL woke\n";
      }
    }
    std::cerr << "Ending cleanup thread\n";
  }

  class WorkerChannel {
  public:
    WorkerChannel(std::string const& iName, const std::string& iUniqueID)
        : managed_shm_{open_only, iName.c_str()},
          mutex_{open_or_create, unique_name("mtx", iUniqueID).c_str()},
          cndFromController_{open_or_create, unique_name("cndFromMain", iUniqueID).c_str()},
          stop_{managed_shm_.find<bool>("stop").first},
          transitionType_{managed_shm_.find<edm::Transition>("transitionType").first},
          transitionID_{managed_shm_.find<unsigned long long>("transitionID").first},
          toWorkerBufferIndex_{managed_shm_.find<char>("bufferIndexToWorker").first},
          fromWorkerBufferIndex_{managed_shm_.find<char>("bufferIndexFromWorker").first},
          cndToController_{open_or_create, unique_name("cndToMain", iUniqueID).c_str()},
          keepEvent_{managed_shm_.find<bool>("keepEvent").first},
          lock_{mutex_} {
      assert(stop_);
      assert(transitionType_);
      assert(transitionID_);
      assert(toWorkerBufferIndex_);
      assert(fromWorkerBufferIndex_);
    }

    scoped_lock<named_mutex>* accessLock() { return &lock_; }
    char* toWorkerBufferIndex() { return toWorkerBufferIndex_; }
    char* fromWorkerBufferIndex() { return fromWorkerBufferIndex_; }

    edm::Transition transition() const noexcept { return *transitionType_; }
    unsigned long long transitionID() const noexcept { return *transitionID_; }

    void notifyController() { cndToController_.notify_all(); }

    void waitForController() { cndFromController_.wait(lock_); }

    bool stopRequested() const noexcept { return *stop_; }

    void shouldKeepEvent(bool iChoice) { *keepEvent_ = iChoice; }

  private:
    managed_shared_memory managed_shm_;

    named_mutex mutex_;
    named_condition cndFromController_;
    bool* stop_;
    edm::Transition* transitionType_;
    unsigned long long* transitionID_;
    char* toWorkerBufferIndex_;
    char* fromWorkerBufferIndex_;
    named_condition cndToController_;
    bool* keepEvent_;
    scoped_lock<named_mutex> lock_;
  };

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

    //Need to use signal handler since signals generated
    // from within a program are thread specific which can
    // only be handed by a signal handler
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
      auto controlNameUnique = unique_name(memoryName, uniqueID);

      //This class is holding the lock
      WorkerChannel communicationChannel(controlNameUnique, uniqueID);

      SMWriteBuffer sm_buffer{controlNameUnique, communicationChannel.fromWorkerBufferIndex()};
      int counter = 0;

      s_lock.store(communicationChannel.accessLock());
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
        struct sigaction act;
        act.sa_sigaction = sig_handler;
        act.sa_flags = 0;
        sigemptyset(&act.sa_mask);
        sigaction(SIGABRT, &act, nullptr);
        sigaction(SIGILL, &act, nullptr);
        sigaction(SIGBUS, &act, nullptr);
        sigaction(SIGSEGV, &act, nullptr);
        sigaction(SIGTERM, &act, nullptr);
      }

      Serializer<edmtest::ThingCollection> serializer(sm_buffer);
      Serializer<edmtest::ThingCollection> br_serializer(sm_buffer);
      Serializer<edmtest::ThingCollection> bl_serializer(sm_buffer);
      Serializer<edmtest::ThingCollection> el_serializer(sm_buffer);
      Serializer<edmtest::ThingCollection> er_serializer(sm_buffer);

      std::cerr << uniqueID << " process: done initializing" << std::endl;
      communicationChannel.notifyController();
      while (true) {
        {
          ++counter;
          std::cerr << uniqueID << " process: waiting " << counter << std::endl;
          communicationChannel.waitForController();
          if (communicationChannel.stopRequested()) {
            break;
          }
        }

        switch (communicationChannel.transition()) {
          case edm::Transition::BeginRun: {
            std::cerr << uniqueID << " process: start beginRun " << std::endl;
            auto value = harness.getBeginRunValue(communicationChannel.transitionID());

            br_serializer.serialize(value);
            std::cerr << uniqueID << " process: end beginRun " << value.size() << std::endl;

            break;
          }
          case edm::Transition::BeginLuminosityBlock: {
            std::cerr << uniqueID << " process: start beginLumi " << std::endl;
            auto value = harness.getBeginLumiValue(communicationChannel.transitionID());

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
        communicationChannel.notifyController();
      }
    }
  } catch (std::exception const& iExcept) {
    std::cerr << "caught exception \n" << iExcept.what() << "\n";
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