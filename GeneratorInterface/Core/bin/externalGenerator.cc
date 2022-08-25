#include "boost/program_options.hpp"

#include <atomic>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>
#include <memory>
#include <filesystem>
#include <ctime>

#include "FWCore/TestProcessor/interface/TestProcessor.h"

#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenRunInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoHeader.h"
#include "SimDataFormats/GeneratorProducts/interface/GenLumiInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/GenEventInfoProduct.h"
#include "SimDataFormats/GeneratorProducts/interface/ExternalGeneratorEventInfo.h"
#include "SimDataFormats/GeneratorProducts/interface/ExternalGeneratorLumiInfo.h"

#include "FWCore/Services/interface/ExternalRandomNumberGeneratorService.h"

#include "FWCore/SharedMemory/interface/WriteBuffer.h"
#include "FWCore/SharedMemory/interface/ReadBuffer.h"
#include "FWCore/SharedMemory/interface/WorkerChannel.h"
#include "FWCore/SharedMemory/interface/ROOTSerializer.h"
#include "FWCore/SharedMemory/interface/ROOTDeserializer.h"
#include "FWCore/SharedMemory/interface/WorkerMonitorThread.h"

#include "FWCore/Utilities/interface/thread_safety_macros.h"

static char const* const kMemoryNameOpt = "memory-name";
static char const* const kMemoryNameCommandOpt = "memory-name,m";
static char const* const kUniqueIDOpt = "unique-id";
static char const* const kUniqueIDCommandOpt = "unique-id,i";
static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt = "help,h";
static char const* const kVerboseOpt = "verbose";
static char const* const kVerboseCommandOpt = "verbose,v";

//This application only uses 1 thread
CMS_THREAD_SAFE static std::string s_uniqueID;

//NOTE: Can use TestProcessor as the harness for the worker

namespace {
  //Based on MessageLogger time handling
  constexpr char timeFormat[] = "dd-Mon-yyyy hh:mm:ss TZN     ";
  constexpr size_t kTimeSize = sizeof(timeFormat);
  std::array<char, kTimeSize> formattedTime() {
    auto t = time(nullptr);
    std::array<char, kTimeSize> ts;

    struct tm timebuf;
    std::strftime(ts.data(), ts.size(), "%d-%b-%Y %H:%M:%S %Z", localtime_r(&t, &timebuf));
    return ts;
  }
}  // namespace

using namespace edm::shared_memory;
class Harness {
public:
  Harness(std::string const& iConfig, edm::ServiceToken iToken)
      : tester_(edm::test::TestProcessor::Config{iConfig}, iToken) {}

  ExternalGeneratorLumiInfo getBeginLumiValue(unsigned int iLumi) {
    auto lumi = tester_.testBeginLuminosityBlock(iLumi);
    ExternalGeneratorLumiInfo returnValue;
    returnValue.header_ = *lumi.get<GenLumiInfoHeader>();
    return returnValue;
  }

  ExternalGeneratorEventInfo getEventValue() {
    ExternalGeneratorEventInfo returnValue;
    auto event = tester_.test();
    returnValue.hepmc_ = *event.get<edm::HepMCProduct>("unsmeared");
    returnValue.eventInfo_ = *event.get<GenEventInfoProduct>();
    returnValue.keepEvent_ = event.modulePassed();
    return returnValue;
  }

  GenLumiInfoProduct getEndLumiValue() {
    auto lumi = tester_.testEndLuminosityBlock();
    return *lumi.get<GenLumiInfoProduct>();
  }

  GenRunInfoProduct getEndRunValue() {
    auto run = tester_.testEndRun();
    return *run.get<GenRunInfoProduct>();
  }

private:
  edm::test::TestProcessor tester_;
};

template <typename T>
using Serializer = ROOTSerializer<T, WriteBuffer>;

namespace {
  //needed for atexit handling
  CMS_THREAD_SAFE boost::interprocess::scoped_lock<boost::interprocess::named_mutex>* s_sharedLock = nullptr;

  void atexit_handler() {
    if (s_sharedLock) {
      std::cerr << s_uniqueID << " process: early exit called: unlock " << formattedTime().data() << "\n";
      s_sharedLock->unlock();
    }
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
      kUniqueIDCommandOpt, boost::program_options::value<std::string>(), "unique id")(kVerboseCommandOpt,
                                                                                      "verbose output");

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

  bool verbose = false;
  if (vm.count(kVerboseOpt)) {
    verbose = true;
  }

  if (!vm.count(kMemoryNameOpt)) {
    std::cout << " no argument given" << std::endl;
    return 1;
  }

  if (!vm.count(kUniqueIDOpt)) {
    std::cout << " no second argument given" << std::endl;
    return 1;
  }

  using namespace std::string_literals;
  using namespace std::filesystem;

  auto newDir = path("thread"s + vm[kUniqueIDOpt].as<std::string>());
  create_directory(newDir);
  current_path(newDir);

  WorkerMonitorThread monitorThread;

  monitorThread.startThread();

  std::string presentState = "setting up communicationChannel";

  CMS_SA_ALLOW try {
    std::string const memoryName(vm[kMemoryNameOpt].as<std::string>());
    std::string const uniqueID(vm[kUniqueIDOpt].as<std::string>());
    s_uniqueID = uniqueID;
    {
      //This class is holding the lock
      WorkerChannel communicationChannel(memoryName, uniqueID);

      presentState = "setting up read/write buffers";
      WriteBuffer sm_buffer{memoryName, communicationChannel.fromWorkerBufferInfo()};
      ReadBuffer sm_readbuffer{std::string("Rand") + memoryName, communicationChannel.toWorkerBufferInfo()};
      int counter = 0;

      presentState = "setting up monitor thread";
      //The lock must be released if there is a catastrophic signal
      auto lockPtr = communicationChannel.accessLock();

      monitorThread.setAction([lockPtr]() {
        if (lockPtr) {
          std::cerr << s_uniqueID << " process: SIGNAL CAUGHT: unlock " << formattedTime().data() << "\n";
          lockPtr->unlock();
        }
      });

      presentState = "setting up termination handler";
      //be sure to unset the address of the shared lock before the lock goes away
      s_sharedLock = lockPtr;
      auto unsetLockPtr = [](void*) { s_sharedLock = nullptr; };
      std::unique_ptr<decltype(s_sharedLock), decltype(unsetLockPtr)> sharedLockGuard{&s_sharedLock, unsetLockPtr};
      std::atexit(atexit_handler);
      auto releaseLock = []() {
        if (s_sharedLock) {
          std::cerr << s_uniqueID << " process: terminate called: unlock " << formattedTime().data() << "\n";
          s_sharedLock->unlock();
          s_sharedLock = nullptr;
          //deactivate the abort signal

          struct sigaction act;
          act.sa_sigaction = nullptr;
          act.sa_flags = SA_SIGINFO;
          sigemptyset(&act.sa_mask);
          sigaction(SIGABRT, &act, nullptr);
          std::abort();
        }
      };
      std::set_terminate(releaseLock);

      presentState = "setting up serializers";
      Serializer<ExternalGeneratorEventInfo> serializer(sm_buffer);
      Serializer<ExternalGeneratorLumiInfo> bl_serializer(sm_buffer);
      Serializer<GenLumiInfoProduct> el_serializer(sm_buffer);
      Serializer<GenRunInfoProduct> er_serializer(sm_buffer);

      ROOTDeserializer<edm::RandomNumberGeneratorState, ReadBuffer> random_deserializer(sm_readbuffer);

      presentState = "reading configuration";
      std::cerr << uniqueID << " process: initializing " << formattedTime().data() << std::endl;
      int nlines;
      std::cin >> nlines;

      std::string configuration;
      for (int i = 0; i < nlines; ++i) {
        std::string c;
        std::getline(std::cin, c);
        if (verbose) {
          std::cerr << c << "\n";
        }
        configuration += c + "\n";
      }

      presentState = "setting up random number generator";
      edm::ExternalRandomNumberGeneratorService* randomService = new edm::ExternalRandomNumberGeneratorService;
      auto serviceToken =
          edm::ServiceRegistry::createContaining(std::unique_ptr<edm::RandomNumberGenerator>(randomService));
      Harness harness(configuration, serviceToken);

      //Some generator libraries override the signal handlers
      monitorThread.setupSignalHandling();
      std::set_terminate(releaseLock);

      if (verbose) {
        std::cerr << uniqueID << " process: done initializing " << formattedTime().data() << std::endl;
      }
      presentState = "finished initialization";
      communicationChannel.workerSetupDone();

      presentState = "waiting for transition";
      if (verbose)
        std::cerr << uniqueID << " process: waiting " << counter << " " << formattedTime().data() << std::endl;
      communicationChannel.handleTransitions([&](edm::Transition iTransition, unsigned long long iTransitionID) {
        ++counter;
        switch (iTransition) {
          case edm::Transition::BeginRun: {
            presentState = "beginRun transition";
            if (verbose)
              std::cerr << uniqueID << " process: start beginRun " << formattedTime().data() << std::endl;
            if (verbose)
              std::cerr << uniqueID << " process: end beginRun " << formattedTime().data() << std::endl;

            break;
          }
          case edm::Transition::BeginLuminosityBlock: {
            presentState = "begin lumi";
            if (verbose)
              std::cerr << uniqueID << " process: start beginLumi " << formattedTime().data() << std::endl;
            auto randState = random_deserializer.deserialize();
            presentState = "deserialized random state in begin lumi";
            if (verbose)
              std::cerr << uniqueID << " random " << randState.state_.size() << " " << randState.seed_ << std::endl;
            randomService->setState(randState.state_, randState.seed_);
            presentState = "processing begin lumi";
            auto value = harness.getBeginLumiValue(iTransitionID);
            value.randomState_.state_ = randomService->getState();
            value.randomState_.seed_ = randomService->mySeed();

            presentState = "serialize lumi";
            bl_serializer.serialize(value);
            if (verbose)
              std::cerr << uniqueID << " process: end beginLumi " << formattedTime().data() << std::endl;
            if (verbose)
              std::cerr << uniqueID << "   rand " << value.randomState_.state_.size() << " " << value.randomState_.seed_
                        << std::endl;
            break;
          }
          case edm::Transition::Event: {
            presentState = "begin event";
            if (verbose)
              std::cerr << uniqueID << " process: event " << counter << " " << formattedTime().data() << std::endl;
            presentState = "deserialized random state in event";
            auto randState = random_deserializer.deserialize();
            randomService->setState(randState.state_, randState.seed_);
            presentState = "processing event";
            auto value = harness.getEventValue();
            value.randomState_.state_ = randomService->getState();
            value.randomState_.seed_ = randomService->mySeed();

            if (verbose)
              std::cerr << uniqueID << " process: event " << counter << " " << formattedTime().data() << std::endl;

            presentState = "serialize event";
            serializer.serialize(value);
            if (verbose)
              std::cerr << uniqueID << " process: "
                        << " " << counter << std::endl;
            //usleep(10000000);
            break;
          }
          case edm::Transition::EndLuminosityBlock: {
            presentState = "begin end lumi";
            if (verbose)
              std::cerr << uniqueID << " process: start endLumi " << formattedTime().data() << std::endl;
            presentState = "processing end lumi";
            auto value = harness.getEndLumiValue();

            presentState = "serialize end lumi";
            el_serializer.serialize(value);
            if (verbose)
              std::cerr << uniqueID << " process: end endLumi " << formattedTime().data() << std::endl;

            break;
          }
          case edm::Transition::EndRun: {
            presentState = "begin end run";
            if (verbose)
              std::cerr << uniqueID << " process: start endRun " << formattedTime().data() << std::endl;
            presentState = "process end run";
            auto value = harness.getEndRunValue();

            presentState = "serialize end run";
            er_serializer.serialize(value);
            if (verbose)
              std::cerr << uniqueID << " process: end endRun " << formattedTime().data() << std::endl;

            break;
          }
          default: {
            assert(false);
          }
        }
        presentState = "notifying and waiting after " + presentState;
        if (verbose)
          std::cerr << uniqueID << " process: notifying and waiting " << counter << " " << std::endl;
      });
    }
  } catch (std::exception const& iExcept) {
    std::cerr << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
              << s_uniqueID << " process: caught exception \n"
              << iExcept.what() << " " << formattedTime().data() << "\n"
              << "  while " << presentState << "\n"
              << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    return 1;
  } catch (...) {
    std::cerr << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n"
              << s_uniqueID << " process: caught unknown exception " << formattedTime().data() << "\n  while "
              << presentState << "\n"
              << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n";
    return 1;
  }
  return 0;
}
