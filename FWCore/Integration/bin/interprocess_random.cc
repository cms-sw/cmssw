#include "boost/program_options.hpp"

#include <atomic>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

#include "FWCore/TestProcessor/interface/TestProcessor.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "DataFormats/Common/interface/RandomNumberGeneratorState.h"

#include "FWCore/Services/interface/ExternalRandomNumberGeneratorService.h"

#include "FWCore/SharedMemory/interface/WriteBuffer.h"
#include "FWCore/SharedMemory/interface/WorkerChannel.h"
#include "FWCore/SharedMemory/interface/ROOTSerializer.h"
#include "FWCore/SharedMemory/interface/ReadBuffer.h"
#include "FWCore/SharedMemory/interface/ROOTDeserializer.h"
#include "FWCore/SharedMemory/interface/WorkerMonitorThread.h"

static char const* const kMemoryNameOpt = "memory-name";
static char const* const kMemoryNameCommandOpt = "memory-name,m";
static char const* const kUniqueIDOpt = "unique-id";
static char const* const kUniqueIDCommandOpt = "unique-id,i";
static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt = "help,h";

//NOTE: Can use TestProcessor as the harness for the worker

using SentType = std::pair<edmtest::IntProduct, edm::RandomNumberGeneratorState>;

using namespace edm::shared_memory;
class Harness {
public:
  Harness(std::string const& iConfig, edm::ServiceToken iToken)
      : tester_(edm::test::TestProcessor::Config{iConfig}, iToken) {}

  edmtest::IntProduct getBeginLumiValue(unsigned int iLumi) {
    auto lumi = tester_.testBeginLuminosityBlock(iLumi);
    return *lumi.get<edmtest::IntProduct>("lumi");
  }

  edmtest::IntProduct getEventValue() {
    auto event = tester_.test();
    return *event.get<edmtest::IntProduct>();
  }

private:
  edm::test::TestProcessor tester_;
};

int main(int argc, char* argv[]) {
  std::string descString(argv[0]);
  descString += " [--";
  descString += kMemoryNameOpt;
  descString += "] memory_name";
  boost::program_options::options_description desc(descString);

  desc.add_options()(kHelpCommandOpt, "produce help message")(
      kMemoryNameCommandOpt,
      boost::program_options::value<std::string>(),
      "memory name")(kUniqueIDCommandOpt, boost::program_options::value<std::string>(), "unique id");

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

  WorkerMonitorThread monitorThread;

  monitorThread.startThread();

  CMS_SA_ALLOW try {
    std::string const memoryName(vm[kMemoryNameOpt].as<std::string>());
    std::string const uniqueID(vm[kUniqueIDOpt].as<std::string>());
    {
      //This class is holding the lock
      WorkerChannel communicationChannel(memoryName, uniqueID);

      WriteBuffer sm_buffer{memoryName, communicationChannel.fromWorkerBufferInfo()};
      ReadBuffer sm_readbuffer{std::string("Rand") + memoryName, communicationChannel.toWorkerBufferInfo()};
      int counter = 0;

      //The lock must be released if there is a catastrophic signal
      auto lockPtr = communicationChannel.accessLock();
      monitorThread.setAction([lockPtr]() {
        if (lockPtr) {
          std::cerr << "SIGNAL CAUGHT: unlock\n";
          lockPtr->unlock();
        }
      });

      using TCSerializer = ROOTSerializer<SentType, WriteBuffer>;
      TCSerializer serializer(sm_buffer);
      TCSerializer bl_serializer(sm_buffer);

      using TCDeserializer = ROOTDeserializer<edm::RandomNumberGeneratorState, ReadBuffer>;
      TCDeserializer random_deserializer(sm_readbuffer);

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

      edm::ExternalRandomNumberGeneratorService* randomService = new edm::ExternalRandomNumberGeneratorService;
      auto serviceToken =
          edm::ServiceRegistry::createContaining(std::unique_ptr<edm::RandomNumberGenerator>(randomService));

      Harness harness(configuration, serviceToken);

      //Either ROOT or the Framework are overriding the signal handlers
      monitorThread.setupSignalHandling();

      std::cerr << uniqueID << " process: done initializing" << std::endl;
      communicationChannel.workerSetupDone();

      std::cerr << uniqueID << " process: waiting " << counter << std::endl;
      communicationChannel.handleTransitions([&](edm::Transition iTransition, unsigned long long iTransitionID) {
        ++counter;
        switch (iTransition) {
          case edm::Transition::BeginLuminosityBlock: {
            std::cerr << uniqueID << " process: start beginLumi " << std::endl;
            auto randState = random_deserializer.deserialize();
            std::cerr << " state " << randState.seed_ << std::endl;
            randomService->setState(randState.state_, randState.seed_);
            SentType toSend;
            toSend.first = harness.getBeginLumiValue(iTransitionID);
            toSend.second.state_ = randomService->getState();
            toSend.second.seed_ = randomService->mySeed();
            bl_serializer.serialize(toSend);
            std::cerr << uniqueID << " process: end beginLumi " << toSend.first.value << std::endl;

            break;
          }
          case edm::Transition::Event: {
            std::cerr << uniqueID << " process: begin event " << counter << std::endl;
            auto randState = random_deserializer.deserialize();
            randomService->setState(randState.state_, randState.seed_);
            SentType toSend;
            toSend.first = harness.getEventValue();
            toSend.second.state_ = randomService->getState();
            toSend.second.seed_ = randomService->mySeed();
            std::cerr << uniqueID << " process: end event " << counter << std::endl;

            serializer.serialize(toSend);
            std::cerr << uniqueID << " process: " << toSend.first.value << " " << counter << std::endl;
            //usleep(10000000);
            break;
          }
          default: {
            assert(false);
          }
        }
        std::cerr << uniqueID << " process: notifying and waiting" << counter << std::endl;
      });
    }
  } catch (std::exception const& iExcept) {
    std::cerr << "caught exception \n" << iExcept.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "caught unknown exception";
    return 1;
  }
  return 0;
}
