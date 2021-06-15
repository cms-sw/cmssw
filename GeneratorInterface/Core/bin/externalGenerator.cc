#include "boost/program_options.hpp"

#include <atomic>
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

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

static char const* const kMemoryNameOpt = "memory-name";
static char const* const kMemoryNameCommandOpt = "memory-name,m";
static char const* const kUniqueIDOpt = "unique-id";
static char const* const kUniqueIDCommandOpt = "unique-id,i";
static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt = "help,h";
static char const* const kVerboseOpt = "verbose";
static char const* const kVerboseCommandOpt = "verbose,v";

//NOTE: Can use TestProcessor as the harness for the worker

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

int main(int argc, char* argv[]) {
  std::string descString(argv[0]);
  descString += " [--";
  descString += kMemoryNameOpt;
  descString += "] memory_name";
  boost::program_options::options_description desc(descString);

  desc.add_options()(kHelpCommandOpt, "produce help message")(
      kMemoryNameCommandOpt,
      boost::program_options::value<std::string>(),
      "memory name")(kUniqueIDCommandOpt, boost::program_options::value<std::string>(), "unique id")(kVerboseCommandOpt,
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

      Serializer<ExternalGeneratorEventInfo> serializer(sm_buffer);
      Serializer<ExternalGeneratorLumiInfo> bl_serializer(sm_buffer);
      Serializer<GenLumiInfoProduct> el_serializer(sm_buffer);
      Serializer<GenRunInfoProduct> er_serializer(sm_buffer);

      ROOTDeserializer<edm::RandomNumberGeneratorState, ReadBuffer> random_deserializer(sm_readbuffer);

      std::cerr << uniqueID << " process: initializing " << std::endl;
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

      edm::ExternalRandomNumberGeneratorService* randomService = new edm::ExternalRandomNumberGeneratorService;
      auto serviceToken =
          edm::ServiceRegistry::createContaining(std::unique_ptr<edm::RandomNumberGenerator>(randomService));
      Harness harness(configuration, serviceToken);

      //Either ROOT or the Framework are overriding the signal handlers
      monitorThread.setupSignalHandling();

      if (verbose) {
        std::cerr << uniqueID << " process: done initializing" << std::endl;
      }
      communicationChannel.workerSetupDone();

      if (verbose)
        std::cerr << uniqueID << " process: waiting " << counter << std::endl;
      communicationChannel.handleTransitions([&](edm::Transition iTransition, unsigned long long iTransitionID) {
        ++counter;
        switch (iTransition) {
          case edm::Transition::BeginRun: {
            if (verbose)
              std::cerr << uniqueID << " process: start beginRun " << std::endl;
            if (verbose)
              std::cerr << uniqueID << " process: end beginRun " << std::endl;

            break;
          }
          case edm::Transition::BeginLuminosityBlock: {
            if (verbose)
              std::cerr << uniqueID << " process: start beginLumi " << std::endl;
            auto randState = random_deserializer.deserialize();
            if (verbose)
              std::cerr << uniqueID << " random " << randState.state_.size() << " " << randState.seed_ << std::endl;
            randomService->setState(randState.state_, randState.seed_);
            auto value = harness.getBeginLumiValue(iTransitionID);
            value.randomState_.state_ = randomService->getState();
            value.randomState_.seed_ = randomService->mySeed();

            bl_serializer.serialize(value);
            if (verbose)
              std::cerr << uniqueID << " process: end beginLumi " << std::endl;
            if (verbose)
              std::cerr << uniqueID << "   rand " << value.randomState_.state_.size() << " " << value.randomState_.seed_
                        << std::endl;
            break;
          }
          case edm::Transition::Event: {
            if (verbose)
              std::cerr << uniqueID << " process: event " << counter << std::endl;
            auto randState = random_deserializer.deserialize();
            randomService->setState(randState.state_, randState.seed_);
            auto value = harness.getEventValue();
            value.randomState_.state_ = randomService->getState();
            value.randomState_.seed_ = randomService->mySeed();

            if (verbose)
              std::cerr << uniqueID << " process: event " << counter << std::endl;

            serializer.serialize(value);
            if (verbose)
              std::cerr << uniqueID << " process: "
                        << " " << counter << std::endl;
            //usleep(10000000);
            break;
          }
          case edm::Transition::EndLuminosityBlock: {
            if (verbose)
              std::cerr << uniqueID << " process: start endLumi " << std::endl;
            auto value = harness.getEndLumiValue();

            el_serializer.serialize(value);
            if (verbose)
              std::cerr << uniqueID << " process: end endLumi " << std::endl;

            break;
          }
          case edm::Transition::EndRun: {
            if (verbose)
              std::cerr << uniqueID << " process: start endRun " << std::endl;
            auto value = harness.getEndRunValue();

            er_serializer.serialize(value);
            if (verbose)
              std::cerr << uniqueID << " process: end endRun " << std::endl;

            break;
          }
          default: {
            assert(false);
          }
        }
        if (verbose)
          std::cerr << uniqueID << " process: notifying and waiting " << counter << std::endl;
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
