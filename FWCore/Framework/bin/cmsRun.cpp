/*----------------------------------------------------------------------
This is a generic main that can be used with any plugin and a
PSet script.   See notes in EventProcessor.cpp for details about it.
----------------------------------------------------------------------*/

#include "FWCore/AbstractServices/interface/TimingServiceBase.h"
#include "FWCore/Concurrency/interface/ThreadsController.h"
#include "FWCore/Concurrency/interface/setNThreads.h"
#include "FWCore/Framework/interface/CmsRunParser.h"
#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Framework/interface/defaultCmsRunServices.h"
#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/Entry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/ThreadsInfo.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/ServiceRegistry/interface/ServiceWrapper.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include "PyLikeParser.h"

#include <TError.h>

#include <oneapi/tbb/task_arena.h>

#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <string>
#include <vector>

// -----------------------------------------------
namespace {
  class EventProcessorWithSentry {
  public:
    explicit EventProcessorWithSentry() : ep_(nullptr), callEndJob_(false) {}
    explicit EventProcessorWithSentry(std::unique_ptr<edm::EventProcessor> ep)
        : ep_(std::move(ep)), callEndJob_(false) {}
    ~EventProcessorWithSentry() {
      if (callEndJob_ && ep_.get()) {
        //  See the message in catch clause
        CMS_SA_ALLOW try { ep_->endJob(); } catch (...) {
          edm::LogSystem("MoreExceptions")
              << "After a fatal primary exception was caught, there was an attempt to run\n"
              << "endJob methods. Another exception was caught while endJob was running\n"
              << "and we give up trying to run endJob.";
        }
      }
      edm::clearMessageLog();
    }
    EventProcessorWithSentry(EventProcessorWithSentry const&) = delete;
    EventProcessorWithSentry const& operator=(EventProcessorWithSentry const&) = delete;
    EventProcessorWithSentry(EventProcessorWithSentry&&) = default;             // Allow Moving
    EventProcessorWithSentry& operator=(EventProcessorWithSentry&&) = default;  // Allow moving

    void on() { callEndJob_ = true; }
    void off() { callEndJob_ = false; }
    edm::EventProcessor* operator->() { return ep_.get(); }
    edm::EventProcessor* get() { return ep_.get(); }

  private:
    std::unique_ptr<edm::EventProcessor> ep_;
    bool callEndJob_;
  };

  class TaskCleanupSentry {
  public:
    TaskCleanupSentry(edm::EventProcessor* ep) : ep_(ep) {}
    ~TaskCleanupSentry() { ep_->taskCleanup(); }

  private:
    edm::EventProcessor* ep_;
  };
}  // namespace

static std::string trim(std::string arg) {
  size_t first = arg.find_first_not_of(' ');
  size_t last = arg.find_last_not_of(' ');
  if (first == std::string::npos) {
    return "";
  } else {
    return arg.substr(first, (last - first + 1));
  }
}

void parseProcessOption(edm::ParameterSet& options, std::string const& name, std::string const& value) {
  if (options.exists(name)) {
    // update the options from the arguments
    edm::Entry const* entry = options.retrieveUntracked(name);
    edm::PyLikeParser parser(value);
    if (entry->typeName() == "bool") {
      options.addUntrackedParameter<bool>(name, parser.parse<bool>());
    } else if (entry->typeName() == "int32") {
      options.addUntrackedParameter<int>(name, parser.parse<int>());
    } else if (entry->typeName() == "uint32") {
      options.addUntrackedParameter<unsigned int>(name, parser.parse<unsigned int>());
    } else if (entry->typeName() == "int64") {
      options.addUntrackedParameter<long long>(name, parser.parse<long long>());
    } else if (entry->typeName() == "uint64") {
      options.addUntrackedParameter<unsigned long long>(name, parser.parse<unsigned long long>());
      /* float arguments are not yet supported by CMSSW
    } else if (entry->typeName() == "float") {
      options.addUntrackedParameter<float>(name, parser.parse<float>());
      */
    } else if (entry->typeName() == "double") {
      options.addUntrackedParameter<double>(name, parser.parse<double>());
    } else if (entry->typeName() == "string") {
      options.addUntrackedParameter<std::string>(name, parser.parse<std::string>());
      /* vbool arguments are not supported by CMSSW
    } else if (entry->typeName() == "vbool") {
      options.addUntrackedParameter<std::vector<bool>>(name, parser.parse_vector<bool>());
      */
    } else if (entry->typeName() == "vint32") {
      options.addUntrackedParameter<std::vector<int>>(name, parser.parse_vector<int>());
    } else if (entry->typeName() == "vuint32") {
      options.addUntrackedParameter<std::vector<unsigned int>>(name, parser.parse_vector<unsigned int>());
    } else if (entry->typeName() == "vint64") {
      options.addUntrackedParameter<std::vector<long long>>(name, parser.parse_vector<long long>());
    } else if (entry->typeName() == "vuint64") {
      options.addUntrackedParameter<std::vector<unsigned long long>>(name, parser.parse_vector<unsigned long long>());
      /* float arguments are not yet supported by CMSSW
    } else if (entry->typeName() == "vfloat") {
      options.addUntrackedParameter<std::vector<float>>(name, parser.parse_vector<float>());
      */
    } else if (entry->typeName() == "vdouble") {
      options.addUntrackedParameter<std::vector<double>>(name, parser.parse_vector<double>());
    } else if (entry->typeName() == "vstring") {
      options.addUntrackedParameter<std::vector<std::string>>(name, parser.parse_vector<std::string>());
    } else {
      std::cerr << "cmsRun: error: unsupported option type \"" << entry->typeName() << "\".\n";
      std::exit(static_cast<int>(edm::errors::CommandLineProcessing));
    }
  } else if (name == "sizeOfStackForThreadsInKB") {
    // sizeOfStackForThreadsInKB is an optional parameter, so it may be missing from process.options;
    // see Process.defaultOptions_() in FWCore/ParameterSet/python/Config.py
    options.addUntrackedParameter<uint32_t>("sizeOfStackForThreadsInKB", std::stoul(value));
  } else {
    // "name" is not a valid process option
    std::cerr << "cmsRun: error: \"" << name << "\" is not a valid process option.\n";
    std::exit(static_cast<int>(edm::errors::CommandLineProcessing));
  }
}

void setProcessOption(edm::ParameterSet& process, std::string const& name, std::string const& value) {
  // read the options from the configuration
  edm::ParameterSet options = process.getUntrackedParameterSet("options");

  parseProcessOption(options, name, value);

  // update the configuration with the updated options
  process.insertParameterSet(true, "options", edm::ParameterSetEntry(options, false));
}

void setProcessOptions(edm::ParameterSet& process, std::unordered_map<std::string, std::string> const& values) {
  // read the options from the configuration
  edm::ParameterSet options = process.getUntrackedParameterSet("options");

  for (auto& [name, value] : values) {
    parseProcessOption(options, name, value);
  }

  // update the configuration with the updated options
  process.insertParameterSet(true, "options", edm::ParameterSetEntry(options, false));
}

int main(int argc, const char* argv[]) {
  edm::TimingServiceBase::jobStarted();

  int returnCode = 0;
  std::string context;
  bool alwaysAddContext = true;
  //Default to only use 1 thread. We define this early (before parsing the command line options
  // and python configuration) since the plugin system or message logger may be using TBB.
  unsigned int nThreads = edm::s_defaultNumberOfThreads;
  auto tsiPtr = std::make_unique<edm::ThreadsController>(nThreads, edm::s_defaultSizeOfStackForThreadsInKB * 1024);
  std::shared_ptr<edm::Presence> theMessageServicePresence;
  std::unique_ptr<std::ofstream> jobReportStreamPtr;
  std::shared_ptr<edm::serviceregistry::ServiceWrapper<edm::JobReport>> jobRep;
  EventProcessorWithSentry proc;

  try {
    returnCode = edm::convertException::wrap([&]() -> int {

    // NOTE: MacOs X has a lower rlimit for opened file descriptor than Linux (256
    // in Snow Leopard vs 512 in SLC5). This is a problem for some of the workflows
    // that open many small root datafiles.  Notice that this is safe to do also
    // for Linux, but we agreed not to change the behavior there for the moment.
    // Also the limits imposed by ulimit are not affected and still apply, if
    // there.
#ifdef __APPLE__
      context = "Setting file descriptor limit";
      struct rlimit limits;
      getrlimit(RLIMIT_NOFILE, &limits);
      limits.rlim_cur = (OPEN_MAX < limits.rlim_max) ? OPEN_MAX : limits.rlim_max;
      setrlimit(RLIMIT_NOFILE, &limits);
#endif

      context = "Initializing plug-in manager";
      edmplugin::PluginManager::configure(edmplugin::standard::config());

      context = "Initializing message logger";
      // Load the message service plug-in
      theMessageServicePresence =
          std::shared_ptr<edm::Presence>(edm::PresenceFactory::get()->makePresence("SingleThreadMSPresence").release());

      context = "Processing command line arguments";
      edm::CmsRunParser parser(argv[0]);

      // Parse the command line options
      const auto& parserOutput = parser.parse(argc, argv);

      // If there was an error, exit with the error code from the parser
      if (not parserOutput.has_value()) {
        return parserOutput.error();
      }
      auto vm = parserOutput.value();

      std::string cmdString;
      std::string fileName;
      if (vm.count(edm::CmsRunParser::kCmdOpt)) {
        cmdString = vm[edm::CmsRunParser::kCmdOpt].as<std::string>();
        if (vm.count(edm::CmsRunParser::kParameterSetOpt)) {
          edm::LogAbsolute("CommandLineProcessing") << "cmsRun: Error while trying to process command line arguments:\n"
                                                    << "cannot use '-c [command line input]' with 'config_file'\n"
                                                    << "For usage and an options list, please do 'cmsRun --help'.";
          edm::HaltMessageLogging();
          return edm::errors::CommandLineProcessing;
        }
      } else if (!vm.count(edm::CmsRunParser::kParameterSetOpt)) {
        edm::LogAbsolute("ConfigFileNotFound") << "cmsRun: No configuration file given.\n"
                                               << "For usage and an options list, please do 'cmsRun --help'.";
        edm::HaltMessageLogging();
        return edm::errors::ConfigFileNotFound;
      } else
        fileName = vm[edm::CmsRunParser::kParameterSetOpt].as<std::string>();
      std::vector<std::string> pythonOptValues;
      if (vm.count(edm::CmsRunParser::kPythonOpt)) {
        pythonOptValues = vm[edm::CmsRunParser::kPythonOpt].as<std::vector<std::string>>();
      }
      pythonOptValues.insert(pythonOptValues.begin(), fileName);

      if (vm.count(edm::CmsRunParser::kStrictOpt)) {
        //edm::setStrictParsing(true);
        edm::LogSystem("CommandLineProcessing") << "Strict configuration processing is now done from python";
      }

      context = "Creating the JobReport Service";
      // Decide whether to enable creation of job report xml file
      //  We do this first so any errors will be reported
      std::string jobReportFile;
      if (vm.count(edm::CmsRunParser::kJobreportOpt)) {
        jobReportFile = vm[edm::CmsRunParser::kJobreportOpt].as<std::string>();
      } else if (vm.count(edm::CmsRunParser::kEnableJobreportOpt)) {
        jobReportFile = "FrameworkJobReport.xml";
      }
      jobReportStreamPtr = jobReportFile.empty() ? nullptr : std::make_unique<std::ofstream>(jobReportFile.c_str());

      //NOTE: JobReport must have a lifetime shorter than jobReportStreamPtr so that when the JobReport destructor
      // is called jobReportStreamPtr is still valid
      auto jobRepPtr = std::make_unique<edm::JobReport>(jobReportStreamPtr.get());
      jobRep = std::make_shared<edm::serviceregistry::ServiceWrapper<edm::JobReport>>(std::move(jobRepPtr));
      edm::ServiceToken jobReportToken = edm::ServiceRegistry::createContaining(jobRep);

      if (!fileName.empty()) {
        context = "Processing the python configuration file named ";
        context += fileName;
      } else {
        context = "Processing the python configuration from command line ";
        context += cmdString;
      }
      std::shared_ptr<edm::ProcessDesc> processDesc;
      try {
        edm::TimingServiceBase::pythonStarting();
        std::unique_ptr<edm::ParameterSet> parameterSet;
        if (!fileName.empty())
          parameterSet = edm::readConfig(fileName, pythonOptValues);
        else
          edm::makeParameterSets(cmdString, parameterSet);
        edm::TimingServiceBase::pythonFinished();
        processDesc = std::make_shared<edm::ProcessDesc>(std::move(parameterSet));
      } catch (edm::Exception const&) {
        throw;
      } catch (cms::Exception& iException) {
        //check for "SystemExit: 0" on second line
        const std::string& sysexit0("SystemExit: 0");
        const auto& msg = iException.message();
        size_t pos2 = msg.find('\n');
        if (pos2 != std::string::npos and (msg.size() - (pos2 + 1)) > sysexit0.size() and
            msg.compare(pos2 + 1, sysexit0.size(), sysexit0) == 0)
          return 0;

        edm::Exception e(edm::errors::ConfigFileReadError, "", iException);
        throw e;
      }

      // Determine the number of threads to use, and the per-thread stack size.
      //
      // First, update the top level "options" ParameterSet based on
      //   - the -o/--option command line options
      //   - the -n/--numThreads and -s/--sizeOfStackForThreadsInKB obsolete command line options

      // Parse the -o/--option command line options
      std::unordered_map<std::string, std::string> processOptions;
      if (vm.count(edm::CmsRunParser::kOptionOpt)) {
        auto const& options = vm[edm::CmsRunParser::kOptionOpt].as<std::vector<std::string>>();
        for (auto const& option : options) {
          auto idx = option.find_first_of('=');
          if (idx == option.npos) {
            std::cerr << "cmsRun: error: invalid option '" << option
                      << "', missing '=' separator between name and value.\n";
            std::exit(static_cast<int>(edm::errors::CommandLineProcessing));
          }
          auto name = trim(option.substr(0, idx));
          auto value = trim(option.substr(idx + 1));
          processOptions[std::move(name)] = std::move(value);
        }
      }

      // Parse the -n/--numThreads and -s/--sizeOfStackForThreadsInKB obsolete command line options
      if (vm.count(edm::CmsRunParser::kNumberOfThreadsOpt)) {
        /*
        std::cerr << "cmsRun: the option\n"
                  << "  cmsRun -n THREADS\n"
                  << "is deprecated; please use\n"
                  << "  cmsRun -o numberOfThreads=THREADS\n";
        */
        processOptions["numberOfThreads"] =
            std::to_string(vm[edm::CmsRunParser::kNumberOfThreadsOpt].as<unsigned int>());
      }
      if (vm.count(edm::CmsRunParser::kSizeOfStackForThreadOpt)) {
        /*
        std::cerr << "cmsRun: the option\n"
                  << "  cmsRun -s SIZE\n"
                  << "is deprecated; please use\n"
                  << "  cmsRun -o sizeOfStackForThreadsInKB=SIZE\n";
        */
        processOptions["sizeOfStackForThreadsInKB"] =
            std::to_string(vm[edm::CmsRunParser::kSizeOfStackForThreadOpt].as<unsigned int>());
      }

      // Update the process.options ParameterSet from the command line options
      std::shared_ptr<edm::ParameterSet> pset = processDesc->getProcessPSet();
      setProcessOptions(*pset, processOptions);

      // Then, determine the number of threads to use, and the per-thread stack size,
      // based on the default values (currently, 1 thread and 10 MB) and the updated
      // "options" top level ParameterSet.
      context = "Setting up number of threads";

      // Since TBB has already been initialised with the default values, re-initialise
      // it only if different values are discovered.
      {
        auto threadsInfo = threadOptions(*pset);
        // if needed, re-initialise TBB
        if (threadsInfo.nThreads_ != nThreads or threadsInfo.stackSize_ != edm::s_defaultSizeOfStackForThreadsInKB) {
          nThreads = edm::setNThreads(threadsInfo.nThreads_, threadsInfo.stackSize_, tsiPtr);
        }
        // if the process was configured to let TBB chose the number of threads, update
        // the configuration with the chosen value.
        if (threadsInfo.nThreads_ == 0) {
          setProcessOption(*pset, "numberOfThreads", std::to_string(nThreads));
        }
      }

      context = "Initializing default service configurations";
      // Default parameters will be used for the default services
      // if they are not overridden from the configuration files.
      processDesc->addServices(edm::defaultCmsRunServices());

      context = "Setting MessageLogger defaults";
      // Decide what mode of hardcoded MessageLogger defaults to use
      if (vm.count(edm::CmsRunParser::kJobModeOpt)) {
        std::string jobMode = vm[edm::CmsRunParser::kJobModeOpt].as<std::string>();
        edm::MessageDrop::instance()->jobMode = jobMode;
      }

      oneapi::tbb::task_arena arena(nThreads);
      arena.execute([&]() {
        context = "Constructing the EventProcessor";
        EventProcessorWithSentry procTmp(
            std::make_unique<edm::EventProcessor>(processDesc, jobReportToken, edm::serviceregistry::kTokenOverrides));
        proc = std::move(procTmp);
        TaskCleanupSentry sentry{proc.get()};

        alwaysAddContext = false;

        proc.on();
        context = "Calling beginJob";
        proc->beginJob();

        processDesc.reset();

        context =
            "Calling EventProcessor::runToCompletion (which does almost everything after beginJob and before endJob)";
        auto status = proc->runToCompletion();
        if (status == edm::EventProcessor::epSignal) {
          returnCode = edm::errors::CaughtSignal;
        }

        proc.off();
        context = "Calling endJob and endStream";
        proc->endJob();
      });
      return returnCode;
    });
  }
  // All exceptions which are not handled before propagating
  // into main will get caught here.
  catch (cms::Exception& ex) {
    returnCode = ex.returnCode();
    if (!context.empty()) {
      if (alwaysAddContext) {
        ex.addContext(context);
      } else if (ex.context().empty()) {
        ex.addContext(context);
      }
    }
    if (!ex.alreadyPrinted()) {
      if (jobRep.get() != nullptr) {
        edm::printCmsException(ex, &(jobRep->get()), returnCode);
      } else {
        edm::printCmsException(ex);
      }
    }
  }
  // Disable Root Error Handler.
  SetErrorHandler(DefaultErrorHandler);
  return returnCode;
}
