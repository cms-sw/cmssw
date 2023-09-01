/*----------------------------------------------------------------------
This is a generic main that can be used with any plugin and a
PSet script.   See notes in EventProcessor.cpp for details about it.
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Framework/interface/defaultCmsRunServices.h"
#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/ParameterSet/interface/ThreadsInfo.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/ParameterSetReader/interface/ParameterSetReader.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/ServiceRegistry/interface/ServiceWrapper.h"
#include "FWCore/Concurrency/interface/setNThreads.h"
#include "FWCore/Concurrency/interface/ThreadsController.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/Utilities/interface/TimingServiceBase.h"
#include "FWCore/Utilities/interface/thread_safety_macros.h"

#include "TError.h"

#include "boost/program_options.hpp"
#include "oneapi/tbb/task_arena.h"

#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

//Command line parameters
static char const* const kParameterSetOpt = "parameter-set";
static char const* const kPythonOpt = "pythonOptions";
static std::string const kPythonOptDefault = "CMSRUN_PYTHONOPT_DEFAULT";
static char const* const kCmdCommandOpt = "command,c";
static char const* const kCmdOpt = "command";
static char const* const kJobreportCommandOpt = "jobreport,j";
static char const* const kJobreportOpt = "jobreport";
static char const* const kEnableJobreportCommandOpt = "enablejobreport,e";
static const char* const kEnableJobreportOpt = "enablejobreport";
static char const* const kJobModeCommandOpt = "mode,m";
static char const* const kJobModeOpt = "mode";
static char const* const kNumberOfThreadsCommandOpt = "numThreads,n";
static char const* const kNumberOfThreadsOpt = "numThreads";
static char const* const kSizeOfStackForThreadCommandOpt = "sizeOfStackForThreadsInKB,s";
static char const* const kSizeOfStackForThreadOpt = "sizeOfStackForThreadsInKB";
static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt = "help,h";
static char const* const kStrictOpt = "strict";

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

  //additional_parser is documented at https://www.boost.org/doc/libs/1_83_0/doc/html/program_options/howto.html#id-1.3.30.6.3
  //extra_style_parser was added in https://www.boost.org/users/history/version_1_33_0.html
  //it allows performing a fully custom processing on whatever arguments have not yet been processed
  //the function is responsible for removing processed arguments from the input vector
  //and assembling the return value, which is a vector of boost's option type
  //some usage examples: https://stackoverflow.com/a/5481228, https://stackoverflow.com/a/37993517
  //internally, when cmdline::run() is called, the function given to extra_style_parser is added to the beginning of list of parsers
  //all parsers are called in order until args is empty (validity of non-empty output is checked after each parser)
  std::vector<boost::program_options::option> final_opts_parser(std::vector<std::string>& args) {
    std::vector<boost::program_options::option> result;
    std::string configName;
    if (!args.empty() and !args[0].empty()) {
      if(args[0][0]!='-'){ // name is first positional arg -> doesn't start with '-'
        configName = args[0];
        args.erase(args.begin());
      }
      else if(args[0]=="--" and args.size()>1){ // name can start with '-' if separator comes first
        configName = args[1];
        args.erase(args.begin(),args.begin()+2);
      }
    }
    if (!configName.empty()) {
      result.emplace_back(kParameterSetOpt, std::vector<std::string>(1, configName));
      result.emplace_back();
      auto& pythonOpts = result.back();
      pythonOpts.string_key = kPythonOpt;
      pythonOpts.value.reserve(args.size());
      pythonOpts.original_tokens.reserve(args.size());
      for (const auto& arg : args) {
        pythonOpts.value.push_back(arg);
        pythonOpts.original_tokens.push_back(arg);
      }
      //default value to avoid "is missing" error
      if (pythonOpts.value.empty()) {
        pythonOpts.value.push_back(kPythonOptDefault);
        pythonOpts.original_tokens.push_back("");
      }
      args.clear();
    }
    return result;
  }
}  // namespace

int main(int argc, char* argv[]) {
  edm::TimingServiceBase::jobStarted();

  int returnCode = 0;
  std::string context;
  bool alwaysAddContext = true;
  //Default to only use 1 thread. We define this early (before parsing the command line options
  // and python configuration) since the plugin system or message logger may be using TBB.
  auto tsiPtr = std::make_unique<edm::ThreadsController>(edm::s_defaultNumberOfThreads,
                                                         edm::s_defaultSizeOfStackForThreadsInKB * 1024);
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
      std::string descString(argv[0]);
      descString += " [options] config_file [python options]\nAllowed options";
      boost::program_options::options_description desc(descString);

      // clang-format off
      desc.add_options()(kHelpCommandOpt, "produce help message")(
          kJobreportCommandOpt,
          boost::program_options::value<std::string>(),
          "file name to use for a job report file: default extension is .xml")(
          kEnableJobreportCommandOpt, "enable job report files (if any) specified in configuration file")(
          kJobModeCommandOpt,
          boost::program_options::value<std::string>(),
          "Job Mode for MessageLogger defaults - default mode is grid")(
          kNumberOfThreadsCommandOpt,
          boost::program_options::value<unsigned int>(),
          "Number of threads to use in job (0 is use all CPUs)")(
          kSizeOfStackForThreadCommandOpt,
          boost::program_options::value<unsigned int>(),
          "Size of stack in KB to use for extra threads (0 is use system default size)")(kStrictOpt, "strict parsing")(
          kCmdCommandOpt, boost::program_options::value<std::string>(), "config passed in as string (config_file [python options] will be ignored)");
      // clang-format on

      // anything at the end will be ignored, and sent to python
      boost::program_options::positional_options_description p;
      p.add(kParameterSetOpt, 1).add(kPythonOpt, -1);

      // This --fwk option is not used anymore, but I'm leaving it around as
      // it might be useful again in the future for code development
      // purposes.  We originally used it when implementing the boost
      // state machine code.
      boost::program_options::options_description hidden("hidden options");
      hidden.add_options()("fwk", "For use only by Framework Developers")(
          kParameterSetOpt, boost::program_options::value<std::string>(), "configuration file")(
          kPythonOpt,
          boost::program_options::value<std::vector<std::string>>(),
          "options at the end to be passed to python");

      boost::program_options::options_description all_options("All Options");
      all_options.add(desc).add(hidden);

      boost::program_options::variables_map vm;
      try {
        store(boost::program_options::command_line_parser(argc, argv)
                  .extra_style_parser(final_opts_parser)
                  .options(all_options)
                  .positional(p)
                  .run(),
              vm);
        notify(vm);
      } catch (boost::program_options::error const& iException) {
        edm::LogAbsolute("CommandLineProcessing")
            << "cmsRun: Error while trying to process command line arguments:\n"
            << iException.what() << "\nFor usage and an options list, please do 'cmsRun --help'.";
        return edm::errors::CommandLineProcessing;
      }

      if (vm.count(kHelpOpt)) {
        std::cout << desc << std::endl;
        if (!vm.count(kParameterSetOpt))
          edm::HaltMessageLogging();
        return 0;
      }

      std::string cmdString;
      std::string fileName;
      if (vm.count(kCmdOpt))
        cmdString = vm[kCmdOpt].as<std::string>();
      else if (!vm.count(kParameterSetOpt)) {
        edm::LogAbsolute("ConfigFileNotFound") << "cmsRun: No configuration file given.\n"
                                               << "For usage and an options list, please do 'cmsRun --help'.";
        edm::HaltMessageLogging();
        return edm::errors::ConfigFileNotFound;
      }
      else
        fileName = vm[kParameterSetOpt].as<std::string>();
      std::vector<std::string> pythonOptValues;
      if (vm.count(kPythonOpt)) {
        pythonOptValues = vm[kPythonOpt].as<std::vector<std::string>>();
        //omit default arg
        if (pythonOptValues.size() == 1 and pythonOptValues[0] == kPythonOptDefault)
          pythonOptValues.clear();
      }
      pythonOptValues.insert(pythonOptValues.begin(), fileName);

      if (vm.count(kStrictOpt)) {
        //edm::setStrictParsing(true);
        edm::LogSystem("CommandLineProcessing") << "Strict configuration processing is now done from python";
      }

      context = "Creating the JobReport Service";
      // Decide whether to enable creation of job report xml file
      //  We do this first so any errors will be reported
      std::string jobReportFile;
      if (vm.count(kJobreportOpt)) {
        jobReportFile = vm[kJobreportOpt].as<std::string>();
      } else if (vm.count(kEnableJobreportOpt)) {
        jobReportFile = "FrameworkJobReport.xml";
      }
      jobReportStreamPtr = jobReportFile.empty() ? nullptr : std::make_unique<std::ofstream>(jobReportFile.c_str());

      //NOTE: JobReport must have a lifetime shorter than jobReportStreamPtr so that when the JobReport destructor
      // is called jobReportStreamPtr is still valid
      auto jobRepPtr = std::make_unique<edm::JobReport>(jobReportStreamPtr.get());
      jobRep.reset(new edm::serviceregistry::ServiceWrapper<edm::JobReport>(std::move(jobRepPtr)));
      edm::ServiceToken jobReportToken = edm::ServiceRegistry::createContaining(jobRep);

      context = "Processing the python configuration file named ";
      context += !fileName.empty() ? fileName : cmdString;
      std::shared_ptr<edm::ProcessDesc> processDesc;
      try {
        std::unique_ptr<edm::ParameterSet> parameterSet;
        if (!fileName.empty())
          parameterSet = edm::readConfig(fileName, pythonOptValues);
        else
          edm::makeParameterSets(cmdString, parameterSet);
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

      // Determine the number of threads to use, and the per-thread stack size:
      //   - from the command line
      //   - from the "options" ParameterSet, if it exists
      //   - from default values (currently, 1 thread and 10 MB)
      //
      // Since TBB has already been initialised with the default values, re-initialise
      // it only if different values are discovered.
      //
      // Finally, reflect the values being used in the "options" top level ParameterSet.
      context = "Setting up number of threads";
      unsigned int nThreads = 0;
      {
        // check the "options" ParameterSet
        std::shared_ptr<edm::ParameterSet> pset = processDesc->getProcessPSet();
        auto threadsInfo = threadOptions(*pset);

        // check the command line options
        if (vm.count(kNumberOfThreadsOpt)) {
          threadsInfo.nThreads_ = vm[kNumberOfThreadsOpt].as<unsigned int>();
        }
        if (vm.count(kSizeOfStackForThreadOpt)) {
          threadsInfo.stackSize_ = vm[kSizeOfStackForThreadOpt].as<unsigned int>();
        }

        // if needed, re-initialise TBB
        if (threadsInfo.nThreads_ != edm::s_defaultNumberOfThreads or
            threadsInfo.stackSize_ != edm::s_defaultSizeOfStackForThreadsInKB) {
          threadsInfo.nThreads_ = edm::setNThreads(threadsInfo.nThreads_, threadsInfo.stackSize_, tsiPtr);
        }
        nThreads = threadsInfo.nThreads_;

        // update the numberOfThreads and sizeOfStackForThreadsInKB in the "options" ParameterSet
        setThreadOptions(threadsInfo, *pset);
      }

      context = "Initializing default service configurations";

      // Default parameters will be used for the default services
      // if they are not overridden from the configuration files.
      processDesc->addServices(edm::defaultCmsRunServices());

      context = "Setting MessageLogger defaults";
      // Decide what mode of hardcoded MessageLogger defaults to use
      if (vm.count(kJobModeOpt)) {
        std::string jobMode = vm[kJobModeOpt].as<std::string>();
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
        context = "Calling beginJob";
        proc->beginJob();

        alwaysAddContext = false;
        context =
            "Calling EventProcessor::runToCompletion (which does almost everything after beginJob and before endJob)";
        proc.on();
        auto status = proc->runToCompletion();
        if (status == edm::EventProcessor::epSignal) {
          returnCode = edm::errors::CaughtSignal;
        }
        proc.off();

        context = "Calling endJob";
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
