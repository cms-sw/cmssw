/*----------------------------------------------------------------------
This is a generic main that can be used with any plugin and a
PSet script.   See notes in EventProcessor.cpp for details about it.
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/ServiceRegistry/interface/ServiceWrapper.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/ConvertException.h"
#include "FWCore/Utilities/interface/Presence.h"

#include "TError.h"

#include "boost/program_options.hpp"
#include "boost/shared_ptr.hpp"
#include "tbb/task_scheduler_init.h"

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
static char const* const kParameterSetCommandOpt = "parameter-set,p";
static char const* const kJobreportCommandOpt = "jobreport,j";
static char const* const kJobreportOpt = "jobreport";
static char const* const kEnableJobreportCommandOpt = "enablejobreport,e";
static const char* const kEnableJobreportOpt = "enablejobreport";
static char const* const kJobModeCommandOpt = "mode,m";
static char const* const kJobModeOpt="mode";
static char const* const kMultiThreadMessageLoggerOpt = "multithreadML,t";
static char const* const kNumberOfThreadsCommandOpt = "numThreads,n";
static char const* const kNumberOfThreadsOpt = "numThreads";
static char const* const kSizeOfStackForThreadCommandOpt = "sizeOfStackForThreadsInKB,s";
static char const* const kSizeOfStackForThreadOpt = "sizeOfStackForThreadsInKB";
static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt = "help,h";
static char const* const kStrictOpt = "strict";
static char const* const kProgramName = "cmsRun";

// -----------------------------------------------
namespace {
  class EventProcessorWithSentry {
  public:
    explicit EventProcessorWithSentry() : ep_(nullptr), callEndJob_(false) {}
    explicit EventProcessorWithSentry(std::auto_ptr<edm::EventProcessor> ep) :
      ep_(ep),
      callEndJob_(false) {}
    ~EventProcessorWithSentry() {
      if(callEndJob_ && ep_.get()) {
        try {
          ep_->endJob();
        }
        catch (...) {
	  edm::LogSystem("MoreExceptions")
            << "After a fatal primary exception was caught, there was an attempt to run\n"
            << "endJob methods. Another exception was caught while endJob was running\n"
            << "and we give up trying to run endJob.";
        }
      }
      edm::clearMessageLog();
    }
    void on() {
      callEndJob_ = true;
    }
    void off() {
      callEndJob_ = false;
    }
    edm::EventProcessor* operator->() {
      return ep_.get();
    }
  private:
    std::auto_ptr<edm::EventProcessor> ep_;
    bool callEndJob_;
  };
  
  void setNThreads(unsigned int iNThreads,
		   unsigned int iStackSize,
                   std::unique_ptr<tbb::task_scheduler_init>& oPtr) {
    //The TBB documentation doesn't explicitly say this, but when the task_scheduler_init's
    // destructor is run it does a 'wait all' for all tasks to finish and then shuts down all the threads.
    // This provides a clean synchronization point.
    //We have to destroy the old scheduler before starting a new one in order to
    // get tbb to actually switch the number of threads. If we do not, tbb stays at 1 threads
    edm::LogInfo("ThreadSetup") <<"setting # threads "<<iNThreads;

    //stack size is given in KB but passed in as bytes
    iStackSize *= 1024;

    oPtr.reset();
    if(0==iNThreads) {
      //Allow TBB to decide how many threads. This is normally the number of CPUs in the machine.
      oPtr = std::unique_ptr<tbb::task_scheduler_init>{new tbb::task_scheduler_init{tbb::task_scheduler_init::automatic,iStackSize}};
    } else {
      oPtr = std::unique_ptr<tbb::task_scheduler_init>{new tbb::task_scheduler_init{static_cast<int>(iNThreads),iStackSize}};
    }
  }
}

int main(int argc, char* argv[]) {
  
  int returnCode = 0;
  std::string context;
  bool alwaysAddContext = true;
  //Default to only use 1 thread. We define this early since plugin system or message logger
  // may be using TBB.
  bool setNThreadsOnCommandLine = false;
  std::unique_ptr<tbb::task_scheduler_init> tsiPtr{new tbb::task_scheduler_init{1}};
  boost::shared_ptr<edm::Presence> theMessageServicePresence;
  std::unique_ptr<std::ofstream> jobReportStreamPtr;
  boost::shared_ptr<edm::serviceregistry::ServiceWrapper<edm::JobReport> > jobRep;
  EventProcessorWithSentry proc;

  try {
    returnCode = edm::convertException::wrap([&]()->int {

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

      // Decide whether to use the multi-thread or single-thread message logger
      //    (Just walk the command-line arguments, since the boost parser will
      //    be run below and can lead to error messages which should be sent via
      //    the message logger)
      context = "Initializing either multi-threaded or single-threaded message logger";
      bool multiThreadML = false;
      for(int i = 0; i < argc; ++i) {
        if((std::strncmp (argv[i], "-t", 20) == 0) ||
           (std::strncmp (argv[i], "--multithreadML", 20) == 0)) {
          multiThreadML = true;
          break;
        }
      }

      // Load the message service plug-in

      if(multiThreadML) {
        theMessageServicePresence = boost::shared_ptr<edm::Presence>(edm::PresenceFactory::get()->
          makePresence("MessageServicePresence").release());
      }
      else {
        theMessageServicePresence = boost::shared_ptr<edm::Presence>(edm::PresenceFactory::get()->
          makePresence("SingleThreadMSPresence").release());
      }

      context = "Processing command line arguments";
      std::string descString(argv[0]);
      descString += " [options] [--";
      descString += kParameterSetOpt;
      descString += "] config_file \nAllowed options";
      boost::program_options::options_description desc(descString);

      desc.add_options()
        (kHelpCommandOpt, "produce help message")
        (kParameterSetCommandOpt, boost::program_options::value<std::string>(), "configuration file")
        (kJobreportCommandOpt, boost::program_options::value<std::string>(),
                "file name to use for a job report file: default extension is .xml")
        (kEnableJobreportCommandOpt,
                "enable job report files (if any) specified in configuration file")
        (kJobModeCommandOpt, boost::program_options::value<std::string>(),
                "Job Mode for MessageLogger defaults - default mode is grid")
	(kNumberOfThreadsCommandOpt,boost::program_options::value<unsigned int>(),
                "Number of threads to use in job (0 is use all CPUs)")
	(kSizeOfStackForThreadCommandOpt,boost::program_options::value<unsigned int>(),
   	        "Size of stack in KB to use for extra threads (0 is use system default size)")
        (kMultiThreadMessageLoggerOpt,
                "MessageLogger handles multiple threads - default is single-thread")
        (kStrictOpt, "strict parsing");

      // anything at the end will be ignored, and sent to python
      boost::program_options::positional_options_description p;
      p.add(kParameterSetOpt, 1).add(kPythonOpt, -1);

      // This --fwk option is not used anymore, but I'm leaving it around as
      // it might be useful again in the future for code development
      // purposes.  We originally used it when implementing the boost
      // state machine code.
      boost::program_options::options_description hidden("hidden options");
      hidden.add_options()("fwk", "For use only by Framework Developers")
        (kPythonOpt, boost::program_options::value< std::vector<std::string> >(),
         "options at the end to be passed to python");

      boost::program_options::options_description all_options("All Options");
      all_options.add(desc).add(hidden);

      boost::program_options::variables_map vm;
      try {
        store(boost::program_options::command_line_parser(argc, argv).options(all_options).positional(p).run(), vm);
        notify(vm);
      }
      catch (boost::program_options::error const& iException) {
        edm::LogAbsolute("CommandLineProcessing") << "cmsRun: Error while trying to process command line arguments:\n"
          << iException.what()
	  << "\nFor usage and an options list, please do 'cmsRun --help'.";
        return edm::errors::CommandLineProcessing;
      }

      if (vm.count(kHelpOpt)) {
        std::cout << desc << std::endl;
        if (!vm.count(kParameterSetOpt)) edm::HaltMessageLogging();
        return 0;
      }
      
      if(vm.count(kNumberOfThreadsOpt)) {
        setNThreadsOnCommandLine=true;
        unsigned int nThreads = vm[kNumberOfThreadsOpt].as<unsigned int>();
        unsigned int stackSize=0;
        if(vm.count(kSizeOfStackForThreadOpt)) {
	  stackSize=vm[kSizeOfStackForThreadOpt].as<unsigned int>();
	}
        setNThreads(nThreads,stackSize,tsiPtr);
      }

      if (!vm.count(kParameterSetOpt)) {
        edm::LogAbsolute("ConfigFileNotFound") << "cmsRun: No configuration file given.\n"
          << "For usage and an options list, please do 'cmsRun --help'.";
        edm::HaltMessageLogging();
        return edm::errors::ConfigFileNotFound;
      }
      std::string fileName(vm[kParameterSetOpt].as<std::string>());

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
      } else if(vm.count(kEnableJobreportOpt)) {
        jobReportFile = "FrameworkJobReport.xml";
      }
      jobReportStreamPtr = std::auto_ptr<std::ofstream>(jobReportFile.empty() ? 0 : new std::ofstream(jobReportFile.c_str()));

      //NOTE: JobReport must have a lifetime shorter than jobReportStreamPtr so that when the JobReport destructor
      // is called jobReportStreamPtr is still valid
      std::auto_ptr<edm::JobReport> jobRepPtr(new edm::JobReport(jobReportStreamPtr.get()));
      jobRep.reset(new edm::serviceregistry::ServiceWrapper<edm::JobReport>(jobRepPtr));
      edm::ServiceToken jobReportToken =
        edm::ServiceRegistry::createContaining(jobRep);

      context = "Processing the python configuration file named ";
      context += fileName;
      boost::shared_ptr<edm::ProcessDesc> processDesc;
      try {
        boost::shared_ptr<edm::ParameterSet> parameterSet = edm::readConfig(fileName, argc, argv);
        processDesc.reset(new edm::ProcessDesc(parameterSet));
      }
      catch(cms::Exception& iException) {
        edm::Exception e(edm::errors::ConfigFileReadError, "", iException);
        throw e;
      }
      
      //See if we were told how many threads to use. If so then inform TBB only if
      // we haven't already been told how many threads to use in the command line
      context = "Setting up number of threads";
      {
        if(not setNThreadsOnCommandLine) {
          boost::shared_ptr<edm::ParameterSet> pset = processDesc->getProcessPSet();
          if(pset->existsAs<edm::ParameterSet>("options",false)) {
            auto const& ops = pset->getUntrackedParameterSet("options");
            if(ops.existsAs<unsigned int>("numberOfThreads",false)) {
              unsigned int nThreads = ops.getUntrackedParameter<unsigned int>("numberOfThreads");
	      unsigned int stackSize=0;
	      if(ops.existsAs<unsigned int>("sizeOfStackForThreadsInKB",0)) {
		stackSize = ops.getUntrackedParameter<unsigned int>("sizeOfStackForThreadsInKB");
	      }
              setNThreads(nThreads,stackSize,tsiPtr);
            }
          }
        }
      }

      context = "Initializing default service configurations";
      std::vector<std::string> defaultServices;
      defaultServices.reserve(7);
      defaultServices.push_back("MessageLogger");
      defaultServices.push_back("InitRootHandlers");
#ifdef linux
      defaultServices.push_back("EnableFloatingPointExceptions");
#endif
      defaultServices.push_back("UnixSignalService");
      defaultServices.push_back("AdaptorConfig");
      defaultServices.push_back("SiteLocalConfigService");
      defaultServices.push_back("StatisticsSenderService");

      // Default parameters will be used for the default services
      // if they are not overridden from the configuration files.
      processDesc->addServices(defaultServices);

      context = "Setting MessageLogger defaults";
      // Decide what mode of hardcoded MessageLogger defaults to use
      if (vm.count(kJobModeOpt)) {
        std::string jobMode = vm[kJobModeOpt].as<std::string>();
        edm::MessageDrop::instance()->jobMode = jobMode;
      }

      context = "Constructing the EventProcessor";
      std::auto_ptr<edm::EventProcessor>
          procP(new
                edm::EventProcessor(processDesc, jobReportToken,
                                    edm::serviceregistry::kTokenOverrides));
      EventProcessorWithSentry procTmp(procP);
      proc = procTmp;

      alwaysAddContext = false;
      context = "Calling beginJob";
      proc->beginJob();

      alwaysAddContext = true;
      context = "Calling EventProcessor::forkProcess";
      if (!proc->forkProcess(jobReportFile)) {
        return 0;
      }

      alwaysAddContext = false;
      context = "Calling EventProcessor::runToCompletion (which does almost everything after beginJob and before endJob)";
      proc.on();
      auto status = proc->runToCompletion();
      if (status == edm::EventProcessor::epSignal) {
        returnCode = edm::errors::CaughtSignal;
      }
      proc.off();

      context = "Calling endJob";
      proc->endJob();
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
      }
      else if (ex.context().empty()) {
        ex.addContext(context);
      }
    }
    if (!ex.alreadyPrinted()) {
      if (jobRep.get() != 0) {
        edm::printCmsException(ex, &(jobRep->get()), returnCode);
      }
      else {
        edm::printCmsException(ex);
      }
    }
  }
  // Disable Root Error Handler.
  SetErrorHandler(DefaultErrorHandler);
  return returnCode;
}
