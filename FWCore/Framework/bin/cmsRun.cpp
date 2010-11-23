/*----------------------------------------------------------------------

This is a generic main that can be used with any plugin and a 
PSet script.   See notes in EventProcessor.cpp for details about
it.


----------------------------------------------------------------------*/  

#include <exception>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <boost/program_options.hpp>
#include "boost/shared_ptr.hpp"
#include <cstring>

#include "FWCore/PythonParameterSet/interface/MakeParameterSets.h"
#include "FWCore/ParameterSet/interface/ProcessDesc.h"
#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/Utilities/interface/RootHandlers.h"
#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/ServiceToken.h"
#include "FWCore/ServiceRegistry/interface/ServiceWrapper.h"

#include "TError.h"

static char const* const kParameterSetOpt = "parameter-set";
static char const* const kPythonOpt = "pythonOptions";
static char const* const kParameterSetCommandOpt = "parameter-set,p";
static char const* const kJobreportCommandOpt = "jobreport,j";
static char const* const kEnableJobreportCommandOpt = "enablejobreport,e";
static char const* const kJobModeCommandOpt = "mode,m";
static char const* const kMultiThreadMessageLoggerOpt = "multithreadML,t";
static char const* const kHelpOpt = "help";
static char const* const kHelpCommandOpt = "help,h";
static char const* const kStrictOpt = "strict";
static char const* const kProgramName = "cmsRun";

// -----------------------------------------------
namespace {
  class EventProcessorWithSentry {
  public:
    explicit EventProcessorWithSentry() : ep_(0), callEndJob_(false) { }
    explicit EventProcessorWithSentry(std::auto_ptr<edm::EventProcessor> ep) :
      ep_(ep),
      callEndJob_(false) { }
    ~EventProcessorWithSentry() {
      if (callEndJob_ && ep_.get()) {
	try {
	  ep_->endJob();
	}
	catch (cms::Exception& e) {
	  edm::printCmsException(e, kProgramName);
	}
	catch (std::bad_alloc& e) {
	  edm::printBadAllocException(kProgramName);
	}
	catch (std::exception& e) {
	  edm::printStdException(e, kProgramName);
	}
	catch (...) {
	  edm::printUnknownException(kProgramName);
        }
      }
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
}

int main(int argc, char* argv[])
{
// NOTE: MacOs X has a lower rlimit for opened file descriptor than Linux (256
// in Snow Leopard vs 512 in SLC5). This is a problem for some of the workflows
// that open many small root datafiles.  Notice that this is safe to do also
// for Linux, but we agreed not to change the behavior there for the moment.
// Also the limits imposed by ulimit are not affected and still apply, if
// there.
#ifdef __APPLE__
  struct rlimit limits;
  getrlimit(RLIMIT_NOFILE, &limits);
  limits.rlim_cur = (OPEN_MAX < limits.rlim_max) ? OPEN_MAX : limits.rlim_max;
  setrlimit(RLIMIT_NOFILE, &limits);
#endif

  // We must initialize the plug-in manager first
  try {
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  } catch(const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  
  // Decide whether to use the multi-thread or single-thread message logger
  //    (Just walk the command-line arguments, since the boost parser will
  //    be run below and can lead to error messages which should be sent via
  //    the message logger)
  bool multiThreadML = false;
  for (int i=0; i<argc; ++i) {
    if ( (std::strncmp (argv[i],"-t", 20) == 0) ||
         (std::strncmp (argv[i],"--multithreadML", 20) == 0) )
    { multiThreadML = true; 
      break; 
    }
  } 
 
  // TEMPORARY -- REMOVE AT ONCE!!!!!
  // if ( multiThreadML ) std::cerr << "\n\n multiThreadML \n\n";
  
  // Load the message service plug-in
  boost::shared_ptr<edm::Presence> theMessageServicePresence;

  if (multiThreadML)
  {
    try {
      theMessageServicePresence = boost::shared_ptr<edm::Presence>(edm::PresenceFactory::get()->
          makePresence("MessageServicePresence").release());
    } catch(cms::Exception& e) {
      std::cerr << e.what() << std::endl;
      return 1;
    }
  } else {
    try {
      theMessageServicePresence = boost::shared_ptr<edm::Presence>(edm::PresenceFactory::get()->
          makePresence("SingleThreadMSPresence").release());
    } catch(cms::Exception& e) {
      std::cerr << e.what() << std::endl;
      return 1;
    }
  }
  
  //
  // Specify default services to be enabled with their default parameters.
  // 
  // The parameters for these can be overridden from the configuration files.
  std::vector<std::string> defaultServices;
  defaultServices.reserve(6);
  defaultServices.push_back("MessageLogger");
  defaultServices.push_back("InitRootHandlers");
#ifdef linux
  defaultServices.push_back("EnableFloatingPointExceptions");
#endif
  defaultServices.push_back("UnixSignalService");
  defaultServices.push_back("AdaptorConfig");
  defaultServices.push_back("SiteLocalConfigService");

  // These cannot be overridden from the configuration files.
  // An exception will be thrown if any of these is specified there.
  std::vector<std::string> forcedServices;
  forcedServices.reserve(1);
  forcedServices.push_back("JobReportService");

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
    store(boost::program_options::command_line_parser(argc,argv).options(all_options).positional(p).run(),vm);
    notify(vm);
  } catch(boost::program_options::error const& iException) {
    edm::LogError("FwkJob") << "Exception from command line processing: " << iException.what();
    edm::LogSystem("CommandLineProcessing") << "Exception from command line processing: " << iException.what() << "\n";
    return 7000;
  }
    
  if(vm.count(kHelpOpt)) {
    std::cout << desc <<std::endl;
    if(!vm.count(kParameterSetOpt)) edm::HaltMessageLogging();
    return 0;
  }
  
  if(!vm.count(kParameterSetOpt)) {
    std::string shortDesc("ConfigFileNotFound");
    std::ostringstream longDesc;
    longDesc << "cmsRun: No configuration file given.\n"
	     << "For usage and an options list, please do '"
	     << argv[0]
	     <<  " --"
	     << kHelpOpt
	     << "'.";
    int exitCode = 7001;
    edm::LogAbsolute(shortDesc) << longDesc.str() << "\n";
    edm::HaltMessageLogging();
    return exitCode;
  }

#ifdef CHANGED_FROM
  if(!vm.count(kParameterSetOpt)) {
    std::string shortDesc("ConfigFileNotFound");
    std::ostringstream longDesc;
    longDesc << "No configuration file given \n"
	     <<" please do '"
	     << argv[0]
	     <<  " --"
	     << kHelpOpt
	     << "'.";
    int exitCode = 7001;
    jobRep->reportError(shortDesc, longDesc.str(), exitCode);
    edm::LogSystem(shortDesc) << longDesc.str() << "\n";
    return exitCode;
  }
#endif

  //
  // Decide whether to enable creation of job report xml file 
  //  We do this first so any errors will be reported
  // 
  std::string jobReportFile;
  if (vm.count("jobreport")) {
    jobReportFile = vm["jobreport"].as<std::string>();
  } else if (vm.count("enablejobreport")) {
    jobReportFile = "FrameworkJobReport.xml";
  } 
  std::auto_ptr<std::ofstream> jobReportStreamPtr = std::auto_ptr<std::ofstream>(jobReportFile.empty() ? 0 : new std::ofstream(jobReportFile.c_str()));
  //
  // Make JobReport Service up front
  // 
  //NOTE: JobReport must have a lifetime shorter than jobReportStreamPtr so that when the JobReport destructor
  // is called jobReportStreamPtr is still valid
  std::auto_ptr<edm::JobReport> jobRepPtr(new edm::JobReport(jobReportStreamPtr.get()));  
  boost::shared_ptr<edm::serviceregistry::ServiceWrapper<edm::JobReport> > jobRep( new edm::serviceregistry::ServiceWrapper<edm::JobReport>(jobRepPtr) );
  edm::ServiceToken jobReportToken = 
    edm::ServiceRegistry::createContaining(jobRep);
  
  std::string fileName(vm[kParameterSetOpt].as<std::string>());
  boost::shared_ptr<edm::ProcessDesc> processDesc;
  try {
    processDesc = edm::readConfig(fileName, argc, argv);
  }
  catch(cms::Exception& iException) {
    std::string shortDesc("ConfigFileReadError");
    std::ostringstream longDesc;
    longDesc << "Problem with configuration file " << fileName
             <<  "\n" << iException.what();
    int exitCode = 7002;
    jobRep->get().reportError(shortDesc, longDesc.str(), exitCode);
    edm::LogSystem(shortDesc) << longDesc.str() << "\n";
    return exitCode;
  }

  processDesc->addServices(defaultServices, forcedServices);
  //
  // Decide what mode of hardcoded MessageLogger defaults to use 
  // 
  if (vm.count("mode")) {
    std::string jobMode = vm["mode"].as<std::string>();
    edm::MessageDrop::instance()->jobMode = jobMode;
  }  

  if(vm.count(kStrictOpt))
  {
    //edm::setStrictParsing(true);
    edm::LogSystem("CommandLineProcessing") << "Strict configuration processing is now done from python";
  }
 
  // Now create and configure the services
  //
  EventProcessorWithSentry proc;
  int rc = -1; // we should never return this value!
  try {
    std::auto_ptr<edm::EventProcessor> 
	procP(new 
	      edm::EventProcessor(processDesc, jobReportToken, 
			     edm::serviceregistry::kTokenOverrides));
    EventProcessorWithSentry procTmp(procP);
    proc = procTmp;
    proc->beginJob();
    if(!proc->forkProcess(jobReportFile)) {
      return 0;
    }
    proc.on();
    bool onlineStateTransitions = false;
    proc->runToCompletion(onlineStateTransitions);
    proc.off();
    proc->endJob();
    rc = 0;
    // Disable Root Error Handler so we do not throw because of ROOT errors.
    edm::ServiceToken token = proc->getToken();
    edm::ServiceRegistry::Operate operate(token);
    edm::Service<edm::RootHandlers> rootHandler;
    rootHandler->disableErrorHandler();
  }
  catch (edm::Exception& e) {
    rc = e.returnCode();
    edm::printCmsException(e, kProgramName, &(jobRep->get()), rc);
  }
  catch (cms::Exception& e) {
    rc = 8001;
    edm::printCmsException(e, kProgramName, &(jobRep->get()), rc);
  }
  catch(std::bad_alloc& bda) {
    rc = 8004;
    edm::printBadAllocException(kProgramName, &(jobRep->get()), rc);
  }
  catch (std::exception& e) {
    rc = 8002;
    edm::printStdException(e, kProgramName, &(jobRep->get()), rc);
  }
  catch (...) {
    rc = 8003;
    edm::printUnknownException(kProgramName, &(jobRep->get()), rc);
  }
  // Disable Root Error Handler again, just in case an exception
  // caused the above disabling of the handler to be bypassed.
  SetErrorHandler(DefaultErrorHandler);
  return rc;
}
