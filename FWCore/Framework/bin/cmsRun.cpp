/*----------------------------------------------------------------------

This is a generic main that can be used with any plugin and a 
PSet script.   See notes in EventProcessor.cpp for details about
it.

$Id: cmsRun.cpp,v 1.43 2007/08/09 22:12:45 wmtan Exp $

----------------------------------------------------------------------*/  

#include <exception>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>

#include "FWCore/ParameterSet/interface/MakeParameterSets.h"
#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/MessageLogger/interface/ExceptionMessages.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/MessageLogger/interface/MessageDrop.h"
#include "FWCore/PluginManager/interface/PresenceFactory.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"


static char const* const kParameterSetOpt = "parameter-set";
static char const* const kParameterSetCommandOpt = "parameter-set,p";
static char const* const kJobreportCommandOpt = "jobreport,j";
static char const* const kEnableJobreportCommandOpt = "enablejobreport,e";
static char const* const kJobModeCommandOpt = "mode,m";
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
  
  // We must initialize the plug-in manager first
  try {
    edmplugin::PluginManager::configure(edmplugin::standard::config());
  } catch(cms::Exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }
  
  // Load the message service plug-in
  boost::shared_ptr<edm::Presence> theMessageServicePresence;
  try {
    theMessageServicePresence = boost::shared_ptr<edm::Presence>(edm::PresenceFactory::get()->
        makePresence("MessageServicePresence").release());
  } catch(cms::Exception& e) {
    std::cerr << e.what() << std::endl;
    return 1;
  }


  //
  // Make JobReport Service up front
  // 
  std::string jobReportFile = "FrameworkJobReport.xml";
  std::auto_ptr<edm::JobReport> jobRep(new edm::JobReport());  
  edm::ServiceToken jobReportToken = 
           edm::ServiceRegistry::createContaining(jobRep);
  
  //
  // Specify default services to be enabled with their default parameters.
  // 
  // The parameters for these can be overridden from the configuration files.
  std::vector<std::string> defaultServices;
  defaultServices.reserve(6);
  defaultServices.push_back("MessageLogger");
  defaultServices.push_back("LoadAllDictionaries");
  defaultServices.push_back("InitRootHandlers");
  defaultServices.push_back("AdaptorConfig");
  defaultServices.push_back("EnableFloatingPointExceptions");
  defaultServices.push_back("UnixSignalService");
  
  // These cannot be overridden from the configuration files.
  // An exception will be thrown if any of these is specified there.
  std::vector<std::string> forcedServices;
  forcedServices.reserve(2);
  forcedServices.push_back("JobReportService");
  forcedServices.push_back("SiteLocalConfigService");

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
    (kStrictOpt, "strict parsing");

  boost::program_options::positional_options_description p;
  p.add(kParameterSetOpt, -1);
  
  boost::program_options::variables_map vm;
  try {
    store(boost::program_options::command_line_parser(argc,argv).options(desc).positional(p).run(),vm);
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

  std::string fileName(vm[kParameterSetOpt].as<std::string>());
  boost::shared_ptr<edm::ProcessDesc> processDesc;
  try {
    processDesc = edm::readConfigFile(fileName);
  }
  catch(cms::Exception& iException) {
    std::string shortDesc("ConfigFileReadError");
    std::ostringstream longDesc;
    longDesc << "Problem with configuration file " << fileName
             <<  "\n" << iException.what();
    int exitCode = 7002;
    jobRep->reportError(shortDesc, longDesc.str(), exitCode);
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

  //
  // Decide whether to enable creation of job report xml file 
  // 
  if (vm.count("jobreport")) {
    std::string jr_name = vm["jobreport"].as<std::string>();
    edm::MessageDrop::instance()->jobreport_name = jr_name;
  } else if (vm.count("enablejobreport")) {
    std::string jr_name = "*";
    edm::MessageDrop::instance()->jobreport_name = jr_name;
  }  

  if(vm.count(kStrictOpt))
  {
    edm::setStrictParsing(true);
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
    proc.on();
    proc->run();
    proc.off();
    proc->endJob();
    rc = 0;
  }
  catch (cms::Exception& e) {
    rc = 8001;
    edm::printCmsException(e, kProgramName, jobRep.get(), rc);
  }
  catch(std::bad_alloc& bda) {
    rc = 8004;
    edm::printBadAllocException(kProgramName, jobRep.get(), rc);
  }
  catch (std::exception& e) {
    rc = 8002;
    edm::printStdException(e, kProgramName, jobRep.get(), rc);
  }
  catch (...) {
    rc = 8003;
    edm::printUnknownException(kProgramName, jobRep.get(), rc);
  }
  
  return rc;
}
