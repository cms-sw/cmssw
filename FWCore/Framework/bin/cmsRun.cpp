/*----------------------------------------------------------------------

This is a generic main that can be used with any plugin and a 
PSet script.   See notes in EventProcessor.cpp for details about
it.

$Id: cmsRun.cpp,v 1.21 2006/11/17 00:35:26 wmtan Exp $

----------------------------------------------------------------------*/  

#include <exception>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <boost/shared_ptr.hpp>
#include <boost/program_options.hpp>

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Utilities/interface/ProblemTracker.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/Presence.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/PresenceFactory.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/ServiceRegistry/interface/ServiceRegistry.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "FWCore/Framework/bin/pythonFileToConfigure.h"

static const char* const kParameterSetOpt = "parameter-set";
static const char* const kParameterSetCommandOpt = "parameter-set,p";
static const char* const kHelpOpt = "help";
static const char* const kHelpCommandOpt = "help,h";
static const char* const kProgramName = "cmsRun";

// -----------------------------------------------
namespace {
  class EventProcessorWithSentry {
  public:
    explicit EventProcessorWithSentry() : ep_(0), callEndJob_(false) { }
    explicit EventProcessorWithSentry(std::auto_ptr<edm::EventProcessor> ep) :
      ep_(ep),
      callEndJob_(false) { }
    ~EventProcessorWithSentry() {
      if (callEndJob_ && ep_.get()) ep_->endJob();
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
  using namespace boost::program_options;

  // We must initialize the plug-in manager first
  edm::AssertHandler ah;

  // Load the message service plug-in
  boost::shared_ptr<edm::Presence> theMessageServicePresence;
  try {
    theMessageServicePresence = boost::shared_ptr<edm::Presence>(edm::PresenceFactory::get()->
        makePresence("MessageServicePresence").release());
  } catch(seal::Error& e) {
    std::cerr << e.explainSelf() << std::endl;
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
  defaultServices.reserve(4);
  defaultServices.push_back("AdaptorConfig");
  defaultServices.push_back("InitRootHandlers");
  defaultServices.push_back("MessageLogger");
  defaultServices.push_back("LoadAllDictionaries");
  
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
  options_description desc(descString);
  
  desc.add_options()
    (kHelpCommandOpt, "produce help message")
    (kParameterSetCommandOpt,value<std::string>(), "configuration file")
    ;

  positional_options_description p;
  p.add(kParameterSetOpt, -1);
  
  variables_map vm;
  try {
    store(command_line_parser(argc,argv).options(desc).positional(p).run(),vm);
    notify(vm);
  } catch(const error& iException) {
    edm::LogError("FwkJob") << "Exception from command line processing: " << iException.what();
    edm::LogSystem("CommandLineProcessing") << "Exception from command line processing: " << iException.what() << "\n";
    return 7000;
  }
    
  if(vm.count(kHelpOpt)) {
    std::cout << desc <<std::endl;
    return 0;
  }
  
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

  std::string configstring;
  std::string fileName(vm[kParameterSetOpt].as<std::string>());
  if (fileName.size() > 3 && fileName.substr(fileName.size()-3) == ".py") {
    try {
      configstring = edm::pythonFileToConfigure(fileName);
    } catch(cms::Exception& iException) {
      std::string shortDesc("ConfigFileReadError");
      std::ostringstream longDesc;
      longDesc << "python found a problem "<<iException.what();
      int exitCode = 7002;
      jobRep->reportError(shortDesc, longDesc.str(), exitCode);
      edm::LogSystem(shortDesc) << longDesc.str() << "\n";
      return exitCode;
    }
  } else {
    std::ifstream configFile(fileName.c_str());
    if(!configFile) {
      std::string shortDesc("ConfigFileReadError");
      std::ostringstream longDesc;
      longDesc << "Unable to open configuration file "
        << vm[kParameterSetOpt].as<std::string>();
      int exitCode = 7002;
      jobRep->reportError(shortDesc, longDesc.str(), exitCode);
      edm::LogSystem(shortDesc) << longDesc.str() << "\n";
      return exitCode;
    }
    
    
    // Create the several parameter sets that will be used to configure
    // the program.
    std::string line;
    
    while(std::getline(configFile,line)) {
      configstring += line; 
      configstring += "\n"; 
    }
  }

  EventProcessorWithSentry proc;
  int rc = -1; // we should never return this value!
  using std::auto_ptr;
  using edm::EventProcessor;
  try {
    auto_ptr<EventProcessor> 
	procP(new 
	      EventProcessor(configstring, jobReportToken, 
			     edm::serviceregistry::kTokenOverrides,
			     defaultServices, forcedServices));
    EventProcessorWithSentry procTmp(procP);
    proc = procTmp;
    proc->beginJob();
    proc.on();
    proc->run();
    proc.off();
    proc->endJob();
    rc = 0;
  }
  catch (seal::Error& e) {
    std::string shortDesc("SEALException");
    std::ostringstream longDesc;
    longDesc << "seal::Exception caught in " 
	     << kProgramName
	     << "\n"
	     << e.explainSelf();
    rc = 8000;
    jobRep->reportError(shortDesc, longDesc.str(), rc);
    edm::LogSystem(shortDesc) << longDesc.str() << "\n";
  }
  catch (cms::Exception& e) {
    std::string shortDesc("CMSException");
    std::ostringstream longDesc;
    longDesc << "cms::Exception caught in " 
	     << kProgramName
	     << "\n"
	     << e.explainSelf();
    rc = 8001;
    jobRep->reportError(shortDesc, longDesc.str(), rc);
    edm::LogSystem(shortDesc) << longDesc.str() << "\n";
  }
  catch (std::exception& e) {
    std::string shortDesc("StdLibException");
    std::ostringstream longDesc;
    longDesc << "Standard library exception caught in " 
	     << kProgramName
	     << "\n"
	     << e.what();
    rc = 8002;
    jobRep->reportError(shortDesc, longDesc.str(), rc);
    edm::LogSystem(shortDesc) << longDesc.str() << "\n";
         
  }
  catch (...) {
    std::string shortDesc("UnknownException");
    std::ostringstream longDesc;
    longDesc << "Unknown exception caught in "
	     << kProgramName
	     << "\n";
    rc = 8003;
    jobRep->reportError(shortDesc, longDesc.str(), rc);
    edm::LogSystem(shortDesc) << longDesc.str() << "\n";
  }
  
  return rc;
}
