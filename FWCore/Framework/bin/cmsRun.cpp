/*----------------------------------------------------------------------

This is a generic main that can be used with any plugin and a 
PSet script.   See notes in EventProcessor.cpp for details about
it.

$Id: cmsRun.cpp,v 1.16 2006/04/06 23:08:32 wmtan Exp $

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
    return 1;
  }
    
  if(vm.count(kHelpOpt)) {
    std::cout << desc <<std::endl;
    return 0;
  }
  
  if(!vm.count(kParameterSetOpt)) {
    edm::LogError("FwkJob") << "No configuration file given \n"
			      <<" please do '"
			      << argv[0]
			      <<  " --"
			      << kHelpOpt
			      << "'.";
    return 1;
  }

  std::ifstream configFile(vm[kParameterSetOpt].as<std::string>().c_str());
  if(!configFile) {
    edm::LogError("FwkJob") << "Unable to open configuration file "
			      << vm[kParameterSetOpt].as<std::string>();
    return 1;
  }


  // Create the several parameter sets that will be used to configure
  // the program.
  std::string configstring;
  std::string line;

  while(std::getline(configFile,line)) { configstring+=line; configstring+="\n"; }
  
  edm::ParameterSet main;
  std::vector<edm::ParameterSet> serviceparams;


  EventProcessorWithSentry proc;
  int rc = -1; // we should never return this value!
  try {
      std::auto_ptr<edm::EventProcessor> procP(new edm::EventProcessor(configstring));
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
      edm::LogError("FwkJob") << "seal::Exception caught in " 
				<< kProgramName
				<< "\n"
				<< e.explainSelf();
      rc = 1;
      // TODO: Put 'job failure' report to JobSummary here
  }
  catch (cms::Exception& e) {
      edm::LogError("FwkJob") << "cms::Exception caught in " 
				<< kProgramName
				<< "\n"
				<< e.explainSelf();
      rc = 1;
      // TODO: Put 'job failure' report to JobSummary here
  }
  catch (std::exception& e) {
      edm::LogError("FwkJob") << "Standard library exception caught in " 
				<< kProgramName
				<< "\n"
				<< e.what();
      rc = 1;
      // TODO: Put 'job failure' report to JobSummary here
  }
  catch (...) {
      edm::LogError("FwkJob") << "Unknown exception caught in "
				<< kProgramName
				<< "\n";
      rc = 2;
      // TODO: Put 'job failure' report to JobSummary here
  }
  
  return rc;
}
