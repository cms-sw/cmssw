/*----------------------------------------------------------------------

This is a generic main that can be used with any plugin and a 
PSet script.   See notes in EventProcessor.cpp for details about
it.

$Id: cmsRun.cpp,v 1.13 2006/03/13 22:31:16 wmtan Exp $

----------------------------------------------------------------------*/  

#include <exception>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
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


  int rc = -1; // we should never return this value!
  try {
      edm::EventProcessor proc(configstring);
      proc.beginJob();
      proc.run();
      if(proc.endJob()) {
        rc = 0;
      } else {
        rc = 1;
      }
  }
  catch (seal::Error& e) {
      edm::LogError("FwkJob") << "Exception caught in " 
				<< kProgramName
				<< "\n"
				<< e.explainSelf();
      rc = 1;
  }
  catch (std::exception& e) {
      edm::LogError("FwkJob") << "Standard library exception caught in " 
				<< kProgramName
				<< "\n"
				<< e.what();
      rc = 1;
  }
  catch (...) {
      edm::LogError("FwkJob") << "Unknown exception caught in "
				<< kProgramName
				<< "\n";
      rc = 2;
  }
  
  return rc;
}
