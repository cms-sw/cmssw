/*----------------------------------------------------------------------

This is a generic main that can be used with any plugin and a 
PSet script.   See notes in EventProcessor.cpp for details about
it.

$Id: cmsRun.cpp,v 1.9 2006/02/15 19:36:14 wmtan Exp $

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

using namespace std;
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
  boost::shared_ptr<edm::Presence> theMessageLoggerSpigot
      (edm::PresenceFactory::get()->
	makePresence("MessageLoggerSpigot").release());

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
    edm::LogError(kProgramName) << "Exception from command line processing: " << iException.what();
    return 1;
  }
    
  if(vm.count(kHelpOpt)){
    std::cout << desc <<std::endl;
    return 0;
  }
  
  if(!vm.count(kParameterSetOpt)){
    edm::LogError(kProgramName) << "No configuration file given \n"
			      <<" please do '"
			      << argv[0]
			      <<  " --"
			      << kHelpOpt
			      << "'.";
    return 1;
  }

  ifstream configFile(vm[kParameterSetOpt].as<std::string>().c_str());
  if(!configFile) {
    edm::LogError(kProgramName) << "Unable to open configuration file "
			      << vm[kParameterSetOpt].as<std::string>();
    return 1;
  }


  // Create the several parameter sets that will be used to configure
  // the program.
  string configstring;
  string line;

  while(std::getline(configFile,line)) { configstring+=line; configstring+="\n"; }
  
  edm::ParameterSet main;
  std::vector<edm::ParameterSet> serviceparams;


  int rc = -1; // we should never return this value!
  try
    {
      edm::EventProcessor proc(configstring);
      proc.beginJob();
      proc.run();
      if(proc.endJob()) {
        rc = 0;
      } else {
        rc = 1;
      }
    }
  catch (seal::Error& e)
    {
      edm::LogError(kProgramName) << "Exception caught in " 
				<< kProgramName 
				<< "\n"
				<< e.explainSelf();
      rc = 1;
    }
  catch (std::exception& e)
    {
      edm::LogError(kProgramName) << "Standard library exception caught in " 
				<< kProgramName 
				<< "\n"
				<< e.what();
      rc = 1;
    }
  catch (...)
    {
      edm::LogError(kProgramName) << "Unknown exception caught in " 
				  << kProgramName;
      rc = 2;
    }
  
  return rc;
}
