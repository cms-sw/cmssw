/*----------------------------------------------------------------------

This is a generic main that can be used with any plugin and a 
PSet script.   See notes in EventProcessor.cpp for details about
it.

$Id: cmsRun.cpp,v 1.3 2005/09/09 16:43:50 wmtan Exp $

----------------------------------------------------------------------*/  

#include <exception>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>

#include "FWCore/Framework/interface/EventProcessor.h"
#include "FWCore/Utilities/interface/ProblemTracker.h"
#include "FWCore/Utilities/interface/Exception.h"

using namespace std;
static const char* const kParameterSetOpt = "parameter-set";
static const char* const kParameterSetCommandOpt = "parameter-set,p";
static const char* const kHelpOpt = "help";
static const char* const kHelpCommandOpt = "help,h";

// -----------------------------------------------

int main(int argc, char* argv[])
{
  using namespace boost::program_options;
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
    std::cout << iException.what()<<std::endl;
    return 1;
  }
    
  if(vm.count(kHelpOpt)){
    std::cout << desc <<std::endl;
    return 1;
  }
  
  if(!vm.count(kParameterSetOpt)){
    std::cout <<"no configuration file given \n"
    <<" please do '"<<argv[0]<<" --"<<kHelpOpt<<"'."<<std::endl;
    return 1;
  }

  ifstream configFile(vm[kParameterSetOpt].as<std::string>().c_str());
  if(!configFile) {
    std::cout <<"unable to open configuration file "
    <<vm[kParameterSetOpt].as<std::string>()<<std::endl;
    return 1;
  }
  
  string configstring;
  string line;
  
  while(std::getline(configFile,line)) { configstring+=line; configstring+="\n"; }
  
  edm::AssertHandler ah;

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
      std::cerr << "Exception caught in " << argv[0] << "\n"
		<< e.explainSelf()
		<< std::endl;
      rc = 1;
    }
  catch (std::exception& e)
    {
      std::cerr << "Standard library exception caught in " << argv[0] << "\n"
		<< e.what()
		<< std::endl;
      rc = 1;
    }
  catch (...)
    {
      std::cerr << "Unknown exception caught in " << argv[0]
		<< std::endl;
      rc = 2;
    }
  
  return rc;
}
