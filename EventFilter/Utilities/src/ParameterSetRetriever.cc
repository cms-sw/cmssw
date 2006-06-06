#include "EventFilter/Utilities/interface/ParameterSetRetriever.h"
#include "EventFilter/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace evf;
using namespace std;


#include <fstream>

ParameterSetRetriever::ParameterSetRetriever(string &in)
{
  std::string::size_type start = 0;
  std::string fileheading = "file:";
  if((start=in.rfind("file:"))!= std::string::npos)
    { 
      std::string filename = in.substr(start+fileheading.size(),in.size()-start);
      std::cout << "FileName is \n" << filename << std::endl;
      std::string line;
      std::ifstream configFile(filename.c_str());
      while(std::getline(configFile,line)) { pset+=line; pset+="\n"; }
    }
  else if((start=in.rfind("db:"))!= std::string::npos)
    {
      XCEPT_RAISE(evf::Exception,"db access for ParameterSet not yet implemented");
    } 
  else
    {
      edm::LogWarning("psetRetriever") << "Using direct config from XML"
				       << endl;
      pset = in;
    }
}

string ParameterSetRetriever::getAsString()const {return pset;}
