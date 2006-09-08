#include "EventFilter/Utilities/interface/ParameterSetRetriever.h"
#include "EventFilter/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace evf;
using namespace std;


#include <fstream>


//______________________________________________________________________________
ParameterSetRetriever::ParameterSetRetriever(const string& in)
{
  string fileheading="file:";
  string dbheading  ="db:";  
  if (fileheading==in.substr(0,fileheading.size()))
    { 
      string filename=in.substr(fileheading.size());
      edm::LogInfo("psetRetriever")<<"filename is --> "<<filename<<" <--";
      
      string   line;
      ifstream configFile(filename.c_str());
      while(std::getline(configFile,line)) {
	pset+=line;
	pset+="\n";
      }
    }
  else if (dbheading==in.substr(0,dbheading.size()))
    {
      XCEPT_RAISE(evf::Exception,"db access for ParameterSet not yet implemented");
    } 
  else
    {
      edm::LogWarning("psetRetriever")<<"Using direct config from XML";
      pset = in;
    }
}


//______________________________________________________________________________
string ParameterSetRetriever::getAsString() const
{
  return pset;
}
