#include "EventFilter/Utilities/interface/ParameterSetRetriever.h"
#include "EventFilter/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <fstream>

namespace evf{
  //______________________________________________________________________________
  ParameterSetRetriever::ParameterSetRetriever(const std::string& in)
  {
    using std::string;
    using std::ifstream; 
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
	edm::LogWarning("psetRetriever")<<"Using string cfg from RunControl or XML";
	pset = in;
      }
  }


  //______________________________________________________________________________
  std::string ParameterSetRetriever::getAsString() const
  {
    return pset;
  }
} //end namespace evf
