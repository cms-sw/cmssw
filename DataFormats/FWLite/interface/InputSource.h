#ifndef DataFormats_FWLite_interface_InputSource_h
#define DataFormats_FWLite_interface_InputSource_h


/**
  \class    fwlite::InputSource "DataFormats/FWLite/interface/InputSource.h"
  \brief    Helper class to handle FWLite file input sources

  This is a very simple class to handle the appropriate python configuration
  of input files in FWLite. 

  \author   Salvatore Rappoccio
*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include <string>

namespace fwlite {
  class InputSource {
  public:
    InputSource() {
      throw cms::Exception("InvalidInput") << "Must specify a vstring fileNames" << std::endl;
    }
    InputSource( edm::ParameterSet const & pset ) :
      files_( pset.getParameter<edm::ParameterSet>("fwliteInput").getParameter<std::vector<std::string> >("fileNames") )
	{
	  if(pset.getParameter<edm::ParameterSet>("fwliteInput").exists("maxEvents")){ 
	    maxEvents_=pset.getParameter<edm::ParameterSet>("fwliteInput").getParameter<int>("maxEvents"); 
	  }
	  if(pset.getParameter<edm::ParameterSet>("fwliteInput").exists("outputEvery")){ 
	    reportAfter_=pset.getParameter<edm::ParameterSet>("fwliteInput").getParameter<unsigned int>("outputEvery"); 
	  }
	}

    std::vector<std::string> const & files() const { return files_; }

    int maxEvents() const { return maxEvents_; }
    unsigned int reportAfter() const { return reportAfter_;}

  protected:
    std::vector<std::string>  files_;
    int                       maxEvents_;
    unsigned int              reportAfter_;
  };
}

#endif
