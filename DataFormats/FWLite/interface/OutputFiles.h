#ifndef DataFormats_FWLite_interface_OutputFiles_h
#define DataFormats_FWLite_interface_OutputFiles_h


/**
  \class    fwlite::OutputFiles "DataFormats/FWLite/interface/OutputFiles.h"
  \brief    Helper class to handle FWLite file output names

  This is a very simple class to handle the appropriate python configuration
  of output files in FWLite. 

  \author   Salvatore Rappoccio
*/

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <vector>
#include <string>

namespace fwlite {
  class OutputFiles {
  public:
    OutputFiles() {
      throw cms::Exception("InvalidInput") << "Must specify a string fileName" << std::endl;
    }
    OutputFiles( edm::ParameterSet const & pset ) :
    file_( pset.getParameter<edm::ParameterSet>("fwliteOutput").getParameter<std::string>("fileName") )
	{
	}

      std::string const & file() const { return file_; }

  protected:
      std::string file_;
    
  };
}

#endif
