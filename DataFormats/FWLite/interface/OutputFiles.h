#ifndef DataFormats_FWLite_interface_OutputFiles_h
#define DataFormats_FWLite_interface_OutputFiles_h

#include <vector>
#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/**
  \class    OutputFiles OutputFiles.h "DataFormats/FWLite/interface/OutputFiles.h"
  \brief    Helper class to handle FWLite file output names

  This is a very simple class to handle the appropriate python configuration of 
  output files in FWLite. 
*/

namespace fwlite {

  class OutputFiles {
  
  public:
    /// empty constructor 
    OutputFiles() {
      throw cms::Exception("InvalidInput") << "Must specify a string fileName" << std::endl;
    }
    /// default constructor from parameter set
    OutputFiles(const edm::ParameterSet& cfg) :
      file_( cfg.getParameter<edm::ParameterSet>("fwliteOutput").getParameter<std::string>("fileName") ) {};
    /// return output fuke name
    std::string const & file() const { return file_; }
      
  protected:
    /// output file name
    std::string file_;
  };
}

#endif
