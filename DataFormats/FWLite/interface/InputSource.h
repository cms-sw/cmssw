#ifndef DataFormats_FWLite_interface_InputSource_h
#define DataFormats_FWLite_interface_InputSource_h

#include <vector>
#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"

/**
  \class    InputSource InputSource.h "DataFormats/FWLite/interface/InputSource.h"
  \brief    Helper class to handle FWLite file input sources

  This is a very simple class to handle the appropriate python configuration of 
  input files in FWLite. 
*/

namespace fwlite {

  class InputSource {
  public:
    /// empty constructor
    InputSource() { throw cms::Exception("InvalidInput") << "Must specify a vstring fileNames" << std::endl; }
    /// default constructor from parameter set
    InputSource(const edm::ParameterSet& cfg)
        : maxEvents_(-1),
          reportAfter_(10),
          files_(cfg.getParameterSet("fwliteInput").getParameter<std::vector<std::string> >("fileNames")) {
      // optional parameter
      if (cfg.getParameterSet("fwliteInput").existsAs<int>("maxEvents")) {
        maxEvents_ = cfg.getParameterSet("fwliteInput").getParameter<int>("maxEvents");
      }
      // optional parameter
      if (cfg.getParameterSet("fwliteInput").existsAs<unsigned int>("outputEvery")) {
        reportAfter_ = cfg.getParameterSet("fwliteInput").getParameter<unsigned int>("outputEvery");
      }
    }
    /// return vector of files_
    const std::vector<std::string>& files() const { return files_; }
    /// return maxEvetns_
    int maxEvents() const { return maxEvents_; }
    /// return reportAfter_
    unsigned int reportAfter() const { return reportAfter_; }

  protected:
    /// maximal number of events to loop
    int maxEvents_;
    /// report after N events
    unsigned int reportAfter_;
    /// vector of input files
    std::vector<std::string> files_;
  };
}  // namespace fwlite

#endif
