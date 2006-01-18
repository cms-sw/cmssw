#ifndef Framework_ExternalInputSource_h
#define Framework_ExternalInputSource_h

/*----------------------------------------------------------------------
$Id: ExternalInputSource.h,v 1.1 2005/12/28 00:30:09 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include <string>

#include "FWCore/Framework/interface/ConfigurableInputSource.h"

namespace edm {
  class ExternalInputSource : public ConfigurableInputSource {
  public:
    explicit ExternalInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~ExternalInputSource();

  std::vector<std::string> const& fileNames() const {return fileNames_;}

  private:
    std::vector<std::string> fileNames_;
  };
}
#endif
