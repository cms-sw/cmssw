#ifndef Framework_ExternalInputSource_h
#define Framework_ExternalInputSource_h

/*----------------------------------------------------------------------
$Id: ExternalInputSource.h,v 1.1 2005/10/17 19:22:41 wmtan Exp $
----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include <string>

#include "FWCore/Framework/interface/GenericInputSource.h"

namespace edm {
  class ExternalInputSource : public GenericInputSource {
  public:
    explicit ExternalInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~ExternalInputSource();

  std::vector<std::string> const& fileNames() const {return fileNames_;}

  private:
    std::vector<std::string> fileNames_;
  };
}
#endif
