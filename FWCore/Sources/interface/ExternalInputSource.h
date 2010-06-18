#ifndef FWCore_Sources_ExternalInputSource_h
#define FWCore_Sources_ExternalInputSource_h

/*----------------------------------------------------------------------
$Id: ExternalInputSource.h,v 1.6 2010/06/09 07:33:58 innocent Exp $
----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include <string>

#include "FWCore/Framework/interface/ConfigurableInputSource.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;

  class ExternalInputSource : public ConfigurableInputSource {
  public:
    ExternalInputSource(ParameterSet const& pset, InputSourceDescription const& desc, bool realData = true);
    virtual ~ExternalInputSource();

    std::vector<std::string> const& logicalFileNames() const {return catalog_.logicalFileNames();}
    std::vector<std::string> const& fileNames() const {return catalog_.fileNames();}
    InputFileCatalog& catalog() {return catalog_;}
    
    static void fillDescription(ParameterSetDescription & desc);

  private:
    InputFileCatalog catalog_;
  };
}
#endif
