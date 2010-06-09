#ifndef FWCore_Sources_EDInputSource_h
#define FWCore_Sources_EDInputSource_h

/*----------------------------------------------------------------------
$Id: EDInputSource.h,v 1.6 2010/01/04 16:10:59 wdd Exp $
----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include <vector>
#include <string>

namespace edm {
  class InputSourceDescription;
  class ParameterSet;
  class ParameterSetDescription;

  class EDInputSource : public InputSource {
  public:
    explicit EDInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~EDInputSource();

    std::vector<std::string> const& logicalFileNames(int n = 0) const {
      return n ? secondaryCatalog_.logicalFileNames() : catalog_.logicalFileNames();
    }
    std::vector<std::string> const& fileNames(int n = 0) const {
      return n ? secondaryCatalog_.fileNames() : catalog_.fileNames();
    }
    std::vector<FileCatalogItem> const& fileCatalogItems(int n = 0) const {
      return n ? secondaryCatalog_.fileCatalogItems() : catalog_.fileCatalogItems();
    }
    InputFileCatalog& catalog(int n = 0) {return n ? secondaryCatalog_ : catalog_;}

    static void fillDescription(ParameterSetDescription & desc);

  private:
    virtual void setRun(RunNumber_t);
    virtual void setLumi(LuminosityBlockNumber_t lb);
    
    InputFileCatalog catalog_;
    InputFileCatalog secondaryCatalog_;
  };
}
#endif
