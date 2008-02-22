#ifndef FWCore_Sources_EDInputSource_h
#define FWCore_Sources_EDInputSource_h

/*----------------------------------------------------------------------
$Id: EDInputSource.h,v 1.3 2007/08/06 19:53:06 wmtan Exp $
----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/RunID.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Catalog/interface/FileCatalog.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include <vector>
#include <string>

namespace edm {
  class InputSourceDescription;
  class ParameterSet;
  class EDInputSource : public InputSource {
  public:
    explicit EDInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~EDInputSource();

    std::vector<std::string> const& logicalFileNames(int n = 0) const {
      return catalogs_.at(n).logicalFileNames();
    }
    std::vector<std::string> const& fileNames(int n = 0) const {
      return catalogs_.at(n).fileNames();
    }
    std::vector<FileCatalogItem> const& fileCatalogItems(int n = 0) const {
      return catalogs_.at(n).fileCatalogItems();
    }
    InputFileCatalog& catalog(int n = 0) {return catalogs_.at(n);}

  private:
    virtual void setRun(RunNumber_t);
    virtual void setLumi(LuminosityBlockNumber_t lb);
    
    std::vector<InputFileCatalog> catalogs_;
  };
}
#endif
