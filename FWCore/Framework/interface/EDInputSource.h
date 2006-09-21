#ifndef Framework_EDInputSource_h
#define Framework_EDInputSource_h

/*----------------------------------------------------------------------
$Id: EDInputSource.h,v 1.3 2006/08/14 23:27:48 wmtan Exp $
----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/FileCatalog.h"
#include <vector>
#include <string>

namespace edm {
  class InputSourceDescription;
  class ParameterSet;
  class EDInputSource : public InputSource {
  public:
    explicit EDInputSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~EDInputSource();

    std::vector<std::string> const& logicalFileNames() const {return catalog_.logicalFileNames();}
    std::vector<std::string> const& fileNames() const {return catalog_.fileNames();}
    std::vector<FileCatalogItem> const& fileCatalogItems() const {return catalog_.fileCatalogItems();}
    InputFileCatalog& catalog() {return catalog_;}

  private:
    virtual void setRun(RunNumber_t);
    
    InputFileCatalog catalog_;
  };
}
#endif
