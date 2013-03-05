#ifndef FWCore_Sources_ProducerSourceFromFiles_h
#define FWCore_Sources_ProducerSourceFromFiles_h

/*----------------------------------------------------------------------
----------------------------------------------------------------------*/

#include <memory>
#include <vector>
#include <string>

#include "FWCore/Sources/interface/ProducerSourceBase.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"

namespace edm {
  class ParameterSet;
  class ParameterSetDescription;

  class ProducerSourceFromFiles : public ProducerSourceBase {
  public:
    ProducerSourceFromFiles(ParameterSet const& pset, InputSourceDescription const& desc, bool realData);
    virtual ~ProducerSourceFromFiles();

    std::vector<std::string> const& logicalFileNames() const {return catalog_.logicalFileNames();}
    std::vector<std::string> const& fileNames() const {return catalog_.fileNames();}
    InputFileCatalog& catalog() {return catalog_;}
    
    static void fillDescription(ParameterSetDescription& desc);

  private:
    virtual bool noFiles() const;

    InputFileCatalog catalog_;
  };
}
#endif
