#ifndef FWCore_Sources_ExternalInputSource_h
#define FWCore_Sources_ExternalInputSource_h

/*----------------------------------------------------------------------
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
    
    static void fillDescription(ParameterSetDescription& desc);

  protected:
    void incrementFileIndex() {++fileIndex_;}

  private:
    virtual size_t fileIndex() const {return fileIndex_;}

    size_t fileIndex_;
    InputFileCatalog catalog_;
  };
}
#endif
