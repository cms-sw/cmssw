#ifndef IOPool_Input_EmbeddedRootSource_h
#define IOPool_Input_EmbeddedRootSource_h

/*----------------------------------------------------------------------

EmbeddedRootSource: This is an InputSource

----------------------------------------------------------------------*/

#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "IOPool/Common/interface/RootServiceChecker.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {

  class ConfigurationDescriptions;
  class FileCatalogItem;
  class RootEmbeddedFileSequence;
  struct VectorInputSourceDescription;

  class EmbeddedRootSource : public VectorInputSource {
  public:
    explicit EmbeddedRootSource(ParameterSet const& pset, VectorInputSourceDescription const& desc);
    virtual ~EmbeddedRootSource();
    using VectorInputSource::processHistoryRegistryForUpdate;
    using VectorInputSource::productRegistryUpdate;

    static void fillDescriptions(ConfigurationDescriptions & descriptions);

  private:
    virtual void closeFile_();
    virtual void beginJob();
    virtual void endJob();
    virtual bool readOneEvent(EventPrincipal& cache, size_t& fileNameHash, CLHEP::HepRandomEngine*, EventID const* id) override;
    virtual void readOneSpecified(EventPrincipal& cache, size_t& fileNameHash, SecondaryEventIDAndFileInfo const& id);
    virtual void dropUnwantedBranches_(std::vector<std::string> const& wantedBranches);
    
    RootServiceChecker rootServiceChecker_;
    InputFileCatalog catalog_;
    std::unique_ptr<RootEmbeddedFileSequence> fileSequence_;
    
  }; // class EmbeddedRootSource
}
#endif
