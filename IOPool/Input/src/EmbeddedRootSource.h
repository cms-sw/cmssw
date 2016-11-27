#ifndef IOPool_Input_EmbeddedRootSource_h
#define IOPool_Input_EmbeddedRootSource_h

/*----------------------------------------------------------------------

EmbeddedRootSource: This is an InputSource

----------------------------------------------------------------------*/

#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProductSelectorRules.h"
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "FWCore/Utilities/interface/propagate_const.h"
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
  class RunHelperBase;
  class RootEmbeddedFileSequence;
  struct VectorInputSourceDescription;

  class EmbeddedRootSource : public VectorInputSource {
  public:
    explicit EmbeddedRootSource(ParameterSet const& pset, VectorInputSourceDescription const& desc);
    virtual ~EmbeddedRootSource();
    using VectorInputSource::processHistoryRegistryForUpdate;
    using VectorInputSource::productRegistryUpdate;

    // const accessors
    bool skipBadFiles() const {return skipBadFiles_;}
    bool bypassVersionCheck() const {return bypassVersionCheck_;}
    unsigned int nStreams() const {return nStreams_;}
    int treeMaxVirtualSize() const {return treeMaxVirtualSize_;}
    ProductSelectorRules const& productSelectorRules() const {return productSelectorRules_;}
    RunHelperBase* runHelper() {return runHelper_.get();}

    static void fillDescriptions(ConfigurationDescriptions & descriptions);

  private:
    virtual void closeFile_();
    virtual void beginJob() override;
    virtual void endJob() override;
    virtual bool readOneEvent(EventPrincipal& cache, size_t& fileNameHash, CLHEP::HepRandomEngine*, EventID const* id) override;
    virtual void readOneSpecified(EventPrincipal& cache, size_t& fileNameHash, SecondaryEventIDAndFileInfo const& id) override;
    virtual void dropUnwantedBranches_(std::vector<std::string> const& wantedBranches) override;
    
    RootServiceChecker rootServiceChecker_;

    unsigned int nStreams_;
    bool skipBadFiles_;
    bool bypassVersionCheck_;
    int const treeMaxVirtualSize_;
    ProductSelectorRules productSelectorRules_;
    std::unique_ptr<RunHelperBase> runHelper_;

    InputFileCatalog catalog_;
    edm::propagate_const<std::unique_ptr<RootEmbeddedFileSequence>> fileSequence_;
    
  }; // class EmbeddedRootSource
}
#endif
