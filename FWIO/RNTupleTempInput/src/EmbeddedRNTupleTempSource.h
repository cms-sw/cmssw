#ifndef FWIO_RNTupleTempInput_EmbeddedRNTupleTempSource_h
#define FWIO_RNTupleTempInput_EmbeddedRNTupleTempSource_h

/*----------------------------------------------------------------------

EmbeddedRNTupleTempSource: This is an InputSource

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
  class InputSourceRunHelperBase;
  struct VectorInputSourceDescription;
}  // namespace edm
namespace edm::rntuple_temp {
  class RootEmbeddedFileSequence;

  class EmbeddedRNTupleTempSource : public VectorInputSource {
  public:
    struct Optimizations {
      bool useClusterCache = true;
    };

    explicit EmbeddedRNTupleTempSource(ParameterSet const& pset, VectorInputSourceDescription const& desc);
    ~EmbeddedRNTupleTempSource() override;
    using VectorInputSource::processHistoryRegistryForUpdate;
    using VectorInputSource::productRegistryUpdate;

    // const accessors
    bool skipBadFiles() const { return skipBadFiles_; }
    bool bypassVersionCheck() const { return bypassVersionCheck_; }
    unsigned int nStreams() const { return nStreams_; }
    int treeMaxVirtualSize() const { return treeMaxVirtualSize_; }
    Optimizations const& optimizations() const { return optimizations_; }
    ProductSelectorRules const& productSelectorRules() const { return productSelectorRules_; }
    InputSourceRunHelperBase* runHelper() { return runHelper_.get(); }

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    virtual void closeFile_();
    void beginJob() override;
    void endJob() override;
    bool readOneEvent(EventPrincipal& cache,
                      size_t& fileNameHash,
                      CLHEP::HepRandomEngine*,
                      EventID const* id,
                      bool recycleFiles) override;
    void readOneSpecified(EventPrincipal& cache, size_t& fileNameHash, SecondaryEventIDAndFileInfo const& id) override;
    void dropUnwantedBranches_(std::vector<std::string> const& wantedBranches) override;

    RootServiceChecker rootServiceChecker_;

    unsigned int nStreams_;
    bool skipBadFiles_;
    bool bypassVersionCheck_;
    int const treeMaxVirtualSize_;
    Optimizations optimizations_;
    ProductSelectorRules productSelectorRules_;
    std::unique_ptr<InputSourceRunHelperBase> runHelper_;

    InputFileCatalog catalog_;
    edm::propagate_const<std::unique_ptr<RootEmbeddedFileSequence>> fileSequence_;

  };  // class EmbeddedRNTupleTempSource
}  // namespace edm::rntuple_temp
#endif
