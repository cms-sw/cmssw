#ifndef IOPool_Input_PoolSource_h
#define IOPool_Input_PoolSource_h

/*----------------------------------------------------------------------

PoolSource: This is an InputSource

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProcessingController.h"
#include "FWCore/Framework/interface/ProductSelectorRules.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "IOPool/Common/interface/RootServiceChecker.h"

#include <array>
#include <memory>
#include <string>
#include <vector>

namespace edm {

  class ConfigurationDescriptions;
  class FileCatalogItem;
  class RootPrimaryFileSequence;
  class RootSecondaryFileSequence;
  class RunHelperBase;

  class PoolSource : public InputSource {
  public:
    explicit PoolSource(ParameterSet const& pset, InputSourceDescription const& desc);
    ~PoolSource() override;
    using InputSource::processHistoryRegistryForUpdate;
    using InputSource::productRegistryUpdate;

    // const accessors
    bool skipBadFiles() const {return skipBadFiles_;}
    bool dropDescendants() const {return dropDescendants_;}
    bool bypassVersionCheck() const {return bypassVersionCheck_;}
    bool labelRawDataLikeMC() const {return labelRawDataLikeMC_;}
    unsigned int nStreams() const {return nStreams_;}
    int treeMaxVirtualSize() const {return treeMaxVirtualSize_;}
    ProductSelectorRules const& productSelectorRules() const {return productSelectorRules_;}
    RunHelperBase* runHelper() {return runHelper_.get();}

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    void readEvent_(EventPrincipal& eventPrincipal) override;
    std::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_() override;
    void readLuminosityBlock_(LuminosityBlockPrincipal& lumiPrincipal) override;
    std::shared_ptr<RunAuxiliary> readRunAuxiliary_() override;
    void readRun_(RunPrincipal& runPrincipal) override;
    std::unique_ptr<FileBlock> readFile_() override;
    void closeFile_() override;
    void endJob() override;
    ItemType getNextItemType() override;
    bool readIt(EventID const& id, EventPrincipal& eventPrincipal, StreamContext& streamContext) override;
    void skip(int offset) override;
    bool goToEvent_(EventID const& eventID) override;
    void rewind_() override;
    bool randomAccess_() const override;
    ProcessingController::ForwardState forwardState_() const override;
    ProcessingController::ReverseState reverseState_() const override;

    std::pair<SharedResourcesAcquirer*,std::recursive_mutex*> resourceSharedWithDelayedReader_() override;
    
    RootServiceChecker rootServiceChecker_;
    InputFileCatalog catalog_;
    InputFileCatalog secondaryCatalog_;
    edm::propagate_const<std::shared_ptr<RunPrincipal>> secondaryRunPrincipal_;
    edm::propagate_const<std::shared_ptr<LuminosityBlockPrincipal>> secondaryLumiPrincipal_;
    std::vector<edm::propagate_const<std::unique_ptr<EventPrincipal>>> secondaryEventPrincipals_;
    std::array<std::vector<BranchID>, NumBranchTypes>  branchIDsToReplace_;

    unsigned int nStreams_;
    bool skipBadFiles_;
    bool bypassVersionCheck_;
    int const treeMaxVirtualSize_;
    ProductSelectorRules productSelectorRules_;
    bool dropDescendants_;
    bool labelRawDataLikeMC_;
    
    edm::propagate_const<std::unique_ptr<RunHelperBase>> runHelper_;
    std::unique_ptr<SharedResourcesAcquirer> resourceSharedWithDelayedReaderPtr_; // We do not use propagate_const because the acquirer is itself mutable.
    std::shared_ptr<std::recursive_mutex> mutexSharedWithDelayedReader_;
    edm::propagate_const<std::unique_ptr<RootPrimaryFileSequence>> primaryFileSequence_;
    edm::propagate_const<std::unique_ptr<RootSecondaryFileSequence>> secondaryFileSequence_;
  }; // class PoolSource
}
#endif
