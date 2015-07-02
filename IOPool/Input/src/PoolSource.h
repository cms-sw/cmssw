#ifndef IOPool_Input_PoolSource_h
#define IOPool_Input_PoolSource_h

/*----------------------------------------------------------------------

PoolSource: This is an InputSource

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/BranchType.h"
#include "FWCore/Catalog/interface/InputFileCatalog.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProcessingController.h"
#include "FWCore/Framework/interface/InputSource.h"
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

  class PoolSource : public InputSource {
  public:
    explicit PoolSource(ParameterSet const& pset, InputSourceDescription const& desc);
    virtual ~PoolSource();
    using InputSource::processHistoryRegistryForUpdate;
    using InputSource::productRegistryUpdate;

    static void fillDescriptions(ConfigurationDescriptions& descriptions);

  private:
    virtual void readEvent_(EventPrincipal& eventPrincipal);
    virtual std::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();
    virtual void readLuminosityBlock_(LuminosityBlockPrincipal& lumiPrincipal);
    virtual std::shared_ptr<RunAuxiliary> readRunAuxiliary_();
    virtual void readRun_(RunPrincipal& runPrincipal);
    virtual std::unique_ptr<FileBlock> readFile_();
    virtual void closeFile_();
    virtual void endJob();
    virtual ItemType getNextItemType();
    virtual bool readIt(EventID const& id, EventPrincipal& eventPrincipal, StreamContext& streamContext) override;
    virtual void skip(int offset);
    virtual bool goToEvent_(EventID const& eventID);
    virtual void rewind_();
    virtual void preForkReleaseResources();
    virtual bool randomAccess_() const;
    virtual ProcessingController::ForwardState forwardState_() const;
    virtual ProcessingController::ReverseState reverseState_() const;

    SharedResourcesAcquirer* resourceSharedWithDelayedReader_() const override;
    
    RootServiceChecker rootServiceChecker_;
    InputFileCatalog catalog_;
    InputFileCatalog secondaryCatalog_;
    std::unique_ptr<RootPrimaryFileSequence> primaryFileSequence_;
    std::unique_ptr<RootSecondaryFileSequence> secondaryFileSequence_;
    std::shared_ptr<RunPrincipal> secondaryRunPrincipal_;
    std::shared_ptr<LuminosityBlockPrincipal> secondaryLumiPrincipal_;
    std::vector<std::unique_ptr<EventPrincipal>> secondaryEventPrincipals_;
    std::array<std::vector<BranchID>, NumBranchTypes>  branchIDsToReplace_;
    
    std::unique_ptr<SharedResourcesAcquirer> resourceSharedWithDelayedReaderPtr_;
  }; // class PoolSource
}
#endif
