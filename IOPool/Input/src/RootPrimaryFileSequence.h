#ifndef IOPool_Input_RootPrimaryFileSequence_h
#define IOPool_Input_RootPrimaryFileSequence_h

/*----------------------------------------------------------------------

RootPrimaryFileSequence: This is an InputSource

----------------------------------------------------------------------*/

#include "RootInputFileSequence.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Framework/interface/ProductSelectorRules.h"
#include "FWCore/Framework/interface/ProcessingController.h"
#include "FWCore/Sources/interface/EventSkipperByID.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

#include <memory>
#include <string>
#include <vector>

namespace edm {

  class BranchID;
  class DuplicateChecker;
  class FileCatalogItem;
  class InputFileCatalog;
  class ParameterSetDescription;
  class PoolSource;
  class RootFile;

  class RootPrimaryFileSequence : public RootInputFileSequence{
  public:
    explicit RootPrimaryFileSequence(ParameterSet const& pset,
                                   PoolSource& input,
                                   InputFileCatalog const& catalog,
                                   unsigned int nStreams);
    virtual ~RootPrimaryFileSequence();

    RootPrimaryFileSequence(RootPrimaryFileSequence const&) = delete; // Disallow copying and moving
    RootPrimaryFileSequence& operator=(RootPrimaryFileSequence const&) = delete; // Disallow copying and moving

    typedef std::shared_ptr<RootFile> RootFileSharedPtr;
    std::unique_ptr<FileBlock> readFile_();
    virtual void closeFile_() override;
    void endJob();
    InputSource::ItemType getNextItemType(RunNumber_t& run, LuminosityBlockNumber_t& lumi, EventNumber_t& event);
    bool skipEvents(int offset);
    bool goToEvent(EventID const& eventID);
    void rewind_();
    static void fillDescription(ParameterSetDescription & desc);
    ProcessingController::ForwardState forwardState() const;
    ProcessingController::ReverseState reverseState() const;
  private:
    virtual void initFile_(bool skipBadFiles) override;
    virtual RootFileSharedPtr makeRootFile(std::shared_ptr<InputFile> filePtr) override; 
    bool nextFile();
    bool previousFile();
    void rewindFile();

    int remainingEvents() const;
    int remainingLuminosityBlocks() const;

    PoolSource& input_;
    bool firstFile_;
    BranchDescription::MatchMode branchesMustMatch_;
    std::vector<ProcessHistoryID> orderedProcessHistoryIDs_;

    unsigned int nStreams_; 
    std::shared_ptr<EventSkipperByID> eventSkipperByID_;
    int initialNumberOfEventsToSkip_;
    bool noEventSort_;
    bool skipBadFiles_;
    bool bypassVersionCheck_;
    unsigned int treeCacheSize_;
    int const treeMaxVirtualSize_;
    RunNumber_t setRun_;
    ProductSelectorRules productSelectorRules_;
    std::shared_ptr<DuplicateChecker> duplicateChecker_;
    bool dropDescendants_;
    bool labelRawDataLikeMC_;
    bool usingGoToEvent_;
    bool enablePrefetching_;
  }; // class RootPrimaryFileSequence
}
#endif
