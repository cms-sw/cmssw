#ifndef IOPool_Input_RootInputFileSequence_h
#define IOPool_Input_RootInputFileSequence_h

/*----------------------------------------------------------------------

RootInputFileSequence: This is an InputSource

----------------------------------------------------------------------*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/ProductSelectorRules.h"
#include "FWCore/Framework/interface/ProcessingController.h"
#include "FWCore/Sources/interface/EventSkipperByID.h"
#include "FWCore/Sources/interface/VectorInputSource.h"
#include "FWCore/Utilities/interface/InputType.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace CLHEP {
  class HepRandomEngine;
}

namespace edm {

  class BranchID;
  class DuplicateChecker;
  class FileCatalogItem;
  class InputFileCatalog;
  class ParameterSetDescription;
  class PoolSource;
  class RootFile;

  class RootInputFileSequence {
  public:
    explicit RootInputFileSequence(ParameterSet const& pset,
                                   PoolSource& input,
                                   InputFileCatalog const& catalog,
                                   unsigned int nStreams,
                                   InputType inputType);
    virtual ~RootInputFileSequence();

    RootInputFileSequence(RootInputFileSequence const&) = delete; // Disallow copying and moving
    RootInputFileSequence& operator=(RootInputFileSequence const&) = delete; // Disallow copying and moving

    typedef std::shared_ptr<RootFile> RootFileSharedPtr;
    void readEvent(EventPrincipal& cache);
    std::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();
    void readLuminosityBlock_(LuminosityBlockPrincipal& lumiPrincipal);
    std::shared_ptr<RunAuxiliary> readRunAuxiliary_();
    void readRun_(RunPrincipal& runPrincipal);
    std::unique_ptr<FileBlock> readFile_();
    void closeFile_();
    void endJob();
    InputSource::ItemType getNextItemType(RunNumber_t& run, LuminosityBlockNumber_t& lumi, EventNumber_t& event);
    bool containedInCurrentFile(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const;
    void skipEntries(unsigned int offset);
    bool skipEvents(int offset);
    bool goToEvent(EventID const& eventID);
    bool skipToItem(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, size_t fileNameHash = 0U, bool currentFileFirst = true);
    bool skipToItemInNewFile(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event);
    bool skipToItemInNewFile(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, size_t fileNameHash);
    void rewind_();
    void readOneRandom(EventPrincipal& cache, size_t& fileNameHash, CLHEP::HepRandomEngine*);
    bool readOneRandomWithID(EventPrincipal& cache, size_t& fileNameHash, LuminosityBlockID const& id, CLHEP::HepRandomEngine*);
    bool readOneSequential(EventPrincipal& cache, size_t& fileNameHash);
    bool readOneSequentialWithID(EventPrincipal& cache, size_t& fileNameHash, LuminosityBlockID const& id);
    void readOneSpecified(EventPrincipal& cache, size_t& fileNameHash, SecondaryEventIDAndFileInfo const& id);

    void dropUnwantedBranches_(std::vector<std::string> const& wantedBranches);
    std::shared_ptr<ProductRegistry const> fileProductRegistry() const;
    std::shared_ptr<BranchIDListHelper const> fileBranchIDListHelper() const;
    ProcessHistoryRegistry const& processHistoryRegistry() const;
    ProcessHistoryRegistry& processHistoryRegistryForUpdate();
    static void fillDescription(ParameterSetDescription & desc);
    ProcessingController::ForwardState forwardState() const;
    ProcessingController::ReverseState reverseState() const;
    void initAssociationsFromSecondary(std::set<BranchID> const&);
  private:
    void initFile(bool skipBadFiles);
    bool nextFile();
    bool previousFile();
    void rewindFile();
    std::vector<FileCatalogItem> const& fileCatalogItems() const;

    std::shared_ptr<ProductRegistry const> productRegistry() const;
    ProcessConfiguration const& processConfiguration() const;
    ProductRegistry & productRegistryUpdate() const;
    int remainingEvents() const;
    int remainingLuminosityBlocks() const;

    PoolSource& input_;
    InputType inputType_;
    InputFileCatalog const& catalog_;
    bool firstFile_;
    std::string lfn_;
    size_t lfnHash_;
    std::unique_ptr<std::unordered_multimap<size_t, size_t> > findFileForSpecifiedID_;
    std::vector<FileCatalogItem>::const_iterator fileIterBegin_;
    std::vector<FileCatalogItem>::const_iterator fileIterEnd_;
    std::vector<FileCatalogItem>::const_iterator fileIter_;
    std::vector<FileCatalogItem>::const_iterator fileIterLastOpened_;
    RootFileSharedPtr rootFile_;
    BranchDescription::MatchMode branchesMustMatch_;

    std::vector<std::shared_ptr<IndexIntoFile> > indexesIntoFiles_;
    std::vector<ProcessHistoryID> orderedProcessHistoryIDs_;
    std::vector<BranchID> associationsFromSecondary_;

    unsigned int nStreams_; 
    std::shared_ptr<EventSkipperByID> eventSkipperByID_;
    int eventsRemainingInFile_;
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
    bool usedFallback_;
  }; // class RootInputFileSequence
}
#endif
