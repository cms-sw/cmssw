#ifndef IOPool_Input_RootFile_h
#define IOPool_Input_RootFile_h

/*----------------------------------------------------------------------

RootFile.h // used by ROOT input sources

----------------------------------------------------------------------*/

#include "RootTree.h"
#include "DataFormats/Provenance/interface/ProductDependencies.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/EntryDescriptionID.h"  // backward compatibility
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventEntryDescription.h"  // backward compatibility
#include "DataFormats/Provenance/interface/EventProcessHistoryID.h"  // backward compatibility
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/EventToProcessBlockIndexes.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/History.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "DataFormats/Provenance/interface/ParentageID.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/Common/interface/FWCoreCommonFwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Utilities/interface/InputType.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include "TBranch.h"

#include <array>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace edm {

  //------------------------------------------------------------
  // Class RootFile: supports file reading.

  class BranchID;
  class BranchIDListHelper;
  class ProductProvenanceRetriever;
  struct DaqProvenanceHelper;
  class DuplicateChecker;
  class EventSkipperByID;
  class ProcessHistoryRegistry;
  class ProductSelectorRules;
  class InputFile;
  class ProvenanceReaderBase;
  class ProvenanceAdaptor;
  class StoredMergeableRunProductMetadata;
  class RunHelperBase;
  class ThinnedAssociationsHelper;

  using EntryDescriptionMap = std::map<EntryDescriptionID, EventEntryDescription>;

  class MakeProvenanceReader {
  public:
    virtual std::unique_ptr<ProvenanceReaderBase> makeReader(RootTree& eventTree,
                                                             DaqProvenanceHelper const* daqProvenanceHelper) const = 0;
    virtual ~MakeProvenanceReader() = default;
  };

  class RootFile {
  public:
    struct FileOptions {
      std::string const& fileName;
      std::string const& logicalFileName;
      std::shared_ptr<InputFile> filePtr;
      bool bypassVersionCheck;
      bool enforceGUIDInFileName;
    };

    struct ProcessingOptions {
      std::shared_ptr<EventSkipperByID> eventSkipperByID{};
      bool skipAnyEvents = false;
      int remainingEvents = -1;
      int remainingLumis = -1;
      InputSource::ProcessingMode processingMode = InputSource::RunsLumisAndEvents;
      bool noRunLumiSort = false;
      bool noEventSort = false;
      bool usingGoToEvent = false;
    };
    struct TTreeOptions {
      unsigned int treeCacheSize = 0U;
      int treeMaxVirtualSize;
      bool enablePrefetching;
    };

    struct ProductChoices {
      ProductSelectorRules const& productSelectorRules;
      std::vector<BranchID> const* associationsFromSecondary = nullptr;
      bool dropDescendantsOfDroppedProducts = false;
      bool labelRawDataLikeMC = false;
    };

    struct CrossFileInfo {
      RunHelperBase* runHelper = nullptr;
      std::shared_ptr<BranchIDListHelper> branchIDListHelper{};
      ProcessBlockHelper* processBlockHelper = 0;
      std::shared_ptr<ThinnedAssociationsHelper> thinnedAssociationsHelper{};
      std::shared_ptr<DuplicateChecker> duplicateChecker{};
      std::vector<std::shared_ptr<IndexIntoFile>> const& indexesIntoFiles;  //duplicate checking
      std::vector<std::shared_ptr<IndexIntoFile>>::size_type currentIndexIntoFile;
    };

    RootFile(FileOptions&& fileOptions,
             InputType inputType,
             ProcessingOptions&& processingOptions,
             TTreeOptions&& ttreeOptions,
             ProductChoices&& productChoices,
             CrossFileInfo&& crossFileInfo,
             unsigned int nStreams,
             ProcessHistoryRegistry& processHistoryRegistry,
             std::vector<ProcessHistoryID>& orderedProcessHistoryIDs);
    ~RootFile();

    RootFile(RootFile const&) = delete;             // Disallow copying and moving
    RootFile& operator=(RootFile const&) = delete;  // Disallow copying and moving

    void reportOpened(std::string const& inputType);
    void close();
    std::tuple<bool, bool> readCurrentEvent(EventPrincipal& cache, bool assertOnFailure = true);
    bool readEvent(EventPrincipal& cache);

    std::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();
    std::shared_ptr<RunAuxiliary> readRunAuxiliary_();
    std::shared_ptr<RunAuxiliary> readFakeRunAuxiliary_();

    void fillProcessBlockHelper_();
    bool initializeFirstProcessBlockEntry();
    bool endOfProcessBlocksReached() const;
    bool nextProcessBlock_(ProcessBlockPrincipal&);
    void readProcessBlock_(ProcessBlockPrincipal&);

    bool readRun_(RunPrincipal& runPrincipal);
    void readFakeRun_(RunPrincipal& runPrincipal);
    bool readLuminosityBlock_(LuminosityBlockPrincipal& lumiPrincipal);
    std::string const& file() const { return file_; }
    std::shared_ptr<ProductRegistry const> productRegistry() const { return productRegistry_; }
    // IndexIntoFile::EntryNumber_t const& entryNumber() const {return indexIntoFileIter().entry();}
    // LuminosityBlockNumber_t const& luminosityBlockNumber() const {return indexIntoFileIter().lumi();}
    // RunNumber_t const& runNumber() const {return indexIntoFileIter().run();}
    RootTree const& eventTree() const { return eventTree_; }
    RootTree const& lumiTree() const { return lumiTree_; }
    RootTree const& runTree() const { return runTree_; }
    FileFormatVersion fileFormatVersion() const { return fileFormatVersion_; }
    int whyNotFastClonable() const { return whyNotFastClonable_; }
    std::array<bool, NumBranchTypes> const& hasNewlyDroppedBranch() const { return hasNewlyDroppedBranch_; }
    bool branchListIndexesUnchanged() const { return branchListIndexesUnchanged_; }
    bool modifiedIDs() const { return daqProvenanceHelper_.get() != nullptr; }
    std::shared_ptr<FileBlock> createFileBlock();
    void updateFileBlock(FileBlock&);

    bool setEntryAtItem(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) {
      return (event != 0) ? setEntryAtEvent(run, lumi, event) : (lumi ? setEntryAtLumi(run, lumi) : setEntryAtRun(run));
    }
    bool containsItem(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) const;
    bool setEntryAtEvent(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event);
    bool setEntryAtLumi(RunNumber_t run, LuminosityBlockNumber_t lumi);
    bool setEntryAtRun(RunNumber_t run);
    bool setEntryAtNextEventInLumi(RunNumber_t run, LuminosityBlockNumber_t lumi);
    void setAtEventEntry(IndexIntoFile::EntryNumber_t entry);

    void rewind() {
      indexIntoFileIter_ = indexIntoFileBegin_;
      eventTree_.rewind();
      lumiTree_.rewind();
      runTree_.rewind();
      currentProcessBlockTree_ = 0;
      for (auto& processBlockTree : processBlockTrees_) {
        processBlockTree->rewindToInvalid();
      }
    }
    void setToLastEntry() { indexIntoFileIter_ = indexIntoFileEnd_; }

    bool skipEntries(unsigned int& offset) { return eventTree_.skipEntries(offset); }
    bool skipEvents(int& offset);
    bool goToEvent(EventID const& eventID);
    bool nextEventEntry() { return eventTree_.nextWithCache(); }
    IndexIntoFile::EntryType getNextItemType(RunNumber_t& run, LuminosityBlockNumber_t& lumi, EventNumber_t& event);
    std::shared_ptr<BranchIDListHelper const> branchIDListHelper() const {
      return get_underlying_safe(branchIDListHelper_);
    }
    std::shared_ptr<BranchIDListHelper>& branchIDListHelper() { return get_underlying_safe(branchIDListHelper_); }
    std::shared_ptr<IndexIntoFile const> indexIntoFileSharedPtr() const {
      return get_underlying_safe(indexIntoFileSharedPtr_);
    }
    std::shared_ptr<IndexIntoFile>& indexIntoFileSharedPtr() { return get_underlying_safe(indexIntoFileSharedPtr_); }
    bool wasLastEventJustRead() const;
    bool wasFirstEventJustRead() const;
    IndexIntoFile::IndexIntoFileItr indexIntoFileIter() const;
    void setPosition(IndexIntoFile::IndexIntoFileItr const& position);
    void initAssociationsFromSecondary(std::vector<BranchID> const&);

    void setSignals(
        signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadSource,
        signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadSource);

  private:
    void makeProcessBlockRootTrees(std::shared_ptr<InputFile> filePtr,
                                   int treeMaxVirtualSize,
                                   bool enablePrefetching,
                                   InputType inputType,
                                   StoredProcessBlockHelper const& storedProcessBlockHelper);
    bool skipThisEntry();
    void setIfFastClonable(int remainingEvents, int remainingLumis);
    void validateFile(InputType inputType,
                      bool usingGoToEvent,
                      std::vector<ProcessHistoryID>& orderedProcessHistoryIDs);
    void fillIndexIntoFile();
    EventAuxiliary fillEventAuxiliary(IndexIntoFile::EntryNumber_t entry);
    EventAuxiliary const& fillThisEventAuxiliary();
    void fillEventToProcessBlockIndexes();
    bool fillEventHistory(EventAuxiliary& evtAux,
                          EventSelectionIDVector& eventSelectionIDs,
                          BranchListIndexes& branchListIndexes,
                          bool assertOnFailure = true);
    std::shared_ptr<LuminosityBlockAuxiliary> fillLumiAuxiliary();
    std::shared_ptr<RunAuxiliary> fillRunAuxiliary();
    std::string const& newBranchToOldBranch(std::string const& newBranch) const;
    void setPresenceInProductRegistry(ProductRegistry&, StoredProcessBlockHelper const&);
    void markBranchToBeDropped(bool dropDescendants,
                               ProductDescription const& branch,
                               std::set<BranchID>& branchesToDrop,
                               std::map<BranchID, BranchID> const& droppedToKeptAlias) const;
    void dropOnInputAndReorder(ProductRegistry&,
                               ProductSelectorRules const&,
                               bool dropDescendants,
                               InputType,
                               StoredProcessBlockHelper&,
                               ProcessBlockHelper const*);
    void dropProcessesAndReorder(StoredProcessBlockHelper&,
                                 std::set<std::string> const& processesWithKeptProcessBlockProducts,
                                 ProcessBlockHelper const*);

    void readParentageTree(InputType inputType);
    void readEntryDescriptionTree(EntryDescriptionMap& entryDescriptionMap,
                                  InputType inputType);  // backward compatibility
    void readEventHistoryTree();
    bool isDuplicateEvent();

    void initializeDuplicateChecker(std::vector<std::shared_ptr<IndexIntoFile>> const& indexesIntoFiles,
                                    std::vector<std::shared_ptr<IndexIntoFile>>::size_type currentIndexIntoFile);

    std::unique_ptr<MakeProvenanceReader> makeProvenanceReaderMaker(InputType inputType);
    std::shared_ptr<ProductProvenanceRetriever> makeProductProvenanceRetriever(unsigned int iStreamIndex);

    std::shared_ptr<RunAuxiliary const> savedRunAuxiliary() const { return get_underlying_safe(savedRunAuxiliary_); }
    std::shared_ptr<RunAuxiliary>& savedRunAuxiliary() { return get_underlying_safe(savedRunAuxiliary_); }

    std::shared_ptr<ProductDependencies const> productDependencies() const {
      return get_underlying_safe(productDependencies_);
    }
    std::shared_ptr<ProductDependencies>& productDependencies() { return get_underlying_safe(productDependencies_); }

    std::shared_ptr<ProductProvenanceRetriever const> eventProductProvenanceRetriever(size_t index) const {
      return get_underlying_safe(eventProductProvenanceRetrievers_[index]);
    }
    std::shared_ptr<ProductProvenanceRetriever>& eventProductProvenanceRetriever(size_t index) {
      return get_underlying_safe(eventProductProvenanceRetrievers_[index]);
    }

    std::string const file_;
    std::string const logicalFile_;
    edm::propagate_const<ProcessHistoryRegistry*> processHistoryRegistry_;  // We don't own this
    edm::propagate_const<std::shared_ptr<InputFile>> filePtr_;
    edm::propagate_const<std::shared_ptr<EventSkipperByID>> eventSkipperByID_;
    FileFormatVersion fileFormatVersion_;
    FileID fid_;
    edm::propagate_const<std::shared_ptr<IndexIntoFile>> indexIntoFileSharedPtr_;
    IndexIntoFile& indexIntoFile_;
    IndexIntoFile::IndexIntoFileItr indexIntoFileBegin_;
    IndexIntoFile::IndexIntoFileItr indexIntoFileEnd_;
    IndexIntoFile::IndexIntoFileItr indexIntoFileIter_;
    edm::propagate_const<std::unique_ptr<StoredMergeableRunProductMetadata>> storedMergeableRunProductMetadata_;
    std::vector<EventProcessHistoryID> eventProcessHistoryIDs_;                   // backward compatibility
    std::vector<EventProcessHistoryID>::const_iterator eventProcessHistoryIter_;  // backward compatibility
    edm::propagate_const<std::shared_ptr<RunAuxiliary>> savedRunAuxiliary_;
    bool skipAnyEvents_;
    bool noRunLumiSort_;
    bool noEventSort_;
    bool enforceGUIDInFileName_;
    int whyNotFastClonable_;
    std::array<bool, NumBranchTypes> hasNewlyDroppedBranch_;
    bool branchListIndexesUnchanged_;
    EventAuxiliary eventAuxCache_;  //Should only be used by fillThisEventAuxiliary()
    RootTree eventTree_;
    RootTree lumiTree_;
    RootTree runTree_;
    std::vector<edm::propagate_const<std::unique_ptr<RootTree>>> processBlockTrees_;
    unsigned int currentProcessBlockTree_ = 0;
    std::vector<edm::propagate_const<RootTree*>> treePointers_;
    //Should only be used by fillThisEventAuxiliary()
    IndexIntoFile::EntryNumber_t lastEventEntryNumberRead_;
    std::shared_ptr<ProductRegistry const> productRegistry_;
    std::shared_ptr<BranchIDLists const> branchIDLists_;
    edm::propagate_const<std::shared_ptr<BranchIDListHelper>> branchIDListHelper_;
    edm::propagate_const<ProcessBlockHelper*> processBlockHelper_;
    edm::propagate_const<std::unique_ptr<StoredProcessBlockHelper>> storedProcessBlockHelper_;
    edm::propagate_const<std::unique_ptr<ThinnedAssociationsHelper>> fileThinnedAssociationsHelper_;
    edm::propagate_const<std::shared_ptr<ThinnedAssociationsHelper>> thinnedAssociationsHelper_;
    InputSource::ProcessingMode processingMode_;
    edm::propagate_const<RunHelperBase*> runHelper_;
    std::map<std::string, std::string> newBranchToOldBranch_;
    edm::propagate_const<TTree*> eventHistoryTree_;  // backward compatibility
    EventToProcessBlockIndexes eventToProcessBlockIndexes_;
    edm::propagate_const<TBranch*> eventToProcessBlockIndexesBranch_;
    edm::propagate_const<std::unique_ptr<History>> history_;  // backward compatibility
    edm::propagate_const<std::shared_ptr<ProductDependencies>> productDependencies_;
    edm::propagate_const<std::shared_ptr<DuplicateChecker>> duplicateChecker_;
    edm::propagate_const<std::unique_ptr<ProvenanceAdaptor>> provenanceAdaptor_;  // backward comatibility
    edm::propagate_const<std::unique_ptr<MakeProvenanceReader>> provenanceReaderMaker_;
    std::vector<edm::propagate_const<std::shared_ptr<ProductProvenanceRetriever>>> eventProductProvenanceRetrievers_;
    std::vector<ParentageID> parentageIDLookup_;
    edm::propagate_const<std::unique_ptr<DaqProvenanceHelper>> daqProvenanceHelper_;
    edm::propagate_const<TClass*> edProductClass_;
    InputType inputType_;
  };  // class RootFile

}  // namespace edm
#endif
