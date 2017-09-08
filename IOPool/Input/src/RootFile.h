#ifndef IOPool_Input_RootFile_h
#define IOPool_Input_RootFile_h

/*----------------------------------------------------------------------

RootFile.h // used by ROOT input sources

----------------------------------------------------------------------*/

#include "RootTree.h"
#include "DataFormats/Provenance/interface/BranchChildren.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/EntryDescriptionID.h" // backward compatibility
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventEntryDescription.h" // backward compatibility
#include "DataFormats/Provenance/interface/EventProcessHistoryID.h" // backward compatibility
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/History.h"
#include "DataFormats/Provenance/interface/IndexIntoFile.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "FWCore/Utilities/interface/InputType.h"
#include "FWCore/Utilities/interface/get_underlying_safe.h"

#include <array>
#include <map>
#include <memory>
#include <string>
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
  class RunHelperBase;
  class ThinnedAssociationsHelper;

  typedef std::map<EntryDescriptionID, EventEntryDescription> EntryDescriptionMap;

  class MakeProvenanceReader {
  public:
    virtual std::unique_ptr<ProvenanceReaderBase> makeReader(RootTree& eventTree, DaqProvenanceHelper const* daqProvenanceHelper) const = 0;
    virtual ~MakeProvenanceReader() = default;
  };

  class RootFile {
  public:
    typedef std::array<RootTree*, NumBranchTypes> RootTreePtrArray;
    RootFile(std::string const& fileName,
             ProcessConfiguration const& processConfiguration,
             std::string const& logicalFileName,
             std::shared_ptr<InputFile> filePtr,
             std::shared_ptr<EventSkipperByID> eventSkipperByID,
             bool skipAnyEvents,
             int remainingEvents,
             int remainingLumis,
             unsigned int nStreams,
             unsigned int treeCacheSize,
             int treeMaxVirtualSize,
             InputSource::ProcessingMode processingMode,
             RunHelperBase* runHelper,
             bool noEventSort,
             ProductSelectorRules const& productSelectorRules,
             InputType inputType,
             std::shared_ptr<BranchIDListHelper> branchIDListHelper,
             std::shared_ptr<ThinnedAssociationsHelper> thinnedAssociationsHelper,
             std::vector<BranchID> const* associationsFromSecondary,
             std::shared_ptr<DuplicateChecker> duplicateChecker,
             bool dropDescendantsOfDroppedProducts,
             ProcessHistoryRegistry& processHistoryRegistry,
             std::vector<std::shared_ptr<IndexIntoFile> > const& indexesIntoFiles,
             std::vector<std::shared_ptr<IndexIntoFile> >::size_type currentIndexIntoFile,
             std::vector<ProcessHistoryID>& orderedProcessHistoryIDs,
             bool bypassVersionCheck,
             bool labelRawDataLikeMC,
             bool usingGoToEvent,
             bool enablePrefetching);

    RootFile(std::string const& fileName,
             ProcessConfiguration const& processConfiguration,
             std::string const& logicalFileName,
             std::shared_ptr<InputFile> filePtr,
             unsigned int nStreams,
             int treeMaxVirtualSize,
             InputSource::ProcessingMode processingMode,
             RunHelperBase* runHelper,
             ProductSelectorRules const& productSelectorRules,
             InputType inputType,
             std::shared_ptr<BranchIDListHelper> branchIDListHelper,
             std::shared_ptr<ThinnedAssociationsHelper> thinnedAssociationsHelper,
             std::vector<BranchID> const* associationsFromSecondary,
             bool dropDescendantsOfDroppedProducts,
             ProcessHistoryRegistry& processHistoryRegistry,
             std::vector<std::shared_ptr<IndexIntoFile> > const& indexesIntoFiles,
             std::vector<std::shared_ptr<IndexIntoFile> >::size_type currentIndexIntoFile,
             std::vector<ProcessHistoryID>& orderedProcessHistoryIDs,
             bool bypassVersionCheck,
             bool labelRawDataLikeMC,
             bool enablePrefetching) : RootFile(
               fileName, processConfiguration, logicalFileName, filePtr,
               nullptr, false, -1, -1, nStreams, 0U, treeMaxVirtualSize,
               processingMode, runHelper,
               false, productSelectorRules, inputType, branchIDListHelper,
               thinnedAssociationsHelper, associationsFromSecondary,
               nullptr, dropDescendantsOfDroppedProducts, processHistoryRegistry,
               indexesIntoFiles, currentIndexIntoFile, orderedProcessHistoryIDs,
               bypassVersionCheck, labelRawDataLikeMC,
               false, enablePrefetching) {}

    RootFile(std::string const& fileName,
             ProcessConfiguration const& processConfiguration,
             std::string const& logicalFileName,
             std::shared_ptr<InputFile> filePtr,
             unsigned int nStreams,
             unsigned int treeCacheSize,
             int treeMaxVirtualSize,
             RunHelperBase* runHelper,
             ProductSelectorRules const& productSelectorRules,
             InputType inputType,
             ProcessHistoryRegistry& processHistoryRegistry,
             std::vector<std::shared_ptr<IndexIntoFile> > const& indexesIntoFiles,
             std::vector<std::shared_ptr<IndexIntoFile> >::size_type currentIndexIntoFile,
             std::vector<ProcessHistoryID>& orderedProcessHistoryIDs,
             bool bypassVersionCheck,
             bool enablePrefetching) : RootFile(
               fileName, processConfiguration, logicalFileName, filePtr,
               nullptr, false, -1, -1, nStreams, treeCacheSize, treeMaxVirtualSize,
               InputSource::RunsLumisAndEvents, runHelper,
               false, productSelectorRules, inputType, nullptr, nullptr,
               nullptr, nullptr, false, processHistoryRegistry,
               indexesIntoFiles, currentIndexIntoFile, orderedProcessHistoryIDs,
               bypassVersionCheck, false,
               false, enablePrefetching) {}

    ~RootFile();

    RootFile(RootFile const&) = delete; // Disallow copying and moving
    RootFile& operator=(RootFile const&) = delete; // Disallow copying and moving

    void reportOpened(std::string const& inputType);
    void close();
    bool readCurrentEvent(EventPrincipal& cache);
    void readEvent(EventPrincipal& cache);

    std::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();
    std::shared_ptr<RunAuxiliary> readRunAuxiliary_();
    std::shared_ptr<RunAuxiliary> readFakeRunAuxiliary_();
    void readRun_(RunPrincipal& runPrincipal);
    void readFakeRun_(RunPrincipal& runPrincipal);
    void readLuminosityBlock_(LuminosityBlockPrincipal& lumiPrincipal);
    std::string const& file() const {return file_;}
    std::shared_ptr<ProductRegistry const> productRegistry() const {return productRegistry_;}
    EventAuxiliary const& eventAux() const {return eventAux_;}
    // IndexIntoFile::EntryNumber_t const& entryNumber() const {return indexIntoFileIter().entry();}
    // LuminosityBlockNumber_t const& luminosityBlockNumber() const {return indexIntoFileIter().lumi();}
    // RunNumber_t const& runNumber() const {return indexIntoFileIter().run();}
    EventID const& eventID() const {return eventAux().id();}
    RootTree const& eventTree() const {return eventTree_;}
    RootTree const& lumiTree() const {return lumiTree_;}
    RootTree const& runTree() const {return runTree_;}
    FileFormatVersion fileFormatVersion() const {return fileFormatVersion_;}
    int whyNotFastClonable() const {return whyNotFastClonable_;}
    std::array<bool, NumBranchTypes> const& hasNewlyDroppedBranch() const {return hasNewlyDroppedBranch_;}
    bool branchListIndexesUnchanged() const {return branchListIndexesUnchanged_;}
    bool modifiedIDs() const {return daqProvenanceHelper_.get() != nullptr;}
    std::unique_ptr<FileBlock> createFileBlock() const;
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
    }
    void setToLastEntry() {
      indexIntoFileIter_ = indexIntoFileEnd_;
    }

    bool skipEntries(unsigned int& offset) {return eventTree_.skipEntries(offset);}
    bool skipEvents(int& offset);
    bool goToEvent(EventID const& eventID);
    bool nextEventEntry() {return eventTree_.next();}
    IndexIntoFile::EntryType getNextItemType(RunNumber_t& run, LuminosityBlockNumber_t& lumi, EventNumber_t& event);
    std::shared_ptr<BranchIDListHelper const> branchIDListHelper() const {return get_underlying_safe(branchIDListHelper_);}
    std::shared_ptr<BranchIDListHelper>& branchIDListHelper() {return get_underlying_safe(branchIDListHelper_);}
    std::shared_ptr<IndexIntoFile const> indexIntoFileSharedPtr() const {return get_underlying_safe(indexIntoFileSharedPtr_);}
    std::shared_ptr<IndexIntoFile>& indexIntoFileSharedPtr() {return get_underlying_safe(indexIntoFileSharedPtr_);}
    bool wasLastEventJustRead() const;
    bool wasFirstEventJustRead() const;
    IndexIntoFile::IndexIntoFileItr indexIntoFileIter() const;
    void setPosition(IndexIntoFile::IndexIntoFileItr const& position);
    void initAssociationsFromSecondary(std::vector<BranchID> const&);

    void setSignals(signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* preEventReadSource,
                    signalslot::Signal<void(StreamContext const&, ModuleCallingContext const&)> const* postEventReadSource);
  private:
    RootTreePtrArray& treePointers() {return treePointers_;}
    bool skipThisEntry();
    void setIfFastClonable(int remainingEvents, int remainingLumis);
    void validateFile(InputType inputType, bool usingGoToEvent);
    void fillIndexIntoFile();
    bool fillEventAuxiliary(IndexIntoFile::EntryNumber_t entry);
    void fillThisEventAuxiliary();
    void fillEventHistory();
    std::shared_ptr<LuminosityBlockAuxiliary> fillLumiAuxiliary();
    std::shared_ptr<RunAuxiliary> fillRunAuxiliary();
    std::string const& newBranchToOldBranch(std::string const& newBranch) const;
    void markBranchToBeDropped(bool dropDescendants, BranchDescription const& branch, std::set<BranchID>& branchesToDrop, std::map<BranchID, BranchID> const& droppedToKeptAlias) const;
    void dropOnInput(ProductRegistry& reg, ProductSelectorRules const& rules, bool dropDescendants, InputType inputType);
    void readParentageTree(InputType inputType);
    void readEntryDescriptionTree(EntryDescriptionMap& entryDescriptionMap, InputType inputType); // backward compatibility
    void readEventHistoryTree();
    bool isDuplicateEvent();

    void initializeDuplicateChecker(std::vector<std::shared_ptr<IndexIntoFile> > const& indexesIntoFiles,
                                    std::vector<std::shared_ptr<IndexIntoFile> >::size_type currentIndexIntoFile);

    std::unique_ptr<MakeProvenanceReader> makeProvenanceReaderMaker(InputType inputType);
    std::shared_ptr<ProductProvenanceRetriever> makeProductProvenanceRetriever(unsigned int iStreamIndex);

    std::shared_ptr<RunAuxiliary const> savedRunAuxiliary() const {return get_underlying_safe(savedRunAuxiliary_);}
    std::shared_ptr<RunAuxiliary>& savedRunAuxiliary() {return get_underlying_safe(savedRunAuxiliary_);}

    std::shared_ptr<BranchChildren const> branchChildren() const {return get_underlying_safe(branchChildren_);}
    std::shared_ptr<BranchChildren>& branchChildren() {return get_underlying_safe(branchChildren_);}

    std::shared_ptr<ProductProvenanceRetriever const> eventProductProvenanceRetriever(size_t index) const {return get_underlying_safe(eventProductProvenanceRetrievers_[index]);}
    std::shared_ptr<ProductProvenanceRetriever>& eventProductProvenanceRetriever(size_t index) {return get_underlying_safe(eventProductProvenanceRetrievers_[index]);}

    std::string const file_;
    std::string const logicalFile_;
    ProcessConfiguration const& processConfiguration_;
    edm::propagate_const<ProcessHistoryRegistry*> processHistoryRegistry_;  // We don't own this
    edm::propagate_const<std::shared_ptr<InputFile>> filePtr_;
    edm::propagate_const<std::shared_ptr<EventSkipperByID>> eventSkipperByID_;
    FileFormatVersion fileFormatVersion_;
    FileID fid_;
    edm::propagate_const<std::shared_ptr<IndexIntoFile>> indexIntoFileSharedPtr_;
    IndexIntoFile& indexIntoFile_;
    std::vector<ProcessHistoryID>& orderedProcessHistoryIDs_;
    IndexIntoFile::IndexIntoFileItr indexIntoFileBegin_;
    IndexIntoFile::IndexIntoFileItr indexIntoFileEnd_;
    IndexIntoFile::IndexIntoFileItr indexIntoFileIter_;
    std::vector<EventProcessHistoryID> eventProcessHistoryIDs_;  // backward compatibility
    std::vector<EventProcessHistoryID>::const_iterator eventProcessHistoryIter_; // backward compatibility
    edm::propagate_const<std::shared_ptr<RunAuxiliary>> savedRunAuxiliary_;
    bool skipAnyEvents_;
    bool noEventSort_;
    int whyNotFastClonable_;
    std::array<bool, NumBranchTypes> hasNewlyDroppedBranch_;
    bool branchListIndexesUnchanged_;
    EventAuxiliary eventAux_;
    RootTree eventTree_;
    RootTree lumiTree_;
    RootTree runTree_;
    RootTreePtrArray treePointers_;
    IndexIntoFile::EntryNumber_t lastEventEntryNumberRead_;
    std::shared_ptr<ProductRegistry const> productRegistry_;
    std::shared_ptr<BranchIDLists const> branchIDLists_;
    edm::propagate_const<std::shared_ptr<BranchIDListHelper>> branchIDListHelper_;
    edm::propagate_const<std::unique_ptr<ThinnedAssociationsHelper>> fileThinnedAssociationsHelper_;
    edm::propagate_const<std::shared_ptr<ThinnedAssociationsHelper>> thinnedAssociationsHelper_;
    InputSource::ProcessingMode processingMode_;
    edm::propagate_const<RunHelperBase*> runHelper_;
    std::map<std::string, std::string> newBranchToOldBranch_;
    edm::propagate_const<TTree*> eventHistoryTree_; // backward compatibility
    EventSelectionIDVector eventSelectionIDs_;
    BranchListIndexes branchListIndexes_;
    edm::propagate_const<std::unique_ptr<History>> history_; // backward compatibility
    edm::propagate_const<std::shared_ptr<BranchChildren>> branchChildren_;
    edm::propagate_const<std::shared_ptr<DuplicateChecker>> duplicateChecker_;
    edm::propagate_const<std::unique_ptr<ProvenanceAdaptor>> provenanceAdaptor_; // backward comatibility
    edm::propagate_const<std::unique_ptr<MakeProvenanceReader>> provenanceReaderMaker_;
    std::vector<edm::propagate_const<std::shared_ptr<ProductProvenanceRetriever>>> eventProductProvenanceRetrievers_;
    std::vector<ParentageID> parentageIDLookup_;
    edm::propagate_const<std::unique_ptr<DaqProvenanceHelper>> daqProvenanceHelper_;
    edm::propagate_const<TClass*> edProductClass_;
  }; // class RootFile

}
#endif
