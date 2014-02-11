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

#include <array>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace edm {

  //------------------------------------------------------------
  // Class RootFile: supports file reading.

  class BranchIDListHelper;
  class ProductProvenanceRetriever;
  class DaqProvenanceHelper;
  class DuplicateChecker;
  class EventSkipperByID;
  class ProcessHistoryRegistry;
  class ProductSelectorRules;
  class InputFile;
  class ProvenanceReaderBase;
  class ProvenanceAdaptor;
  typedef std::map<EntryDescriptionID, EventEntryDescription> EntryDescriptionMap;

  class MakeProvenanceReader {
  public:
    virtual std::unique_ptr<ProvenanceReaderBase> makeReader(RootTree& eventTree, DaqProvenanceHelper const* daqProvenanceHelper) const = 0;
  };

  class RootFile {
  public:
    typedef std::array<RootTree*, NumBranchTypes> RootTreePtrArray;
    RootFile(std::string const& fileName,
             ProcessConfiguration const& processConfiguration,
             std::string const& logicalFileName,
             boost::shared_ptr<InputFile> filePtr,
             boost::shared_ptr<EventSkipperByID> eventSkipperByID,
             bool skipAnyEvents,
             int remainingEvents,
             int remainingLumis,
             unsigned int nStreams,
             unsigned int treeCacheSize,
             int treeMaxVirtualSize,
             InputSource::ProcessingMode processingMode,
             RunNumber_t const& forcedRunNumber,
             bool noEventSort,
             ProductSelectorRules const& productSelectorRules,
             InputType inputType,
             boost::shared_ptr<BranchIDListHelper> branchIDListHelper,
             boost::shared_ptr<DuplicateChecker> duplicateChecker,
             bool dropDescendantsOfDroppedProducts,
             ProcessHistoryRegistry& processHistoryRegistry,
             std::vector<boost::shared_ptr<IndexIntoFile> > const& indexesIntoFiles,
             std::vector<boost::shared_ptr<IndexIntoFile> >::size_type currentIndexIntoFile,
             std::vector<ProcessHistoryID>& orderedProcessHistoryIDs,
             bool labelRawDataLikeMC,
             bool usingGoToEvent,
             bool enablePrefetching);
    ~RootFile();

    RootFile(RootFile const&) = delete; // Disallow copying and moving
    RootFile& operator=(RootFile const&) = delete; // Disallow copying and moving

    void reportOpened(std::string const& inputType);
    void close();
    bool readCurrentEvent(EventPrincipal& cache);
    void readEvent(EventPrincipal& cache);

    boost::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();
    boost::shared_ptr<RunAuxiliary> readRunAuxiliary_();
    void readRun_(RunPrincipal& runPrincipal);
    void readLuminosityBlock_(LuminosityBlockPrincipal& lumiPrincipal);
    std::string const& file() const {return file_;}
    boost::shared_ptr<ProductRegistry const> productRegistry() const {return productRegistry_;}
    boost::shared_ptr<BranchIDListHelper const> branchIDListHelper() const {return branchIDListHelper_;}
    BranchIDLists const& branchIDLists() {return *branchIDLists_;}
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
    bool modifiedIDs() const {return daqProvenanceHelper_.get() != 0;}
    std::unique_ptr<FileBlock> createFileBlock() const;
    bool setEntryAtItem(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event) {
      return event ? setEntryAtEvent(run, lumi, event) : (lumi ? setEntryAtLumi(run, lumi) : setEntryAtRun(run));
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

    bool skipEvents(int& offset);
    bool goToEvent(EventID const& eventID);
    bool nextEventEntry() {return eventTree_.next();}
    IndexIntoFile::EntryType getNextItemType(RunNumber_t& run, LuminosityBlockNumber_t& lumi, EventNumber_t& event);
    boost::shared_ptr<IndexIntoFile> indexIntoFileSharedPtr() const {
      return indexIntoFileSharedPtr_;
    }
    bool wasLastEventJustRead() const;
    bool wasFirstEventJustRead() const;
    IndexIntoFile::IndexIntoFileItr indexIntoFileIter() const;
    void setPosition(IndexIntoFile::IndexIntoFileItr const& position);

  private:
    RootTreePtrArray& treePointers() {return treePointers_;}
    bool skipThisEntry();
    void setIfFastClonable(int remainingEvents, int remainingLumis);
    void validateFile(InputType inputType, bool usingGoToEvent);
    void fillIndexIntoFile();
    bool fillEventAuxiliary(IndexIntoFile::EntryNumber_t entry);
    void fillThisEventAuxiliary();
    void fillEventHistory();
    boost::shared_ptr<LuminosityBlockAuxiliary> fillLumiAuxiliary();
    boost::shared_ptr<RunAuxiliary> fillRunAuxiliary();
    void overrideRunNumber(RunID& id);
    void overrideRunNumber(LuminosityBlockID& id);
    void overrideRunNumber(EventID& id, bool isRealData);
    std::string const& newBranchToOldBranch(std::string const& newBranch) const;
    void dropOnInput(ProductRegistry& reg, ProductSelectorRules const& rules, bool dropDescendants, InputType inputType);
    void readParentageTree(InputType inputType);
    void readEntryDescriptionTree(EntryDescriptionMap& entryDescriptionMap, InputType inputType); // backward compatibility
    void readEventHistoryTree();
    bool isDuplicateEvent();

    void initializeDuplicateChecker(std::vector<boost::shared_ptr<IndexIntoFile> > const& indexesIntoFiles,
                                    std::vector<boost::shared_ptr<IndexIntoFile> >::size_type currentIndexIntoFile);

    std::unique_ptr<MakeProvenanceReader> makeProvenanceReaderMaker(InputType inputType);
    boost::shared_ptr<ProductProvenanceRetriever> makeProductProvenanceRetriever(unsigned int iStreamIndex);

    std::string const file_;
    std::string const logicalFile_;
    ProcessConfiguration const& processConfiguration_;
    ProcessHistoryRegistry* processHistoryRegistry_;  // We don't own this
    boost::shared_ptr<InputFile> filePtr_;
    boost::shared_ptr<EventSkipperByID> eventSkipperByID_;
    FileFormatVersion fileFormatVersion_;
    FileID fid_;
    boost::shared_ptr<IndexIntoFile> indexIntoFileSharedPtr_;
    IndexIntoFile& indexIntoFile_;
    std::vector<ProcessHistoryID>& orderedProcessHistoryIDs_;
    IndexIntoFile::IndexIntoFileItr indexIntoFileBegin_;
    IndexIntoFile::IndexIntoFileItr indexIntoFileEnd_;
    IndexIntoFile::IndexIntoFileItr indexIntoFileIter_;
    std::vector<EventProcessHistoryID> eventProcessHistoryIDs_;  // backward compatibility
    std::vector<EventProcessHistoryID>::const_iterator eventProcessHistoryIter_; // backward compatibility
    boost::shared_ptr<RunAuxiliary> savedRunAuxiliary_; // backward compatibility
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
    boost::shared_ptr<ProductRegistry const> productRegistry_;
    boost::shared_ptr<BranchIDLists const> branchIDLists_;
    boost::shared_ptr<BranchIDListHelper> branchIDListHelper_;
    InputSource::ProcessingMode processingMode_;
    int forcedRunOffset_;
    std::map<std::string, std::string> newBranchToOldBranch_;
    TTree* eventHistoryTree_;			// backward compatibility
    EventSelectionIDVector eventSelectionIDs_;
    BranchListIndexes branchListIndexes_;
    std::unique_ptr<History> history_; // backward compatibility
    boost::shared_ptr<BranchChildren> branchChildren_;
    boost::shared_ptr<DuplicateChecker> duplicateChecker_;
    std::unique_ptr<ProvenanceAdaptor> provenanceAdaptor_; // backward comatibility
    std::unique_ptr<MakeProvenanceReader> provenanceReaderMaker_;
    mutable std::vector<boost::shared_ptr<ProductProvenanceRetriever>> eventProductProvenanceRetrievers_;
    std::vector<ParentageID> parentageIDLookup_;
    std::unique_ptr<DaqProvenanceHelper> daqProvenanceHelper_;
  }; // class RootFile

}
#endif
