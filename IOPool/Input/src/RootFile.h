#ifndef IOPool_Input_RootFile_h
#define IOPool_Input_RootFile_h

/*----------------------------------------------------------------------

RootFile.h // used by ROOT input sources

----------------------------------------------------------------------*/

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "boost/shared_ptr.hpp"
#include "boost/utility.hpp"
#include "boost/array.hpp"

#include "RootTree.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "DataFormats/Provenance/interface/BranchChildren.h"
#include "DataFormats/Provenance/interface/BranchIDListRegistry.h"
#include "DataFormats/Provenance/interface/BranchMapper.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventRange.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockRange.h"
#include "DataFormats/Provenance/interface/ProductStatus.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/FileIndex.h"
#include "DataFormats/Provenance/interface/History.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ProductProvenance.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "DataFormats/Provenance/interface/Parentage.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "DataFormats/Provenance/interface/EventProcessHistoryID.h" // backward compatibility
class TFile;

namespace edm {

  //------------------------------------------------------------
  // Class RootFile: supports file reading.

  class DuplicateChecker;
  class ProvenanceAdaptor;
  class GroupSelectorRules;

  class RootFile : private boost::noncopyable {
  public:
    typedef boost::array<RootTree *, NumBranchTypes> RootTreePtrArray;
    RootFile(std::string const& fileName,
	     std::string const& catalogName,
	     ProcessConfiguration const& processConfiguration,
	     std::string const& logicalFileName,
	     boost::shared_ptr<TFile> filePtr,
	     RunNumber_t const& startAtRun,
	     LuminosityBlockNumber_t const& startAtLumi,
	     EventNumber_t const& startAtEvent,
	     bool skipAnyEvents,
	     std::vector<LuminosityBlockRange> const& whichLumisToSkip,
	     std::vector<EventRange> const& whichEventsToSkip,
	     int remainingEvents,
	     int remainingLumis,
	     unsigned int treeCacheSize,
             int treeMaxVirtualSize,
	     InputSource::ProcessingMode processingMode,
	     int forcedRunOffset,
	     std::vector<LuminosityBlockRange> const& whichLumisToProcess,
	     std::vector<EventRange> const& whichEventsToProcess,
             bool noEventSort,
	     GroupSelectorRules const& groupSelectorRules,
             bool dropMergeable,
             boost::shared_ptr<DuplicateChecker> duplicateChecker,
             bool dropDescendantsOfDroppedProducts,
             std::vector<boost::shared_ptr<FileIndex> > const& fileIndexes,
             std::vector<boost::shared_ptr<FileIndex> >::size_type currentFileIndex);
    ~RootFile();
    void reportOpened(std::string const& inputType);
    void close(bool reallyClose);
    EventPrincipal* readCurrentEvent(EventPrincipal& cache,
		 boost::shared_ptr<LuminosityBlockPrincipal> lb = boost::shared_ptr<LuminosityBlockPrincipal>());
    EventPrincipal* readEvent(EventPrincipal& cache,
		 boost::shared_ptr<LuminosityBlockPrincipal> lb = boost::shared_ptr<LuminosityBlockPrincipal>());

    boost::shared_ptr<LuminosityBlockAuxiliary> readLuminosityBlockAuxiliary_();
    boost::shared_ptr<LuminosityBlockPrincipal> readLuminosityBlock_();
    boost::shared_ptr<RunAuxiliary> readRunAuxiliary_();
    boost::shared_ptr<RunPrincipal> readRun_(boost::shared_ptr<RunPrincipal> rpCache);
    boost::shared_ptr<LuminosityBlockPrincipal> readLumi(boost::shared_ptr<LuminosityBlockPrincipal> lbCache);
    std::string const& file() const {return file_;}
    boost::shared_ptr<RunPrincipal> readRun_(boost::shared_ptr<RunPrincipal> rpCache, boost::shared_ptr<ProductRegistry const> pReg);
    boost::shared_ptr<ProductRegistry const> productRegistry() const {return productRegistry_;}
    BranchIDListRegistry::collection_type const& branchIDLists() {return *branchIDLists_;}
    EventAuxiliary const& eventAux() const {return eventAux_;}
    EventNumber_t const& eventNumber() const {return fileIndexIter_->event_;}
    FileIndex::EntryNumber_t const& entryNumber() const {return fileIndexIter_->entry_;}
    LuminosityBlockNumber_t const& luminosityBlockNumber() const {return fileIndexIter_->lumi_;}
    RunNumber_t const& runNumber() const {return fileIndexIter_->run_;}
    EventID const& eventID() const {return eventAux().id();}
    RootTreePtrArray & treePointers() {return treePointers_;}
    RootTree const& eventTree() const {return eventTree_;}
    RootTree const& lumiTree() const {return lumiTree_;}
    RootTree const & runTree() const {return runTree_;}
    FileFormatVersion fileFormatVersion() const {return fileFormatVersion_;}
    int whyNotFastClonable() const {return whyNotFastClonable_;}
    boost::shared_ptr<FileBlock> createFileBlock() const;
    bool setEntryAtItem(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, bool exact) {
      return event ? setEntryAtEvent(run, lumi, event, exact) : (lumi ? setEntryAtLumi(run, lumi) : setEntryAtRun(run));
    }
    bool setEntryAtEvent(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, bool exact);
    bool setEntryAtEventEntry(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, FileIndex::EntryNumber_t entry, bool exact);
    bool setEntryAtLumi(RunNumber_t run, LuminosityBlockNumber_t lumi);
    bool setEntryAtRun(RunNumber_t run);
    void setAtEventEntry(FileIndex::EntryNumber_t entry);
    void rewind() {
      fileIndexIter_ = fileIndexBegin_;
      eventTree_.rewind();
      lumiTree_.rewind();
      runTree_.rewind();
    }
    void setToLastEntry() {
      fileIndexIter_ = fileIndexEnd_;
    }

    int skipEvents(int offset);
    int setForcedRunOffset(RunNumber_t const& forcedRunNumber);
    bool nextEventEntry() {return eventTree_.next();}
    FileIndex::EntryType getEntryType() const;
    FileIndex::EntryType getEntryTypeSkippingDups();
    FileIndex::EntryType getNextEntryTypeWanted();
    boost::shared_ptr<FileIndex> fileIndexSharedPtr() const {
      return fileIndexSharedPtr_;
    }

  private:
    void setIfFastClonable(int remainingEvents, int remainingLumis);
    void validateFile();
    void fillFileIndex();
    void fillEventAuxiliary();
    void fillHistory();
    boost::shared_ptr<LuminosityBlockAuxiliary> fillLumiAuxiliary();
    boost::shared_ptr<RunAuxiliary> fillRunAuxiliary();
    void overrideRunNumber(RunID & id);
    void overrideRunNumber(LuminosityBlockID & id);
    void overrideRunNumber(EventID & id, bool isRealData);
    std::string const& newBranchToOldBranch(std::string const& newBranch) const;
    void dropOnInput(ProductRegistry& reg, GroupSelectorRules const& rules, bool dropDescendants, bool dropMergeable);
    void readParentageTree();
    void readEntryDescriptionTree();
    void readEventHistoryTree();
    bool isDuplicateEvent() const;

    void initializeDuplicateChecker(std::vector<boost::shared_ptr<FileIndex> > const& fileIndexes,
                                    std::vector<boost::shared_ptr<FileIndex> >::size_type currentFileIndex);

    boost::shared_ptr<BranchMapper> makeBranchMapper(RootTree & rootTree, BranchType const& type) const;

    std::string const file_;
    std::string const logicalFile_;
    std::string const catalog_;
    ProcessConfiguration const& processConfiguration_;
    boost::shared_ptr<TFile> filePtr_;
    FileFormatVersion fileFormatVersion_;
    FileID fid_;
    boost::shared_ptr<FileIndex> fileIndexSharedPtr_;
    FileIndex & fileIndex_;
    FileIndex::const_iterator fileIndexBegin_;
    FileIndex::const_iterator fileIndexEnd_;
    FileIndex::const_iterator fileIndexIter_;
    std::vector<EventProcessHistoryID> eventProcessHistoryIDs_;  // backward compatibility
    std::vector<EventProcessHistoryID>::const_iterator eventProcessHistoryIter_; // backward compatibility
    bool skipAnyEvents_;
    bool noEventSort_;
    int whyNotFastClonable_;
    JobReport::Token reportToken_;
    EventAuxiliary eventAux_;
    RootTree eventTree_;
    RootTree lumiTree_;
    RootTree runTree_;
    RootTreePtrArray treePointers_;
    boost::shared_ptr<ProductRegistry const> productRegistry_;
    boost::shared_ptr<BranchIDListRegistry::collection_type const> branchIDLists_;
    InputSource::ProcessingMode processingMode_;
    int forcedRunOffset_;
    std::map<std::string, std::string> newBranchToOldBranch_;
    TTree * eventHistoryTree_;
    boost::shared_ptr<History> history_;    
    boost::shared_ptr<BranchChildren> branchChildren_;
    boost::shared_ptr<DuplicateChecker> duplicateChecker_;
    boost::shared_ptr<ProvenanceAdaptor> provenanceAdaptor_;
  }; // class RootFile

}
#endif
