#ifndef IOPool_Input_RootFile_h
#define IOPool_Input_RootFile_h

/*----------------------------------------------------------------------

RootFile.h // used by ROOT input sources

$Id: RootFile.h,v 1.56 2008/04/09 23:00:24 wmtan Exp $

----------------------------------------------------------------------*/

#include <map>
#include <memory>
#include <string>

#include "boost/shared_ptr.hpp"
#include "boost/array.hpp"

#include "RootTree.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventProcessHistoryID.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "DataFormats/Provenance/interface/ProductStatus.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/FileIndex.h"
#include "DataFormats/Provenance/interface/History.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
class TFile;

namespace edm {

  //------------------------------------------------------------
  // Class RootFile: supports file reading.

  class RootFile : boost::noncopyable {
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
	     unsigned int eventsToSkip,
	     std::vector<LuminosityBlockID> const& whichLumisToSkip,
	     int remainingEvents,
	     unsigned int treeCacheSize,
             int treeMaxVirtualSize,
	     int forcedRunOffset,
	     std::vector<EventID> const& whichEventsToProcess,
             bool dropMetaData);
    void reportOpened();
    void close(bool reallyClose);
    std::auto_ptr<EventPrincipal> readCurrentEvent(
	boost::shared_ptr<ProductRegistry const> pReg,
	boost::shared_ptr<LuminosityBlockPrincipal> lbp);
    std::auto_ptr<EventPrincipal> readEvent(
	boost::shared_ptr<ProductRegistry const> pReg,
	boost::shared_ptr<LuminosityBlockPrincipal> lbp);
    boost::shared_ptr<LuminosityBlockPrincipal> readLumi(
	boost::shared_ptr<ProductRegistry const> pReg,
	boost::shared_ptr<RunPrincipal> rp);
    boost::shared_ptr<RunPrincipal> readRun(boost::shared_ptr<ProductRegistry const> pReg);
    boost::shared_ptr<ProductRegistry const> productRegistry() const {return productRegistry_;}
    EventAuxiliary const& eventAux() const {return eventAux_;}
    LuminosityBlockAuxiliary const& luminosityBlockAux() {return lumiAux_;}
    RunAuxiliary const& runAux() const {return runAux_;}
    EventID const& eventID() const {return eventAux().id();}
    RootTreePtrArray & treePointers() {return treePointers_;}
    RootTree const& eventTree() const {return eventTree_;}
    RootTree const& lumiTree() const {return lumiTree_;}
    RootTree const & runTree() const {return runTree_;}
    FileFormatVersion fileFormatVersion() const {return fileFormatVersion_;}
    bool fastClonable() const {return fastClonable_;}
    boost::shared_ptr<FileBlock> createFileBlock() const;
    bool setEntryAtEvent(RunNumber_t run, LuminosityBlockNumber_t lumi, EventNumber_t event, bool exact);
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

    unsigned int eventsToSkip() const {return eventsToSkip_;}
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
    bool setIfFastClonable(int remainingEvents) const;
    void validateFile();
    void fillFileIndex();
    void fillEventAuxiliary();
    void fillHistory();
    void fillLumiAuxiliary();
    void fillRunAuxiliary();
    void overrideRunNumber(RunID & id);
    void overrideRunNumber(LuminosityBlockID & id);
    void overrideRunNumber(EventID & id, bool isRealData);
    std::string const& newBranchToOldBranch(std::string const& newBranch) const;
    void readEventDescriptionTree();
    void readEventHistoryTree();

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
    std::vector<EventProcessHistoryID> eventProcessHistoryIDs_;
    std::vector<EventProcessHistoryID>::const_iterator eventProcessHistoryIter_;
    RunNumber_t startAtRun_;
    LuminosityBlockNumber_t startAtLumi_;
    EventNumber_t startAtEvent_;
    unsigned int eventsToSkip_;
    std::vector<LuminosityBlockID> whichLumisToSkip_;
    std::vector<EventID> whichEventsToProcess_;
    std::vector<EventID>::const_iterator eventListIter_;
    bool fastClonable_;
    bool dropMetaData_;
    JobReport::Token reportToken_;
    EventAuxiliary eventAux_;
    LuminosityBlockAuxiliary lumiAux_;
    RunAuxiliary runAux_;
    RootTree eventTree_;
    RootTree lumiTree_;
    RootTree runTree_;
    RootTreePtrArray treePointers_;
    boost::shared_ptr<ProductRegistry const> productRegistry_;
    int forcedRunOffset_;
    std::map<std::string, std::string> newBranchToOldBranch_;
    std::vector<std::string> sortedNewBranchNames_;
    std::vector<std::string> oldBranchNames_;
    TTree * eventHistoryTree_;
    History history_;    
  }; // class RootFile

}
#endif
