#ifndef IOPool_Input_RootFile_h
#define IOPool_Input_RootFile_h

/*----------------------------------------------------------------------

RootFile.h // used by ROOT input sources

----------------------------------------------------------------------*/

#include <map>
#include <memory>
#include <string>

#include "boost/shared_ptr.hpp"
#include "boost/utility.hpp"
#include "boost/array.hpp"

#include "RootTree.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/GroupSelector.h"
#include "FWCore/Framework/interface/InputSource.h"
#include "DataFormats/Provenance/interface/BranchChildren.h"
#include "DataFormats/Provenance/interface/BranchMapper.h"
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
#include "DataFormats/Provenance/interface/EventEntryInfo.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/RunLumiEntryInfo.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "DataFormats/Provenance/interface/EventEntryDescription.h"
#include "DataFormats/Provenance/interface/BranchEntryDescription.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
class TFile;

namespace edm {

  //------------------------------------------------------------
  // Class RootFile: supports file reading.

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
	     unsigned int eventsToSkip,
	     std::vector<LuminosityBlockID> const& whichLumisToSkip,
	     int remainingEvents,
	     int remainingLumis,
	     unsigned int treeCacheSize,
             int treeMaxVirtualSize,
	     InputSource::ProcessingMode processingMode,
	     int forcedRunOffset,
	     std::vector<EventID> const& whichEventsToProcess,
             bool dropMetaData,
	     GroupSelectorRules const& groupSelectorRules);
    void reportOpened();
    void close(bool reallyClose);
    std::auto_ptr<EventPrincipal> readCurrentEvent(
	boost::shared_ptr<ProductRegistry const> pReg);
    std::auto_ptr<EventPrincipal> readEvent(
	boost::shared_ptr<ProductRegistry const> pReg);
    boost::shared_ptr<LuminosityBlockPrincipal> readLumi(
	boost::shared_ptr<ProductRegistry const> pReg,
	boost::shared_ptr<RunPrincipal> rp);
    std::string const& file() const {return file_;}
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
    bool setIfFastClonable(int remainingEvents, int remainingLumis) const;
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
    void readEntryDescriptionTree();
    void readEventHistoryTree();

    bool selected(BranchDescription const& desc) const;

    template <typename T>
    boost::shared_ptr<BranchMapper<T> > makeBranchMapper(RootTree & rootTree, BranchType const& type) const;

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
    GroupSelector groupSelector_;
    JobReport::Token reportToken_;
    EventAuxiliary eventAux_;
    LuminosityBlockAuxiliary lumiAux_;
    RunAuxiliary runAux_;
    RootTree eventTree_;
    RootTree lumiTree_;
    RootTree runTree_;
    RootTreePtrArray treePointers_;
    boost::shared_ptr<ProductRegistry const> productRegistry_;
    InputSource::ProcessingMode processingMode_;
    int forcedRunOffset_;
    std::map<std::string, std::string> newBranchToOldBranch_;
    std::vector<std::string> sortedNewBranchNames_;
    std::vector<std::string> oldBranchNames_;
    TTree * eventHistoryTree_;
    History history_;    
    boost::shared_ptr<BranchChildren> branchChildren_;
  }; // class RootFile

  template <typename T>
  boost::shared_ptr<BranchMapper<T> >
  RootFile::makeBranchMapper(RootTree & rootTree, BranchType const& type) const {
    if (fileFormatVersion_.value_ >= 8) {
      return rootTree.makeBranchMapper<T>();
    } 
    // backward compatibility
    boost::shared_ptr<BranchMapper<T> > mapper(new BranchMapper<T>);
    if (fileFormatVersion_.value_ >= 7) {
      rootTree.fillStatus();
      for(ProductRegistry::ProductList::const_iterator it = productRegistry_->productList().begin(),
          itEnd = productRegistry_->productList().end(); it != itEnd; ++it) {
        if (type == it->second.branchType() && !it->second.transient()) {
	  input::BranchMap::const_iterator ix = rootTree.branches().find(it->first);
	  input::BranchInfo const& ib = ix->second;
	  TBranch *br = ib.provenanceBranch_;
	  //TBranch *br = rootTree.branches().find(it->first)->second.provenanceBranch_;
          std::auto_ptr<EntryDescriptionID> pb(new EntryDescriptionID);
          EntryDescriptionID* ppb = pb.get();
          br->SetAddress(&ppb);
          br->GetEntry(rootTree.entryNumber());
          br->SetAddress(0);
	  std::vector<ProductStatus>::size_type index = it->second.oldProductID().id() - 1;
	  T entry(it->second.branchID(),
		  rootTree.productStatuses()[index], it->second.oldProductID(), *pb);
	  mapper->insert(entry);
        }
      }
    } else {
      for(ProductRegistry::ProductList::const_iterator it = productRegistry_->productList().begin(),
          itEnd = productRegistry_->productList().end(); it != itEnd; ++it) {
        if (type == it->second.branchType() && !it->second.transient()) {
	  TBranch *br = rootTree.branches().find(it->first)->second.provenanceBranch_;
          std::auto_ptr<BranchEntryDescription> pb(new BranchEntryDescription);
          BranchEntryDescription* ppb = pb.get();
          br->SetAddress(&ppb);
          br->GetEntry(rootTree.entryNumber());
          std::auto_ptr<EntryDescription> entryDesc = pb->convertToEntryDescription();
	  ProductStatus status = (ppb->creatorStatus() == BranchEntryDescription::Success ? productstatus::present() : productstatus::neverCreated());
	  // Throws parents away for now.
	  T entry(it->second.branchID(), status, entryDesc->moduleDescriptionID_, it->second.oldProductID());
	  mapper->insert(entry);
       }
      }
    }
    return mapper;
  }

}
#endif
