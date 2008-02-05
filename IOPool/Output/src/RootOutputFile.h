#ifndef IOPool_Output_RootOutputFile_h
#define IOPool_Output_RootOutputFile_h

//////////////////////////////////////////////////////////////////////
//
// $Id: RootOutputFile.h,v 1.23 2008/02/01 20:23:42 wmtan Exp $
//
// Class PoolOutputModule. Output module to POOL file
//
// Oringinal Author: Luca Lista
// Current Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include <map>
#include <memory>
#include <string>
#include <iosfwd>
#include <vector>
#include "boost/array.hpp"
#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/EntryDescriptionID.h"
#include "DataFormats/Provenance/interface/EventAuxiliary.h"
#include "DataFormats/Provenance/interface/EventProcessHistoryID.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/FileIndex.h"
#include "DataFormats/Provenance/interface/LuminosityBlockAuxiliary.h"
#include "DataFormats/Provenance/interface/ProductStatus.h"
#include "DataFormats/Provenance/interface/RunAuxiliary.h"
#include "DataFormats/Provenance/interface/Selections.h"
#include "IOPool/Output/src/RootOutputTree.h"

class TTree;
class TFile;

namespace edm {
  class ParameterSet;
  class PoolOutputModule;

  class RootOutputFile {
  public:
    typedef boost::array<RootOutputTree *, NumBranchTypes> RootOutputTreePtrArray;
    explicit RootOutputFile(PoolOutputModule * om, std::string const& fileName, std::string const& logicalFileName);
    ~RootOutputFile() {}
    void writeOne(EventPrincipal const& e);
    //void endFile();
    void writeLuminosityBlock(LuminosityBlockPrincipal const& lb);
    bool writeRun(RunPrincipal const& r);
    void writeEntryDescriptions();
    void writeFileFormatVersion();
    void writeFileIdentifier();
    void writeFileIndex();
    void writeEventHistory();
    void writeProcessConfigurationRegistry();
    void writeProcessHistoryRegistry();
    void writeModuleDescriptionRegistry();
    void writeParameterSetRegistry();
    void writeProductDescriptionRegistry();

    void finishEndFile();
    void beginInputFile(FileBlock const& fb, bool fastClone, bool fastMetaClone);
    void respondToCloseInputFile(FileBlock const& fb);

    bool isFileFull() const {return newFileAtEndOfRun_;}

  private:
    void setBranchAliases(TTree *tree, Selections const& branches) const;

  private:
    struct OutputItem {
      OutputItem() : branchDescription_(0),
	 entryDescriptionIDSharedPtr_(new EntryDescriptionID),
	 entryDescriptionIDPtr_(entryDescriptionIDSharedPtr_.get()),
	 selected_(false), renamed_(false), product_(0) {}
      OutputItem(BranchDescription const* bd, bool sel, bool ren) :
	branchDescription_(bd),
	entryDescriptionIDSharedPtr_(new EntryDescriptionID),
	entryDescriptionIDPtr_(entryDescriptionIDSharedPtr_.get()),
	selected_(sel), renamed_(ren), product_(0) {}
      ~OutputItem() {}
      BranchDescription const* branchDescription_;
      boost::shared_ptr<EntryDescriptionID> entryDescriptionIDSharedPtr_;
      mutable EntryDescriptionID * entryDescriptionIDPtr_;
      bool selected_;
      bool renamed_;
      mutable void const* product_;
      bool operator <(OutputItem const& rh) const {
        return *branchDescription_ < *rh.branchDescription_;
      }
    };
    typedef std::vector<OutputItem> OutputItemList;
    typedef boost::array<OutputItemList, NumBranchTypes> OutputItemListArray;
    void fillItemList(Selections const& keptVector,
		      Selections const& droppedVector,
		      OutputItemList & outputItemList);

    void fillBranches(BranchType const& branchType, Principal const& principal) const;

    void addEntryDescription(EntryDescription const& desc);

    OutputItemListArray outputItemList_;
    std::string file_;
    std::string logicalFile_;
    JobReport::Token reportToken_;
    unsigned int eventCount_;
    unsigned int fileSizeCheckEvent_;
    PoolOutputModule const* om_;
    bool currentlyFastCloning_;
    bool currentlyFastMetaCloning_;
    boost::shared_ptr<TFile> filePtr_;
    FileID fid_;
    FileIndex fileIndex_;
    FileIndex::EntryNumber_t eventEntryNumber_;
    FileIndex::EntryNumber_t lumiEntryNumber_;
    FileIndex::EntryNumber_t runEntryNumber_;
    std::vector<EventProcessHistoryID> eventProcessHistoryIDs_;
    TTree * metaDataTree_;
    TTree * entryDescriptionTree_;
    EventAuxiliary const* pEventAux_;
    LuminosityBlockAuxiliary const* pLumiAux_;
    RunAuxiliary const* pRunAux_;
    ProductStatusVector const* pProductStatuses_;
    RootOutputTree eventTree_;
    RootOutputTree lumiTree_;
    RootOutputTree runTree_;
    RootOutputTreePtrArray treePointers_;
    mutable bool newFileAtEndOfRun_;
  };
}

#endif
