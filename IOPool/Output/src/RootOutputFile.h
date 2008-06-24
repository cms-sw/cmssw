#ifndef IOPool_Output_RootOutputFile_h
#define IOPool_Output_RootOutputFile_h

//////////////////////////////////////////////////////////////////////
//
// Class RootOutputFile
//
// Oringinal Author: Luca Lista
// Current Author: Bill Tanenbaum
//
//////////////////////////////////////////////////////////////////////

#include <map>
#include <string>
#include <vector>

#include "boost/array.hpp"
#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchID.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/FileID.h"
#include "DataFormats/Provenance/interface/FileIndex.h"
#include "DataFormats/Provenance/interface/Selections.h"
#include "DataFormats/Provenance/interface/EventEntryInfo.h"
#include "DataFormats/Provenance/interface/RunLumiEntryInfo.h"
#include "IOPool/Output/src/RootOutputTree.h"

class TTree;
class TFile;
#include "TROOT.h"

namespace edm {
  class PoolOutputModule;
  class History;

  class RootOutputFile {
  public:
    typedef boost::array<RootOutputTree *, NumBranchTypes> RootOutputTreePtrArray;
    explicit RootOutputFile(PoolOutputModule * om, std::string const& fileName,
                            std::string const& logicalFileName);
    ~RootOutputFile() {}
    void writeOne(EventPrincipal const& e);
    //void endFile();
    void writeLuminosityBlock(LuminosityBlockPrincipal const& lb);
    bool writeRun(RunPrincipal const& r);
    //BMM void writeBranchMapper();
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
    void writeProductDependencies();

    void finishEndFile();
    void beginInputFile(FileBlock const& fb, bool fastClone);
    void respondToCloseInputFile(FileBlock const& fb);

    bool isFileFull() const {return newFileAtEndOfRun_;}


    struct OutputItem {
      class Sorter {
      public:
        explicit Sorter(TTree * tree);
        bool operator() (OutputItem const& lh, OutputItem const& rh) const;
      private:
        std::map<std::string, int> treeMap_;
      };

      OutputItem() : branchDescription_(0),
	selected_(false), renamed_(false), product_(0) {}

      OutputItem(BranchDescription const* bd, bool sel, bool ren) :
	branchDescription_(bd),
	selected_(sel), renamed_(ren), product_(0) {}

      ~OutputItem() {}

      BranchID branchID() const { return branchDescription_->branchID(); }
      bool     selected() const { return selected_; }


      BranchDescription const* branchDescription_;
      bool selected_;
      bool renamed_;
      mutable void const* product_;

      bool operator <(OutputItem const& rh) const {
        return *branchDescription_ < *rh.branchDescription_;
      }
    };

    typedef std::vector<OutputItem> OutputItemList;

  private:

    //-------------------------------
    // Local types
    //

    typedef boost::array<OutputItemList, NumBranchTypes> OutputItemListArray;

    //-------------------------------
    // Private functions

    void setBranchAliases(TTree *tree, Selections const& branches) const;
    void fillItemList(BranchType branchtype, TTree *theTree);

    template <typename T>
    void fillBranches(BranchType const& branchType, Principal<T> const& principal, std::vector<T> * entryInfoVecPtr);

    //    void addEntryDescription(EntryDescription const& desc);
    void pruneOutputItemList(BranchType branchType, FileBlock const& inputFile);

    //-------------------------------
    // Member data

    OutputItemListArray outputItemList_;
    std::set<BranchID> registryItems_;
    std::string file_;
    std::string logicalFile_;
    JobReport::Token reportToken_;
    unsigned int eventCount_;
    unsigned int fileSizeCheckEvent_;
    PoolOutputModule const* om_;
    bool currentlyFastCloning_;
    boost::shared_ptr<TFile> filePtr_;
    FileID fid_;
    FileIndex fileIndex_;
    FileIndex::EntryNumber_t eventEntryNumber_;
    FileIndex::EntryNumber_t lumiEntryNumber_;
    FileIndex::EntryNumber_t runEntryNumber_;
    TTree * metaDataTree_;
    //BMM TTree * branchMapperTree_;
    TTree * entryDescriptionTree_;
    TTree * eventHistoryTree_;
    EventAuxiliary const*           pEventAux_;
    LuminosityBlockAuxiliary const* pLumiAux_;
    RunAuxiliary const*             pRunAux_;
    EventEntryInfoVector            eventEntryInfoVector_;
    LumiEntryInfoVector	            lumiEntryInfoVector_;
    RunEntryInfoVector              runEntryInfoVector_;
    EventEntryInfoVector *          pEventEntryInfoVector_;
    LumiEntryInfoVector *           pLumiEntryInfoVector_;
    RunEntryInfoVector *            pRunEntryInfoVector_;
    History const*                  pHistory_;
    RootOutputTree eventTree_;
    RootOutputTree lumiTree_;
    RootOutputTree runTree_;
    RootOutputTreePtrArray treePointers_;
    mutable bool newFileAtEndOfRun_;
    bool dataTypeReported_;
  };

  template <typename T>
  void RootOutputFile::fillBranches(
		BranchType const& branchType,
		Principal<T> const& principal,
		std::vector<T> * entryInfoVecPtr) {

    bool const fastCloning = (branchType == InEvent) && currentlyFastCloning_;
    
    OutputItemList const& items = outputItemList_[branchType];

    // Loop over EDProduct branches, fill the provenance, and write the branch.
    for (OutputItemList::const_iterator i = items.begin(), iEnd = items.end(); i != iEnd; ++i) {

      BranchID const& id = i->branchDescription_->branchID();

      bool getProd = i->selected_ && (i->branchDescription_->produced() || i->renamed_ || !fastCloning);

      EDProduct const* product = 0;
      OutputHandle<T> const oh = principal.getForOutput(id, getProd);
      if (!oh.entryInfo()) {
	// No product with this ID is in the event.
	// Create and write the provenance.
	if (i->branchDescription_->produced()) {
          entryInfoVecPtr->push_back(T(i->branchDescription_->branchID(),
			      productstatus::neverCreated(),
			      i->branchDescription_->moduleDescriptionID()));
	} else {
          entryInfoVecPtr->push_back(T(i->branchDescription_->branchID(),
			      productstatus::dropped(),
			      i->branchDescription_->moduleDescriptionID()));
	}
      } else {
	product = oh.wrapper();
        entryInfoVecPtr->push_back(*oh.entryInfo());
      }
      if (getProd) {
	if (product == 0) {
	  // No product with this ID is in the event.
	  // Add a null product.
	  TClass *cp = gROOT->GetClass(i->branchDescription_->wrappedName().c_str());
	  product = static_cast<EDProduct *>(cp->New());
	}
	i->product_ = product;
      }
    }
    sort_all(*entryInfoVecPtr);
    treePointers_[branchType]->fillTree();
    entryInfoVecPtr->clear();
  }

}

#endif
