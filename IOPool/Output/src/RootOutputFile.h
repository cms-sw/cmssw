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
#include "IOPool/Output/src/PoolOutputModule.h"
#include "IOPool/Output/src/RootOutputTree.h"

class TTree;
class TFile;
#include "TROOT.h"

namespace edm {
  class PoolOutputModule;
  class History;

  class RootOutputFile {
  public:
    typedef PoolOutputModule::OutputItem OutputItem;
    typedef PoolOutputModule::OutputItemList OutputItemList;
    typedef boost::array<RootOutputTree *, NumBranchTypes> RootOutputTreePtrArray;
    explicit RootOutputFile(PoolOutputModule * om, std::string const& fileName,
                            std::string const& logicalFileName);
    ~RootOutputFile() {}
    void writeOne(EventPrincipal const& e);
    //void endFile();
    void writeLuminosityBlock(LuminosityBlockPrincipal const& lb);
    void writeRun(RunPrincipal const& r);
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
    bool shouldWeCloseFile() const;

  private:

    //-------------------------------
    // Local types
    //
    typedef EventEntryInfo::EntryInfoVector EventEntryInfoVector;
    typedef LumiEntryInfo::EntryInfoVector LumiEntryInfoVector;
    typedef RunEntryInfo::EntryInfoVector RunEntryInfoVector;

    //-------------------------------
    // Private functions

    void setBranchAliases(TTree *tree, Selections const& branches) const;

    template <typename T>
    void fillBranches(BranchType const& branchType, Principal const& principal, std::vector<T> * entryInfoVecPtr);

     template <typename T>
     static void insertAncestors(const T& iParents,
                          const BranchMapper& iMapper,
                          std::set<T>& oToFill);

     void insertAncestors(const EventEntryInfo& iGetParents,
                          const BranchMapper& iMapper,
                          std::set<EventEntryInfo>& oToFill);
        
    //-------------------------------
    // Member data

    std::string file_;
    std::string logicalFile_;
    JobReport::Token reportToken_;
    PoolOutputModule const* om_;
    bool currentlyFastCloning_;
    boost::shared_ptr<TFile> filePtr_;
    FileID fid_;
    FileIndex fileIndex_;
    FileIndex::EntryNumber_t eventEntryNumber_;
    FileIndex::EntryNumber_t lumiEntryNumber_;
    FileIndex::EntryNumber_t runEntryNumber_;
    TTree * metaDataTree_;
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
    bool dataTypeReported_;
    std::set<BranchID> branchesWithStoredHistory_;
  };
   
   //Used by the 'fillBranches' code
   template <typename T>
   void RootOutputFile::insertAncestors(const T& iGetParents,
                                        const BranchMapper& iMapper,
                                        std::set<T>& oToFill) {
      //do nothing
   }

   
   
  template <typename T>
  void RootOutputFile::fillBranches(
		BranchType const& branchType,
		Principal const& principal,
		std::vector<T> * entryInfoVecPtr) {

    std::vector<boost::shared_ptr<EDProduct> > dummies;

    bool const fastCloning = (branchType == InEvent) && currentlyFastCloning_;
    
    OutputItemList const& items = om_->selectedOutputItemList()[branchType];

    std::set<T> keep;

    std::set<T> keepPlusAncestors;

    // Loop over EDProduct branches, fill the provenance, and write the branch.
    for (OutputItemList::const_iterator i = items.begin(), iEnd = items.end(); i != iEnd; ++i) {

      BranchID const& id = i->branchDescription_->branchID();
      branchesWithStoredHistory_.insert(id);
       
      bool getProd = (i->branchDescription_->produced() ||
	 !fastCloning || treePointers_[branchType]->uncloned(i->branchDescription_->branchName()));

      EDProduct const* product = 0;
      OutputHandle<T> const oh = principal.getForOutput<T>(id, getProd);
      if (!oh.entryInfo()) {
	// No product with this ID is in the event.
	// Create and write the provenance.
	if (i->branchDescription_->produced()) {
          keep.insert(T(i->branchDescription_->branchID(),
		      productstatus::neverCreated(),
		      i->branchDescription_->moduleDescriptionID()));
          keepPlusAncestors.insert(T(i->branchDescription_->branchID(),
			      productstatus::neverCreated(),
			      i->branchDescription_->moduleDescriptionID()));
	} else {
          keep.insert(T(i->branchDescription_->branchID(),
		      productstatus::dropped(),
		      i->branchDescription_->moduleDescriptionID()));
          keepPlusAncestors.insert(T(i->branchDescription_->branchID(),
			      productstatus::dropped(),
			      i->branchDescription_->moduleDescriptionID()));
	}
      } else {
	product = oh.wrapper();
        keep.insert(*oh.entryInfo());
        keepPlusAncestors.insert(*oh.entryInfo());
        assert(principal.branchMapperPtr());
        insertAncestors(*oh.entryInfo(),*principal.branchMapperPtr(),keepPlusAncestors);
      }
      if (getProd) {
	if (product == 0) {
	  // No product with this ID is in the event.
	  // Add a null product.
	  TClass *cp = gROOT->GetClass(i->branchDescription_->wrappedName().c_str());
	  boost::shared_ptr<EDProduct> dummy(static_cast<EDProduct *>(cp->New()));
	  dummies.push_back(dummy);
	  product = dummy.get();
	}
	i->product_ = product;
      }
    }
     
    if (om_->dropMetaData()) {
      entryInfoVecPtr->assign(keep.begin(),keep.end());
    } else {
      entryInfoVecPtr->assign(keepPlusAncestors.begin(),keepPlusAncestors.end());
    }
    treePointers_[branchType]->fillTree();
    entryInfoVecPtr->clear();
  }

}

#endif
