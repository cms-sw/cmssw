#ifndef Input_RootTree_h
#define Input_RootTree_h

/*----------------------------------------------------------------------

RootTree.h // used by ROOT input sources

$Id: RootTree.h,v 1.1 2006/12/23 03:16:12 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <string>

#include "boost/shared_ptr.hpp"

#include "IOPool/Input/src/Inputfwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/Common/interface/BranchEntryDescription.h"
#include "DataFormats/Common/interface/BranchDescription.h"
#include "DataFormats/Common/interface/BranchKey.h"
#include "DataFormats/Common/interface/BranchType.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "TBranch.h"
#include "TFile.h"

namespace edm {

  class RootTree {
  public:
    typedef input::BranchMap BranchMap;
    typedef std::map<ProductID const, BranchDescription const> ProductMap;
    typedef input::EntryNumber EntryNumber;
    RootTree(boost::shared_ptr<TFile> filePtr, BranchType const& branchType);
    ~RootTree() {}
    
    void addBranch(BranchKey const& key,
		   BranchDescription const& prod,
		   std::string const& oldBranchName);
    bool next() {return ++entryNumber_ < entries_;} 
    bool previous() {return --entryNumber_ >= 0;} 
    EntryNumber const& entryNumber() const {return entryNumber_;}
    EntryNumber const& entries() const {return entries_;}
    EntryNumber getBestEntryNumber(unsigned int major, unsigned int minor) const;
    EntryNumber getExactEntryNumber(unsigned int major, unsigned int minor) const;
    void setEntryNumber(EntryNumber theEntryNumber) {entryNumber_ = theEntryNumber;}
    void resetEntryNumber() {entryNumber_ = origEntryNumber_;}
    void setOrigEntryNumber() {origEntryNumber_ = entryNumber_;}
    std::vector<std::string> & branchNames() {return branchNames_;}
    void fillGroups(DataBlockImpl& item);
    boost::shared_ptr<DelayedReader> makeDelayedReader() const;
    //TBranch *auxBranch() {return auxBranch_;}
    template <typename T>
    void fillAux(T *& pAux) const {
      auxBranch_->SetAddress(&pAux);
      auxBranch_->GetEntry(entryNumber_);
    }
  private:
    boost::shared_ptr<TFile> filePtr_;
// We use bare pointers for pointers to some ROOT entities.
// Root owns them and uses bare pointers internally.
// Therefore,using smart pointers here will do no good.
    TTree *const tree_;
    TTree *const metaTree_;
    TBranch *const auxBranch_;
    EntryNumber entries_;
    EntryNumber entryNumber_;
    EntryNumber origEntryNumber_;
    std::vector<std::string> branchNames_;
    std::vector<BranchEntryDescription> provenance_;
    std::vector<BranchEntryDescription const*> provenancePtrs_;
    boost::shared_ptr<BranchMap> branches_;
    ProductMap products_;
  };
}
#endif
