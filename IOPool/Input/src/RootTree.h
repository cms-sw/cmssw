#ifndef IOPool_Input_RootTree_h
#define IOPool_Input_RootTree_h

/*----------------------------------------------------------------------

RootTree.h // used by ROOT input sources

$Id: RootTree.h,v 1.14 2007/08/17 22:54:17 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <string>

#include "boost/shared_ptr.hpp"

#include "Inputfwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "TBranch.h"
class TFile;
class TTree;

namespace edm {

  class RootTree {
  public:
    typedef input::BranchMap BranchMap;
    typedef input::EntryNumber EntryNumber;
    RootTree(boost::shared_ptr<TFile> filePtr, BranchType const& branchType);
    ~RootTree() {}
    
    bool isValid() const;
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
    std::vector<std::string> const& branchNames() const {return branchNames_;}
    void fillGroups(Principal& item);
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
    boost::shared_ptr<BranchMap> branches_;
  };
}
#endif
