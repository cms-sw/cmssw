#ifndef IOPool_Input_RootTree_h
#define IOPool_Input_RootTree_h

/*----------------------------------------------------------------------

RootTree.h // used by ROOT input sources

----------------------------------------------------------------------*/

#include <memory>
#include <string>
#include <vector>

#include "boost/shared_ptr.hpp"
#include "boost/utility.hpp"

#include "Inputfwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "TBranch.h"
class TFile;

namespace edm {

  class RootTree : private boost::noncopyable {
  public:
    typedef input::BranchMap BranchMap;
    typedef input::EntryNumber EntryNumber;
    RootTree(boost::shared_ptr<TFile> filePtr, BranchType const& branchType);
    ~RootTree();
    
    bool isValid() const;
    void addBranch(BranchKey const& key,
		   BranchDescription const& prod,
		   std::string const& oldBranchName);
    void dropBranch(std::string const& oldBranchName);
    void setPresence(BranchDescription const& prod);
    bool next() {return ++entryNumber_ < entries_;} 
    bool previous() {return --entryNumber_ >= 0;} 
    bool current() {return entryNumber_ < entries_ && entryNumber_ >= 0;} 
    void rewind() {entryNumber_ = 0;} 
    void close();
    EntryNumber const& entryNumber() const {return entryNumber_;}
    EntryNumber const& entries() const {return entries_;}
    void setEntryNumber(EntryNumber theEntryNumber);
    std::vector<std::string> const& branchNames() const {return branchNames_;}
    boost::shared_ptr<DelayedReader> makeDelayedReader(FileFormatVersion const& fileFormatVersion) const;
    //TBranch *auxBranch() {return auxBranch_;}
    template <typename T>
    void fillAux(T *& pAux) {
      auxBranch_->SetAddress(&pAux);
      input::getEntryWithCache(auxBranch_, entryNumber_, treeCache_.get(), filePtr_.get());
    }
    TTree const* tree() const {return tree_;}
    TTree const* metaTree() const {return metaTree_;}
    void setCacheSize(unsigned int cacheSize);
    void setTreeMaxVirtualSize(int treeMaxVirtualSize);
    BranchMap const& branches() const {return *branches_;}
    std::vector<ProductStatus> const& productStatuses() const {return productStatuses_;} // backward compatibility

    // below for backward compatibility
    void fillStatus() { // backward compatibility
      statusBranch_->SetAddress(&pProductStatuses_); // backward compatibility
      input::getEntry(statusBranch_, entryNumber_); // backward compatibility
    } // backward compatibility

    TBranch *const branchEntryInfoBranch() const {return branchEntryInfoBranch_;}
    void resetTraining() {trained_ = kFALSE;}

  private:
    boost::shared_ptr<TFile> filePtr_;
// We use bare pointers for pointers to some ROOT entities.
// Root owns them and uses bare pointers internally.
// Therefore,using smart pointers here will do no good.
    TTree * tree_;
    TTree * metaTree_;
    BranchType branchType_;
    TBranch * auxBranch_;
    TBranch * branchEntryInfoBranch_;
// We use a smart pointer to own the TTreeCache.
// Unfortunately, ROOT owns it when attached to a TFile, but not after it is detatched.
// So, we make sure to it is detatched before closing the TFile so there is no double delete.
    boost::shared_ptr<TTreeCache> treeCache_;
    EntryNumber entries_;
    EntryNumber entryNumber_;
    std::vector<std::string> branchNames_;
    boost::shared_ptr<BranchMap> branches_;
    bool trained_; // Set to true if the ROOT TTreeCache started training.

    // below for backward compatibility
    std::vector<ProductStatus> productStatuses_; // backward compatibility
    std::vector<ProductStatus>* pProductStatuses_; // backward compatibility
    TTree * infoTree_; // backward compatibility
    TBranch * statusBranch_; // backward compatibility
  };
}
#endif
