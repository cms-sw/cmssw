#ifndef IOPool_Input_RootTree_h
#define IOPool_Input_RootTree_h

/*----------------------------------------------------------------------

RootTree.h // used by ROOT input sources

----------------------------------------------------------------------*/

#include "DataFormats/Provenance/interface/ConstBranchDescription.h"
#include "DataFormats/Provenance/interface/ProvenanceFwd.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "Rtypes.h"
#include "TBranch.h"

#include "boost/shared_ptr.hpp"
#include "boost/utility.hpp"

#include <memory>
#include <map>
#include <string>
#include <vector>

class TBranch;
class TClass;
class TFile;
class TTree;
class TTreeCache;

namespace edm {
  struct BranchKey;
  class FileFormatVersion;
  class RootDelayedReader;
  class RootFile;
  class RootTree;

  namespace roottree {
    unsigned int const defaultCacheSize = 20U * 1024 * 1024;
    unsigned int const defaultNonEventCacheSize = 1U * 1024 * 1024;
    unsigned int const defaultLearningEntries = 20U;
    unsigned int const defaultNonEventLearningEntries = 1U;
    typedef Long64_t EntryNumber;
    struct BranchInfo {
      BranchInfo(ConstBranchDescription const& prod) :
        branchDescription_(prod),
        productBranch_(0),
        provenanceBranch_(0),
        classCache_(0),
        offsetToEDProduct_(0) {}
      ConstBranchDescription branchDescription_;
      TBranch* productBranch_;
      TBranch* provenanceBranch_; // For backward compatibility
      mutable TClass* classCache_;
      mutable Int_t offsetToEDProduct_;
    };
    typedef std::map<BranchKey const, BranchInfo> BranchMap;
    Int_t getEntry(TBranch* branch, EntryNumber entryNumber);
    Int_t getEntry(TTree* tree, EntryNumber entryNumber);
    void trainCache(TTree* tree, TFile& file, unsigned int cacheSize, char const* branchNames);
  }

  class RootTree : private boost::noncopyable {
  public:
    typedef roottree::BranchMap BranchMap;
    typedef roottree::EntryNumber EntryNumber;
    RootTree(boost::shared_ptr<TFile> filePtr,
             BranchType const& branchType,
             unsigned int maxVirtualSize,
             unsigned int cacheSize,
             unsigned int learningEntries);
    ~RootTree();

    bool isValid() const;
    void addBranch(BranchKey const& key,
                   BranchDescription const& prod,
                   std::string const& oldBranchName);
    void dropBranch(std::string const& oldBranchName);
    void getEntry(TBranch *branch, EntryNumber entry) const;
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
    template <typename T>
    void fillAux(T*& pAux) {
      auxBranch_->SetAddress(&pAux);
      getEntry(auxBranch_, entryNumber_);
    }
    template <typename T>
    void fillBranchEntryMeta(TBranch* branch, T*& pbuf) {
      if (metaTree_ != 0) {
        // Metadata was in separate tree.  Not cached.
        branch->SetAddress(&pbuf);
        roottree::getEntry(branch, entryNumber_);
      } else {
        fillBranchEntry<T>(branch, pbuf);
      }
    }

    template <typename T>
    void fillBranchEntry(TBranch* branch, T*& pbuf) {
      branch->SetAddress(&pbuf);
      getEntry(branch, entryNumber_);
    }

    TTree const* tree() const {return tree_;}
    TTree* tree() {return tree_;}
    TTree const* metaTree() const {return metaTree_;}
    BranchMap const& branches() const;
    std::vector<ProductStatus> const& productStatuses() const {return productStatuses_;} // backward compatibility

    // below for backward compatibility
    void fillStatus() { // backward compatibility
      statusBranch_->SetAddress(&pProductStatuses_); // backward compatibility
      roottree::getEntry(statusBranch_, entryNumber_); // backward compatibility
    } // backward compatibility

    TBranch* const branchEntryInfoBranch() const {return branchEntryInfoBranch_;}
    void resetTraining() {trainNow_ = true;}

  private:
    void setCacheSize(unsigned int cacheSize);
    void setTreeMaxVirtualSize(int treeMaxVirtualSize);
    void startTraining();
    void stopTraining();

    boost::shared_ptr<TFile> filePtr_;
// We use bare pointers for pointers to some ROOT entities.
// Root owns them and uses bare pointers internally.
// Therefore,using smart pointers here will do no good.
    TTree* tree_;
    TTree* metaTree_;
    BranchType branchType_;
    TBranch* auxBranch_;
    TBranch* branchEntryInfoBranch_;
// We use a smart pointer to own the TTreeCache.
// Unfortunately, ROOT owns it when attached to a TFile, but not after it is detatched.
// So, we make sure to it is detatched before closing the TFile so there is no double delete.
    boost::shared_ptr<TTreeCache> treeCache_;
    boost::shared_ptr<TTreeCache> rawTreeCache_;
    EntryNumber entries_;
    EntryNumber entryNumber_;
    std::vector<std::string> branchNames_;
    boost::shared_ptr<BranchMap> branches_;
    bool trainNow_;
    EntryNumber switchOverEntry_;
    unsigned int learningEntries_;
    unsigned int cacheSize_;

    // below for backward compatibility
    std::vector<ProductStatus> productStatuses_; // backward compatibility
    std::vector<ProductStatus>* pProductStatuses_; // backward compatibility
    TTree* infoTree_; // backward compatibility
    TBranch* statusBranch_; // backward compatibility
  };
}
#endif
