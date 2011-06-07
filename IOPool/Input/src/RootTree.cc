#include "RootTree.h"
#include "RootDelayedReader.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "InputFile.h"
#include "TTree.h"
#include "TTreeIndex.h"
#include "TTreeCache.h"

#include <iostream>

namespace edm {
  namespace {
    TBranch* getAuxiliaryBranch(TTree* tree, BranchType const& branchType) {
      TBranch* branch = tree->GetBranch(BranchTypeToAuxiliaryBranchName(branchType).c_str());
      if (branch == 0) {
        branch = tree->GetBranch(BranchTypeToAuxBranchName(branchType).c_str());
      }
      return branch;
    }
    TBranch* getProductProvenanceBranch(TTree* tree, BranchType const& branchType) {
      TBranch* branch = tree->GetBranch(BranchTypeToBranchEntryInfoBranchName(branchType).c_str());
      return branch;
    }
    TBranch* getStatusBranch(TTree* tree, BranchType const& branchType) { // backward compatibility
      TBranch* branch = tree->GetBranch(BranchTypeToProductStatusBranchName(branchType).c_str()); // backward compatibility
      return branch; // backward compatibility
    } // backward compatibility
  }
  RootTree::RootTree(boost::shared_ptr<InputFile> filePtr,
                     BranchType const& branchType,
                     unsigned int maxVirtualSize,
                     unsigned int cacheSize,
                     unsigned int learningEntries) :
    filePtr_(filePtr),
    tree_(dynamic_cast<TTree*>(filePtr_.get() != 0 ? filePtr->Get(BranchTypeToProductTreeName(branchType).c_str()) : 0)),
    metaTree_(dynamic_cast<TTree*>(filePtr_.get() != 0 ? filePtr->Get(BranchTypeToMetaDataTreeName(branchType).c_str()) : 0)),
    branchType_(branchType),
    auxBranch_(tree_ ? getAuxiliaryBranch(tree_, branchType_) : 0),
    treeCache_(),
    rawTreeCache_(),
    entries_(tree_ ? tree_->GetEntries() : 0),
    entryNumber_(-1),
    branchNames_(),
    branches_(new BranchMap),
    trainNow_(false),
    switchOverEntry_(-1),
    learningEntries_(learningEntries),
    cacheSize_(cacheSize),
    branchEntryInfoBranch_(metaTree_ ? getProductProvenanceBranch(metaTree_, branchType_) : (tree_ ? getProductProvenanceBranch(tree_, branchType_) : 0)),
    productStatuses_(), // backward compatibility
    pProductStatuses_(&productStatuses_), // backward compatibility
    infoTree_(dynamic_cast<TTree*>(filePtr_.get() != 0 ? filePtr->Get(BranchTypeToInfoTreeName(branchType).c_str()) : 0)), // backward compatibility
    statusBranch_(infoTree_ ? getStatusBranch(infoTree_, branchType_) : 0) { // backward compatibility
      assert(tree_);
      setTreeMaxVirtualSize(maxVirtualSize);
      setCacheSize(cacheSize);
  }

  RootTree::~RootTree() {
  }

  bool
  RootTree::isValid() const {
    if (metaTree_ == 0 || metaTree_->GetNbranches() == 0) {
      return tree_ != 0 && auxBranch_ != 0;
    }
    if (tree_ != 0 && auxBranch_ != 0 && metaTree_ != 0) { // backward compatibility
      if (branchEntryInfoBranch_ != 0 || statusBranch_ != 0) return true; // backward compatibility
      return (entries_ == metaTree_->GetEntries() && tree_->GetNbranches() <= metaTree_->GetNbranches() + 1);  // backward compatibility
    } // backward compatibility
    return false;
  }

  void
  RootTree::setPresence(BranchDescription const& prod) {
      assert(isValid());
      prod.init();
      if(tree_->GetBranch(prod.branchName().c_str()) == 0){
        prod.setDropped();
      }
  }

  void
  RootTree::addBranch(BranchKey const& key,
                      BranchDescription const& prod,
                      std::string const& oldBranchName) {
      assert(isValid());
      prod.init();
      //use the translated branch name
      TBranch* branch = tree_->GetBranch(oldBranchName.c_str());
      roottree::BranchInfo info = roottree::BranchInfo(ConstBranchDescription(prod));
      info.productBranch_ = 0;
      if (prod.present()) {
        info.productBranch_ = branch;
        //we want the new branch name for the JobReport
        branchNames_.push_back(prod.branchName());
      }
      TTree* provTree = (metaTree_ != 0 ? metaTree_ : tree_);
      info.provenanceBranch_ = provTree->GetBranch(oldBranchName.c_str());
      branches_->insert(std::make_pair(key, info));
  }

  void
  RootTree::dropBranch(std::string const& oldBranchName) {
      //use the translated branch name
      TBranch* branch = tree_->GetBranch(oldBranchName.c_str());
      if (branch != 0) {
        TObjArray* leaves = tree_->GetListOfLeaves();
        int entries = leaves->GetEntries();
        for (int i = 0; i < entries; ++i) {
          TLeaf* leaf = (TLeaf*)(*leaves)[i];
          if (leaf == 0) continue;
          TBranch* br = leaf->GetBranch();
          if (br == 0) continue;
          if (br->GetMother() == branch) {
            leaves->Remove(leaf);
          }
        }
        leaves->Compress();
        tree_->GetListOfBranches()->Remove(branch);
        tree_->GetListOfBranches()->Compress();
        delete branch;
      }
  }

  roottree::BranchMap const&
  RootTree::branches() const {return *branches_;}

  boost::shared_ptr<DelayedReader>
  RootTree::makeDelayedReader(FileFormatVersion const& fileFormatVersion, boost::shared_ptr<RootFile> rootFilePtr) const {
    boost::shared_ptr<DelayedReader>
        store(new RootDelayedReader(entryNumber_, branches_, *this, fileFormatVersion, rootFilePtr));
    return store;
  }

  void
  RootTree::setCacheSize(unsigned int cacheSize) {
    cacheSize_ = cacheSize;
    tree_->SetCacheSize(static_cast<Long64_t>(cacheSize));
    treeCache_.reset(dynamic_cast<TTreeCache*>(filePtr_->GetCacheRead()));
    filePtr_->SetCacheRead(0);
    rawTreeCache_.reset();
  }

  void
  RootTree::setTreeMaxVirtualSize(int treeMaxVirtualSize) {
    if (treeMaxVirtualSize >= 0) tree_->SetMaxVirtualSize(static_cast<Long64_t>(treeMaxVirtualSize));
  }

  void
  RootTree::setEntryNumber(EntryNumber theEntryNumber) {
    filePtr_->SetCacheRead(treeCache_.get());
    entryNumber_ = theEntryNumber;
    tree_->LoadTree(entryNumber_);
    filePtr_->SetCacheRead(0);
    if(treeCache_ && trainNow_ && entryNumber_ >= 0) {
      startTraining();
      trainNow_ = false;
    }
    if (treeCache_ && treeCache_->IsLearning() && switchOverEntry_ >= 0 && entryNumber_ >= switchOverEntry_) {
      stopTraining();
    }
  }

  void
  RootTree::getEntry(TBranch* branch, EntryNumber entryNumber) const {
    if (!treeCache_) {
      filePtr_->SetCacheRead(0);
      roottree::getEntry(branch, entryNumber);
    } else if (treeCache_->IsLearning() && rawTreeCache_) {
      treeCache_->AddBranch(branch, kTRUE);
      filePtr_->SetCacheRead(rawTreeCache_.get());
      roottree::getEntry(branch, entryNumber);
      filePtr_->SetCacheRead(0);
    } else {
      filePtr_->SetCacheRead(treeCache_.get());
      roottree::getEntry(branch, entryNumber);
      filePtr_->SetCacheRead(0);
    }
  }

  void
  RootTree::startTraining() {
    if (cacheSize_ == 0) {
      return;
    }
    assert(treeCache_ && treeCache_->GetOwner() == tree_);
    assert(branchType_ == InEvent);
    assert(!rawTreeCache_);
    treeCache_->SetLearnEntries(learningEntries_);
    tree_->SetCacheSize(static_cast<Long64_t>(cacheSize_));
    rawTreeCache_.reset(dynamic_cast<TTreeCache *>(filePtr_->GetCacheRead()));
    filePtr_->SetCacheRead(0);
    rawTreeCache_->SetLearnEntries(0);
    switchOverEntry_ = entryNumber_ + learningEntries_;
    rawTreeCache_->StartLearningPhase();
    rawTreeCache_->SetEntryRange(entryNumber_, switchOverEntry_);
    rawTreeCache_->AddBranch("*", kTRUE);
    rawTreeCache_->StopLearningPhase();
    treeCache_->StartLearningPhase();
    treeCache_->SetEntryRange(switchOverEntry_, tree_->GetEntries());
    treeCache_->AddBranch(poolNames::branchListIndexesBranchName().c_str(), kTRUE);
    treeCache_->AddBranch(BranchTypeToAuxiliaryBranchName(branchType_).c_str(), kTRUE);
  }

  void
  RootTree::stopTraining() {
    filePtr_->SetCacheRead(treeCache_.get());
    treeCache_->StopLearningPhase();
    rawTreeCache_.reset();
  }

  void
  RootTree::close () {
    // The TFile is about to be closed, and destructed.
    // Just to play it safe, zero all pointers to quantities that are owned by the TFile.
    auxBranch_  = branchEntryInfoBranch_ = statusBranch_ = 0;
    tree_ = metaTree_ = infoTree_ = 0;
    // We own the treeCache_.
    // We make sure the treeCache_ is detached from the file,
    // so that ROOT does not also delete it.
    filePtr_->SetCacheRead(0);
    // We give up our shared ownership of the TFile itself.
    filePtr_.reset();
  }

  void
  RootTree::trainCache(char const* branchNames) {
    if (cacheSize_ == 0) {
      return;
    }
    tree_->LoadTree(0);
    assert(treeCache_);
    filePtr_->SetCacheRead(treeCache_.get());
    assert(treeCache_->GetOwner() == tree_);
    treeCache_->StartLearningPhase();
    treeCache_->SetEntryRange(0, tree_->GetEntries());
    treeCache_->AddBranch(branchNames, kTRUE);
    treeCache_->StopLearningPhase();
    // We own the treeCache_.
    // We make sure the treeCache_ is detached from the file,
    // so that ROOT does not also delete it.
    filePtr_->SetCacheRead(0);
  }

  namespace roottree {
    Int_t
    getEntry(TBranch* branch, EntryNumber entryNumber) {
      Int_t n = 0;
      try {
        n = branch->GetEntry(entryNumber);
      }
      catch(cms::Exception const& e) {
        throw Exception(errors::FileReadError, "", e);
      }
      return n;
    }

    Int_t
    getEntry(TTree* tree, EntryNumber entryNumber) {
      Int_t n = 0;
      try {
        n = tree->GetEntry(entryNumber);
      }
      catch(cms::Exception const& e) {
        throw Exception (errors::FileReadError, "", e);
      }
      return n;
    }

    std::auto_ptr<TTreeCache>
    trainCache(TTree* tree, InputFile& file, unsigned int cacheSize, char const* branchNames) {
      tree->LoadTree(0);
      tree->SetCacheSize(cacheSize);
      std::auto_ptr<TTreeCache> treeCache(dynamic_cast<TTreeCache*>(file.GetCacheRead()));
      if (0 != treeCache.get()) {
        treeCache->StartLearningPhase();
        treeCache->SetEntryRange(0, tree->GetEntries());
        treeCache->AddBranch(branchNames, kTRUE);
        treeCache->StopLearningPhase();
      }
      // We own the treeCache_.
      // We make sure the treeCache_ is detached from the file,
      // so that ROOT does not also delete it.
      file.SetCacheRead(0);
      return treeCache;
    }
  }
}
