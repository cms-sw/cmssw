
#include "RootOutputTree.h"

#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/RootHandlers.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TBranch.h"
#include "TBranchElement.h"
#include "TCollection.h"
#include "TFile.h"
#include "TTreeCloner.h"
#include "Rtypes.h"
#include "RVersion.h"

#include <limits>

namespace edm {

    /**
     * Currently, ROOT doesn't use any latency-hiding optimizations for
     * fast-cloning.  This causes a significant slowdown when doing fast-cloning
     * over a high-latency network (30ms latency makes this multiple factors slower).
     *
     * Accordingly, we allow sites to provide a separate hint on how to treat fast-
     * cloning.  The DuplicateTreeSentry allows us to implement it - given a tree
     * we are about to clone, with the appropriate configs, this will re-open the
     * file with lazy-download and re-open the tree.  The new tree is appropriate
     * for cloning.  When the object is destroyed, the new file and tree are cleaned up.
     *
     */
    class DuplicateTreeSentry
    {
    public:
        DuplicateTreeSentry(TTree * tree)
          : tree_(tree)
        {
            dup();
        }

        TTree *tree() const {return mytree_ ? mytree_.get() : tree_;}

    private:
        DuplicateTreeSentry(DuplicateTreeSentry const&) = delete; // Disallow copying and moving
        DuplicateTreeSentry& operator=(DuplicateTreeSentry const&) = delete;
        struct CloseBeforeDelete { void operator()(TFile* iFile) const { if( iFile) { iFile->Close(); } delete iFile; } };

        void dup()
        {
            edm::Service<edm::SiteLocalConfig> pSLC;
            if (!pSLC.isAvailable()) {return;}
            if (pSLC->sourceCacheHint() && *(pSLC->sourceCacheHint()) == "lazy-download") {return;}
            if (!pSLC->sourceCloneCacheHint() || *(pSLC->sourceCloneCacheHint()) != "lazy-download") {return;}
            edm::LogWarning("DuplicateTreeSentry") << "Re-opening file for fast-cloning";

            TFile *file = tree_->GetCurrentFile();
            const TUrl *url = file->GetEndpointUrl();
            if (!url)
            {
                return;
            }
            file_.reset(TFile::Open(url->GetUrl(), "READWRAP")); // May throw an exception.
            if (!file_)
            {
                return;
            }
            mytree_.reset(dynamic_cast<TTree*>(file_->Get(tree_->GetName())));
            if (!mytree_) {return;}
        }

        /**
         * Note this relies on the implicit delete ordering - mytree_ (if non-null)
         * must be deleted before file_.  Do not reorder the class members!
         */
        std::unique_ptr<TFile, CloseBeforeDelete> file_;
        TTree *tree_ = nullptr;
        std::unique_ptr<TTree> mytree_ = nullptr;
    };

    RootOutputTree::RootOutputTree(
                   std::shared_ptr<TFile> filePtr,
                   BranchType const& branchType,
                   int splitLevel,
                   int treeMaxVirtualSize) :
      filePtr_(filePtr),
      tree_(makeTTree(filePtr.get(), BranchTypeToProductTreeName(branchType), splitLevel)),
      producedBranches_(),
      readBranches_(),
      auxBranches_(),
      unclonedReadBranches_(),
      clonedReadBranchNames_(),
      currentlyFastCloning_(),
      fastCloneAuxBranches_(false) {

    if(treeMaxVirtualSize >= 0) tree_->SetMaxVirtualSize(treeMaxVirtualSize);
  }

  TTree*
  RootOutputTree::assignTTree(TFile* filePtr, TTree* tree) {
    tree->SetDirectory(filePtr);
    // Turn off autosaving because it is such a memory hog and we are not using
    // this check-pointing feature anyway.
    tree->SetAutoSave(std::numeric_limits<Long64_t>::max());
    return tree;
  }

  TTree*
  RootOutputTree::makeTTree(TFile* filePtr, std::string const& name, int splitLevel) {
    TTree* tree = new TTree(name.c_str(), "", splitLevel);
    if(!tree) throw edm::Exception(errors::FatalRootError)
      << "Failed to create the tree: " << name << "\n";
    if(tree->IsZombie())
      throw edm::Exception(errors::FatalRootError)
        << "Tree: " << name << " is a zombie." << "\n";

    return assignTTree(filePtr, tree);
  }

  bool
  RootOutputTree::checkSplitLevelsAndBasketSizes(TTree* inputTree) const {

    assert(inputTree != nullptr);

    // Do the split level and basket size match in the input and output?
    for(std::vector<TBranch*>::const_iterator it = readBranches_.begin(), itEnd = readBranches_.end();
      it != itEnd; ++it) {

      TBranch* outputBranch = *it;
      if(outputBranch != nullptr) {
        TBranch* inputBranch = inputTree->GetBranch(outputBranch->GetName());

        if(inputBranch != nullptr) {
          if(inputBranch->GetSplitLevel() != outputBranch->GetSplitLevel() ||
              inputBranch->GetBasketSize() != outputBranch->GetBasketSize()) {
            return false;
          }
        }
      }
    }
    return true;
  }

  namespace {
    bool checkMatchingBranches(TBranchElement* inputBranch, TBranchElement* outputBranch) {
      if(inputBranch->GetStreamerType() != outputBranch->GetStreamerType()) {
        return false;
      }
      TObjArray* inputArray = inputBranch->GetListOfBranches();
      TObjArray* outputArray = outputBranch->GetListOfBranches();

      if(outputArray->GetSize() < inputArray->GetSize()) {
        return false;
      }
      TIter iter(outputArray);
      TObject* obj = nullptr;
      while((obj = iter.Next()) != nullptr) {
        TBranchElement* outBranch = dynamic_cast<TBranchElement*>(obj);
        if(outBranch) {
          TBranchElement* inBranch = dynamic_cast<TBranchElement*>(inputArray->FindObject(outBranch->GetName()));
          if(!inBranch) {
            return false;
          }
          if(!checkMatchingBranches(inBranch, outBranch)) {
            return false;
          }
        }
      }
      return true;
    }
  }

  bool RootOutputTree::checkIfFastClonable(TTree* inputTree) const {

    if(inputTree == nullptr) return false;

    // Do the sub-branches match in the input and output. Extra sub-branches in the input are OK for fast cloning, but not in the output.
    for(std::vector<TBranch*>::const_iterator it = readBranches_.begin(), itEnd = readBranches_.end(); it != itEnd; ++it) {
      TBranchElement* outputBranch = dynamic_cast<TBranchElement*>(*it);
      if(outputBranch != nullptr) {
        TBranchElement* inputBranch = dynamic_cast<TBranchElement*>(inputTree->GetBranch(outputBranch->GetName()));
        if(inputBranch != nullptr) {
          // We have a matching top level branch. Do the recursive check on subbranches.
          if(!checkMatchingBranches(inputBranch, outputBranch)) {
            LogInfo("FastCloning")
              << "Fast Cloning disabled because a data member has been added to split branch: " << inputBranch->GetName() << "\n.";
            return false;
          }
        }
      }
    }
    return true;
  }

  bool RootOutputTree::checkEntriesInReadBranches(Long64_t expectedNumberOfEntries) const {
    for(std::vector<TBranch*>::const_iterator it = readBranches_.begin(), itEnd = readBranches_.end(); it != itEnd; ++it) {
      if((*it)->GetEntries() != expectedNumberOfEntries) {
        return false;
      }
    }
    return true;
  }

  void
  RootOutputTree::fastCloneTTree(TTree* in, std::string const& option) {
    if(in->GetEntries() != 0) {
      TObjArray* branches = tree_->GetListOfBranches();
      // If any products were produced (not just event products), the EventAuxiliary will be modified.
      // In that case, don't fast copy auxiliary branches. Remove them, and add back after fast copying.
      std::map<Int_t, TBranch *> auxIndexes;
      bool mustRemoveSomeAuxs = false;
      if(!fastCloneAuxBranches_) {
        for(std::vector<TBranch *>::const_iterator it = auxBranches_.begin(), itEnd = auxBranches_.end();
             it != itEnd; ++it) {
          int auxIndex = branches->IndexOf(*it);
          assert (auxIndex >= 0);
          auxIndexes.insert(std::make_pair(auxIndex, *it));
          branches->RemoveAt(auxIndex);
        }
        mustRemoveSomeAuxs = true;
      }

      //Deal with any aux branches which can never be cloned
      for(std::vector<TBranch *>::const_iterator it = unclonedAuxBranches_.begin(),
           itEnd = unclonedAuxBranches_.end();
           it != itEnd; ++it) {
        int auxIndex = branches->IndexOf(*it);
        assert (auxIndex >= 0);
        auxIndexes.insert(std::make_pair(auxIndex, *it));
        branches->RemoveAt(auxIndex);
        mustRemoveSomeAuxs = true;
      }

      if(mustRemoveSomeAuxs) {
        branches->Compress();
      }

      DuplicateTreeSentry dupTree(in);
      TTreeCloner cloner(dupTree.tree(), tree_, option.c_str(), TTreeCloner::kNoWarnings|TTreeCloner::kIgnoreMissingTopLevel);

      if(!cloner.IsValid()) {
        // Let's check why
        static const char* okerror = "One of the export branch";
        if(strncmp(cloner.GetWarning(), okerror, strlen(okerror)) == 0) {
          // That's fine we will handle it;
        } else {
          throw edm::Exception(errors::FatalRootError)
            << "invalid TTreeCloner (" << cloner.GetWarning() << ")\n";
        }
      }
      tree_->SetEntries(tree_->GetEntries() + in->GetEntries());
      Service<RootHandlers> rootHandler;
      rootHandler->ignoreWarningsWhileDoing([&cloner] { cloner.Exec(); });

      if(mustRemoveSomeAuxs) {
        for(std::map<Int_t, TBranch *>::const_iterator it = auxIndexes.begin(), itEnd = auxIndexes.end();
             it != itEnd; ++it) {
          // Add the auxiliary branches back after fast copying the rest of the tree.
          Int_t last = branches->GetLast();
          if(last >= 0) {
            branches->AddAtAndExpand(branches->At(last), last+1);
            for(Int_t ind = last-1; ind >= it->first; --ind) {
              branches->AddAt(branches->At(ind), ind+1);
            };
            branches->AddAt(it->second, it->first);
          } else {
            branches->Add(it->second);
          }
        }
      }
    }
  }

  void
  RootOutputTree::writeTTree(TTree* tree) {
    if(tree->GetNbranches() != 0) {
      tree->SetEntries(-1);
    }
    setRefCoreStreamer(true);
    tree->AutoSave("FlushBaskets");
  }

  void
  RootOutputTree::fillTTree(std::vector<TBranch*> const& branches) {
    for_all(branches, std::bind(&TBranch::Fill, std::placeholders::_1));
  }

  void
  RootOutputTree::writeTree() const {
    writeTTree(tree_);
  }

  void
  RootOutputTree::maybeFastCloneTree(bool canFastClone, bool canFastCloneAux, TTree* tree, std::string const& option) {
    unclonedReadBranches_.clear();
    clonedReadBranchNames_.clear();
    currentlyFastCloning_ = canFastClone && !readBranches_.empty();
    if(currentlyFastCloning_) {
      fastCloneAuxBranches_ = canFastCloneAux;
      fastCloneTTree(tree, option);
      for(std::vector<TBranch*>::const_iterator it = readBranches_.begin(), itEnd = readBranches_.end();
          it != itEnd; ++it) {
        if((*it)->GetEntries() == tree_->GetEntries()) {
          clonedReadBranchNames_.insert(std::string((*it)->GetName()));
        } else {
          unclonedReadBranches_.push_back(*it);
        }
      }
      Service<JobReport> reportSvc;
      reportSvc->reportFastClonedBranches(clonedReadBranchNames_, tree_->GetEntries());
    }
  }

  void
  RootOutputTree::fillTree() const {
    if(currentlyFastCloning_) {
      if(!fastCloneAuxBranches_)fillTTree(auxBranches_);
      fillTTree(unclonedAuxBranches_);
      fillTTree(producedBranches_);
      fillTTree(unclonedReadBranches_);
    } else {
      tree_->Fill();
    }
  }

  void
  RootOutputTree::addBranch(std::string const& branchName,
                            std::string const& className,
                            void const*& pProd,
                            int splitLevel,
                            int basketSize,
                            bool produced) {
      assert(splitLevel != BranchDescription::invalidSplitLevel);
      assert(basketSize != BranchDescription::invalidBasketSize);
      TBranch* branch = tree_->Branch(branchName.c_str(),
                 className.c_str(),
                 &pProd,
                 basketSize,
                 splitLevel);
      assert(branch != nullptr);
/*
      if(pProd != nullptr) {
        // Delete the product that ROOT has allocated.
        WrapperBase const* edp = static_cast<WrapperBase const *>(pProd);
        delete edp;
        pProd = nullptr;
      }
*/
      if(produced) {
        producedBranches_.push_back(branch);
      } else {
        readBranches_.push_back(branch);
      }
  }

  void
  RootOutputTree::close() {
    // The TFile was just closed.
    // Just to play it safe, zero all pointers to quantities in the file.
    auxBranches_.clear();
    unclonedAuxBranches_.clear();
    producedBranches_.clear();
    readBranches_.clear();
    unclonedReadBranches_.clear();
    tree_ = nullptr;
    filePtr_.reset();
  }
}
