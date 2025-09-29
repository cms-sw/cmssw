
#include "RootOutputRNTuple.h"

#include "DataFormats/Common/interface/RefCoreStreamer.h"
#include "DataFormats/Provenance/interface/ProductDescription.h"
#include "FWCore/AbstractServices/interface/RootHandlers.h"
#include "FWCore/MessageLogger/interface/JobReport.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Algorithms.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Catalog/interface/SiteLocalConfig.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include "TBranch.h"
#include "TBranchElement.h"
#include "TCollection.h"
#include "TFile.h"
#include "Rtypes.h"
#include "RVersion.h"

#include <limits>

#include "oneapi/tbb/task_arena.h"

namespace edm {

  RootOutputRNTuple::RootOutputRNTuple(std::shared_ptr<TFile> filePtr,
                                       BranchType const& branchType,
                                       int splitLevel,
                                       int treeMaxVirtualSize,
                                       std::string const& processName)
      : filePtr_(filePtr),
        tree_(processName.empty()
                  ? makeTTree(filePtr.get(), BranchTypeToProductTreeName(branchType), splitLevel)
                  : makeTTree(filePtr.get(), BranchTypeToProductTreeName(branchType, processName), splitLevel)),
        producedBranches_(),
        auxBranches_() {
    if (treeMaxVirtualSize >= 0)
      tree_->SetMaxVirtualSize(treeMaxVirtualSize);
  }

  TTree* RootOutputRNTuple::assignTTree(TFile* filePtr, TTree* tree) {
    tree->SetDirectory(filePtr);
    // Turn off autosaving because it is such a memory hog and we are not using
    // this check-pointing feature anyway.
    tree->SetAutoSave(std::numeric_limits<Long64_t>::max());
    return tree;
  }

  TTree* RootOutputRNTuple::makeTTree(TFile* filePtr, std::string const& name, int splitLevel) {
    TTree* tree = new TTree(name.c_str(), "", splitLevel);
    if (!tree)
      throw edm::Exception(errors::FatalRootError) << "Failed to create the tree: " << name << "\n";
    if (tree->IsZombie())
      throw edm::Exception(errors::FatalRootError) << "Tree: " << name << " is a zombie."
                                                   << "\n";

    return assignTTree(filePtr, tree);
  }

  void RootOutputRNTuple::writeTTree(TTree* tree) {
    if (tree->GetNbranches() != 0) {
      // This is required when Fill is called on individual branches
      // in the TTree instead of calling Fill once for the entire TTree.
      tree->SetEntries(-1);
    }
    tree->AutoSave("FlushBaskets");
  }

  void RootOutputRNTuple::fillTTree(std::vector<TBranch*> const& branches) {
    for_all(branches, std::bind(&TBranch::Fill, std::placeholders::_1));
  }

  void RootOutputRNTuple::writeTree() { writeTTree(tree()); }

  void RootOutputRNTuple::fillTree() {
    // Isolate the fill operation so that IMT doesn't grab other large tasks
    // that could lead to RNTupleTempOutputModule stalling
    oneapi::tbb::this_task_arena::isolate([&] { tree_->Fill(); });
  }

  void RootOutputRNTuple::addBranch(std::string const& branchName,
                                    std::string const& className,
                                    void const*& pProd,
                                    int splitLevel,
                                    int basketSize,
                                    bool produced) {
    assert(splitLevel != ProductDescription::invalidSplitLevel);
    assert(basketSize != ProductDescription::invalidBasketSize);
    TBranch* branch = tree_->Branch(branchName.c_str(), className.c_str(), &pProd, basketSize, splitLevel);
    assert(branch != nullptr);
    producedBranches_.push_back(branch);
  }

  void RootOutputRNTuple::close() {
    // The TFile was just closed.
    // Just to play it safe, zero all pointers to quantities in the file.
    auxBranches_.clear();
    producedBranches_.clear();
    tree_ = nullptr;     // propagate_const<T> has no reset() function
    filePtr_ = nullptr;  // propagate_const<T> has no reset() function
  }
}  // namespace edm
