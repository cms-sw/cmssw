#ifndef IOPool_Output_RootOutputTree_h
#define IOPool_Output_RootOutputTree_h

/*----------------------------------------------------------------------

RootOutputTree.h // used by ROOT output modules

$Id: RootOutputTree.h,v 1.6 2007/09/10 20:27:09 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>
#include <string>
#include <vector>

#include "boost/shared_ptr.hpp"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchEntryDescription.h"
#include "DataFormats/Provenance/interface/BranchKey.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/ConstBranchDescription.h"
#include "DataFormats/Provenance/interface/Selections.h"
#include "DataFormats/Common/interface/Wrapper.h"
#include "TBranch.h"
#include "TTree.h"
class TFile;
class TChain;

namespace edm {

  class RootOutputTree {
  public:
    template <typename T>
    RootOutputTree(
		   boost::shared_ptr<TFile> filePtr,
		   BranchType const& branchType,
		   T const*& pAux,
		   int bufSize,
		   int splitLevel,
		   TChain * chain = 0,
		   TChain * metaChain = 0,
		   Selections const& dropList = Selections()) :
      fastCloning_(chain != 0 && metaChain != 0),
      filePtr_(filePtr),
      tree_(makeTree(filePtr.get(),
		     BranchTypeToProductTreeName(branchType),
		     splitLevel,
		     (fastCloning_ ? chain : 0),
		     dropList)),
      metaTree_(makeTree(filePtr.get(),
	        BranchTypeToMetaDataTreeName(branchType),
		0,
		(fastCloning_ ? metaChain : 0))),
      auxBranch_(tree_->Branch(BranchTypeToAuxiliaryBranchName(branchType).c_str(), &pAux, bufSize, 0)),
      branches_(),
      metaBranches_(),
      basketSize_(bufSize),
      splitLevel_(splitLevel),
      branchNames_() {
      branches_.push_back(auxBranch_);
    }
    ~RootOutputTree() {}
    
    static TTree * makeTree(TFile * filePtr,
			    std::string const& name,
			    int splitLevel,
			    TChain * chain,
			    Selections const& dropList = Selections());

    static void writeTTree(TTree *tree);

    bool isValid() const;

    void addBranch(BranchDescription const& prod, bool selected, BranchEntryDescription const*& pProv, void const*& pProd);

    std::vector<std::string> const& branchNames() const {return branchNames_;}

    void fillTree() const;

    void writeTree() const;

    TTree *const tree() const {
      return tree_;
    }

    bool const& fastCloning() const {return fastCloning_;}

  private:
    static void fillTTree(TTree *tree, std::vector<TBranch *> const& branches);
    static void fillHelper(TBranch * br) {br->Fill();}
// We use bare pointers for pointers to some ROOT entities.
// Root owns them and uses bare pointers internally.
// Therefore,using smart pointers here will do no good.
    bool fastCloning_;
    boost::shared_ptr<TFile> filePtr_;
    TTree *const tree_;
    TTree *const metaTree_;
    TBranch *const auxBranch_;
    std::vector<TBranch *> branches_; // does not include cloned branches
    std::vector<TBranch *> metaBranches_; // does not include cloned branches
    int basketSize_;
    int splitLevel_;
    std::vector<std::string> branchNames_;
  };
}
#endif
