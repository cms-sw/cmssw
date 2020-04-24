#ifndef FWLite_BranchMapReader_h
#define FWLite_BranchMapReader_h
// -*- C++ -*-
//
// Package:     FWLite
// Class  :     BranchMapReader
//
/**\class BranchMapReader BranchMapReader.h FWCore/FWLite/interface/BranchMapReader.h

 Description: <one line class summary>

 Usage:
    <usage>

*/
//
// Original Author:  Dan Riley
//         Created:  Tue May 20 10:31:32 EDT 2008
//

// system include files
#include <memory>
#include "TUUID.h"

// user include files
#include "DataFormats/Provenance/interface/BranchDescription.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "FWCore/Utilities/interface/propagate_const.h"

// forward declarations
class TFile;
class TTree;
class TBranch;

namespace edm {
  class ThinnedAssociationsHelper;
}

namespace fwlite {
  namespace internal {
    class BMRStrategy {
    public:
      BMRStrategy(TFile* file, int fileVersion);
      virtual ~BMRStrategy();

      virtual bool updateFile(TFile* file) = 0;
      virtual bool updateEvent(Long_t eventEntry) = 0;
      virtual bool updateLuminosityBlock(Long_t luminosityBlockEntry) = 0;
      virtual bool updateRun(Long_t runEntry) = 0;
      virtual bool updateMap() = 0;
      virtual edm::BranchID productToBranchID(const edm::ProductID& pid) = 0;
      virtual const edm::BranchDescription& productToBranch(const edm::ProductID& pid) = 0;
      virtual const edm::BranchDescription& branchIDToBranch(const edm::BranchID& bid) const = 0;
      virtual const std::vector<edm::BranchDescription>& getBranchDescriptions() = 0;
      virtual const edm::BranchListIndexes& branchListIndexes() const = 0;
      virtual const edm::ThinnedAssociationsHelper& thinnedAssociationsHelper() const = 0;

      edm::propagate_const<TFile*> currentFile_;
      edm::propagate_const<TTree*> eventTree_;
      edm::propagate_const<TTree*> luminosityBlockTree_;
      edm::propagate_const<TTree*> runTree_;
      TUUID fileUUID_;
      Long_t eventEntry_;
      Long_t luminosityBlockEntry_;
      Long_t runEntry_;
      int fileVersion_;
    };
  }

  class BranchMapReader {
  public:
    BranchMapReader(TFile* file);
    BranchMapReader() : strategy_(nullptr),fileVersion_(0) {}

      // ---------- const member functions ---------------------

      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
    bool updateFile(TFile* file);
    bool updateEvent(Long_t eventEntry);
    bool updateLuminosityBlock(Long_t luminosityBlockEntry);
    bool updateRun(Long_t runEntry);
    edm::BranchID productToBranchID(const edm::ProductID& pid) { return strategy_->productToBranchID(pid); }
    const edm::BranchDescription& productToBranch(const edm::ProductID& pid);
    const edm::BranchDescription& branchIDToBranch(const edm::BranchID& bid) const { return strategy_->branchIDToBranch(bid); }
    int getFileVersion(TFile* file);
    int getFileVersion() const { return  fileVersion_;}

    TFile const* getFile() const { return strategy_->currentFile_; }
    TFile* getFile() { return strategy_->currentFile_; }
    TTree const* getEventTree() const { return strategy_->eventTree_; }
    TTree* getEventTree() { return strategy_->eventTree_; }
    TTree const* getLuminosityBlockTree() const { return strategy_->luminosityBlockTree_; }
    TTree* getLuminosityBlockTree() { return strategy_->luminosityBlockTree_; }
    TTree const* getRunTree() const { return strategy_->runTree_; }
    TTree* getRunTree() { return strategy_->runTree_; }
    TUUID getFileUUID() const { return strategy_->fileUUID_; }
    Long_t getEventEntry() const { return strategy_->eventEntry_; }
    Long_t getLuminosityBlockEntry() const { return strategy_->luminosityBlockEntry_; }
    Long_t getRunEntry() const { return strategy_->runEntry_; }
    const std::vector<edm::BranchDescription>& getBranchDescriptions();
    const edm::BranchListIndexes& branchListIndexes() const { strategy_->updateMap(); return strategy_->branchListIndexes(); }
    const edm::ThinnedAssociationsHelper& thinnedAssociationsHelper() const { return strategy_->thinnedAssociationsHelper(); }

      // ---------- member data --------------------------------
  private:
    std::unique_ptr<internal::BMRStrategy> newStrategy(TFile* file, int fileVersion);
    std::unique_ptr<internal::BMRStrategy> strategy_; // Contains caches, so we do not propagate_const
    int fileVersion_;
  };
}

#endif
