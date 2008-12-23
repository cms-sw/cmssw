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
// $Id: BranchMapReader.h,v 1.7 2008/12/22 18:06:07 dsr Exp $
//

// system include files
#include <memory>
#include "TUUID.h"

// user include files
#include "DataFormats/Provenance/interface/BranchDescription.h"

// forward declarations
class TFile;
class TTree;
class TBranch;

namespace fwlite {
  namespace internal {
    class BMRStrategy {
    public:
      BMRStrategy(TFile* file, int fileVersion);
      virtual ~BMRStrategy();

      virtual bool updateFile(TFile* file) = 0;
      virtual bool updateEvent(Long_t eventEntry) = 0;
      virtual bool updateMap() = 0;
      virtual edm::BranchID productToBranchID(const edm::ProductID& pid) = 0;
      virtual const edm::BranchDescription productToBranch(const edm::ProductID& pid) = 0;
      virtual const std::vector<edm::BranchDescription>& getBranchDescriptions() = 0;

      TFile* currentFile_;
      TTree* eventTree_;
      TUUID fileUUID_;
      Long_t eventEntry_;
      int fileVersion_;
    };
  }

  class BranchMapReader {
  public:
    BranchMapReader(TFile* file);
    BranchMapReader() : strategy_(0) {}

      // ---------- const member functions ---------------------
      
      // ---------- static member functions --------------------

      // ---------- member functions ---------------------------
    bool updateFile(TFile* file);
    bool updateEvent(Long_t eventEntry);
    const edm::BranchDescription productToBranch(const edm::ProductID& pid);
    int getFileVersion(TFile* file) const;

    TFile* getFile() const { return strategy_->currentFile_; }
    TTree* getEventTree() const { return strategy_->eventTree_; }
    TUUID getFileUUID() const { return strategy_->fileUUID_; }
    Long_t getEventEntry() const { return strategy_->eventEntry_; }
    const std::vector<edm::BranchDescription>& getBranchDescriptions();

      // ---------- member data --------------------------------
  private:
    std::auto_ptr<internal::BMRStrategy> newStrategy(TFile* file, int fileVersion);
    std::auto_ptr<internal::BMRStrategy> strategy_;
  };
}

#endif
