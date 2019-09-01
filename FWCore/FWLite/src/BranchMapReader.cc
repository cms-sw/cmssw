// -*- C++ -*-
//
// Package:     FWLite
// Class  :     BranchMapReader
//
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Dan Riley
//         Created:  Tue May 20 10:31:32 EDT 2008
//

// system include files

// user include files
#include "FWCore/FWLite/interface/BranchMapReader.h"

#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/BranchListIndex.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/EventEntryInfo.h"
#include "DataFormats/Provenance/interface/EventSelectionID.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/History.h"
#include "DataFormats/Provenance/interface/ProductIDToBranchID.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/ThinnedAssociationsHelper.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "FWCore/Utilities/interface/propagate_const.h"

#include "TBranch.h"
#include "TFile.h"
#include "TTree.h"

#include <cassert>

namespace fwlite {
  namespace internal {

    static const edm::BranchDescription kDefaultBranchDescription;

    BMRStrategy::BMRStrategy(TFile* file, int fileVersion)
        : currentFile_(file),
          eventTree_(nullptr),
          luminosityBlockTree_(nullptr),
          runTree_(nullptr),
          eventEntry_(-1),
          luminosityBlockEntry_(-1),
          runEntry_(-1),
          fileVersion_(fileVersion) {
      // do in derived obects
      // updateFile(file);
    }

    BMRStrategy::~BMRStrategy() {}

    class Strategy : public BMRStrategy {
    public:
      typedef std::map<edm::BranchID, edm::BranchDescription> bidToDesc;

      Strategy(TFile* file, int fileVersion);
      ~Strategy() override;
      bool updateFile(TFile* file) override;
      bool updateEvent(Long_t eventEntry) override {
        eventEntry_ = eventEntry;
        return true;
      }
      bool updateLuminosityBlock(Long_t luminosityBlockEntry) override {
        luminosityBlockEntry_ = luminosityBlockEntry;
        return true;
      }
      bool updateRun(Long_t runEntry) override {
        runEntry_ = runEntry;
        return true;
      }
      bool updateMap() override { return true; }
      edm::BranchID productToBranchID(edm::ProductID const& pid) override;
      edm::BranchDescription const& productToBranch(edm::ProductID const& pid) override;
      edm::BranchDescription const& branchIDToBranch(edm::BranchID const& bid) const override;
      std::vector<edm::BranchDescription> const& getBranchDescriptions() override;
      edm::ThinnedAssociationsHelper const& thinnedAssociationsHelper() const override {
        return *thinnedAssociationsHelper_;
      }

      TBranch* getBranchRegistry(edm::ProductRegistry** pReg);

      bidToDesc branchDescriptionMap_;
      std::vector<edm::BranchDescription> bDesc_;
      bool mapperFilled_;
      edm::propagate_const<std::unique_ptr<edm::ThinnedAssociationsHelper>> thinnedAssociationsHelper_;
    };

    Strategy::Strategy(TFile* file, int fileVersion)
        : BMRStrategy(file, fileVersion),
          mapperFilled_(false),
          thinnedAssociationsHelper_(new edm::ThinnedAssociationsHelper) {
      // do in derived obects
      // updateFile(file);
    }

    Strategy::~Strategy() {
      // probably need to clean up something here...
    }

    bool Strategy::updateFile(TFile* file) {
      currentFile_ = file;
      eventTree_ = dynamic_cast<TTree*>(currentFile_->Get(edm::poolNames::eventTreeName().c_str()));
      luminosityBlockTree_ = dynamic_cast<TTree*>(currentFile_->Get(edm::poolNames::luminosityBlockTreeName().c_str()));
      runTree_ = dynamic_cast<TTree*>(currentFile_->Get(edm::poolNames::runTreeName().c_str()));
      fileUUID_ = currentFile_->GetUUID();
      branchDescriptionMap_.clear();
      bDesc_.clear();
      return nullptr != eventTree_;
    }

    TBranch* Strategy::getBranchRegistry(edm::ProductRegistry** ppReg) {
      TBranch* bReg(nullptr);

      TTree* metaDataTree = dynamic_cast<TTree*>(currentFile_->Get(edm::poolNames::metaDataTreeName().c_str()));
      if (nullptr != metaDataTree) {
        bReg = metaDataTree->GetBranch(edm::poolNames::productDescriptionBranchName().c_str());
        bReg->SetAddress(ppReg);
        bReg->GetEntry(0);
      }
      return bReg;
    }

    std::vector<edm::BranchDescription> const& Strategy::getBranchDescriptions() {
      if (bDesc_.empty()) {
        for (auto const& item : branchDescriptionMap_) {
          bDesc_.push_back(item.second);
        }
      }
      return bDesc_;
    }

    edm::BranchID Strategy::productToBranchID(edm::ProductID const&) {
      throw edm::Exception(edm::errors::UnimplementedFeature) << "Unsupported EDM file version";
    }

    edm::BranchDescription const& Strategy::productToBranch(edm::ProductID const& pid) {
      edm::BranchID bid = productToBranchID(pid);
      bidToDesc::const_iterator bdi = branchDescriptionMap_.find(bid);
      if (branchDescriptionMap_.end() == bdi) {
        return kDefaultBranchDescription;
      }
      return bdi->second;
    }

    edm::BranchDescription const& Strategy::branchIDToBranch(edm::BranchID const& bid) const {
      bidToDesc::const_iterator bdi = branchDescriptionMap_.find(bid);
      if (branchDescriptionMap_.end() == bdi) {
        return kDefaultBranchDescription;
      }
      return bdi->second;
    }

    class BranchMapReaderStrategyV1 : public Strategy {
    public:
      BranchMapReaderStrategyV1(TFile* file, int fileVersion);
      bool updateFile(TFile* file) override;
      bool updateMap() override;
      edm::BranchListIndexes const& branchListIndexes() const override { return dummyBranchListIndexes_; }

    private:
      edm::BranchListIndexes dummyBranchListIndexes_;
    };

    BranchMapReaderStrategyV1::BranchMapReaderStrategyV1(TFile* file, int fileVersion) : Strategy(file, fileVersion) {
      updateFile(file);
    }

    bool BranchMapReaderStrategyV1::updateFile(TFile* file) {
      if (Strategy::updateFile(file)) {
        mapperFilled_ = false;
        return true;
      }
      return false;
    }

    bool BranchMapReaderStrategyV1::updateMap() {
      if (mapperFilled_) {
        return true;
      }

      branchDescriptionMap_.clear();
      bDesc_.clear();

      edm::ProductRegistry reg;
      edm::ProductRegistry* pReg = &reg;
      TBranch* br = getBranchRegistry(&pReg);

      if (nullptr != br) {
        edm::ProductRegistry::ProductList& prodList = reg.productListUpdator();

        for (auto& item : prodList) {
          edm::BranchDescription& prod = item.second;
          if (edm::InEvent == prod.branchType()) {
            // call to regenerate branchName
            prod.init();
            branchDescriptionMap_.insert(bidToDesc::value_type(prod.branchID(), prod));
          }
        }
        mapperFilled_ = true;
      }
      reg.setFrozen(false);
      return nullptr != br;
    }

    // v7 has differences in product status that are not implemented in BranchMapReader yet
    class BranchMapReaderStrategyV7 : public BranchMapReaderStrategyV1 {
    public:
      BranchMapReaderStrategyV7(TFile* file, int fileVersion);
      edm::BranchListIndexes const& branchListIndexes() const override { return dummyBranchListIndexes_; }

    private:
      edm::BranchListIndexes dummyBranchListIndexes_;
    };

    BranchMapReaderStrategyV7::BranchMapReaderStrategyV7(TFile* file, int fileVersion)
        : BranchMapReaderStrategyV1(file, fileVersion) {
      updateFile(file);
    }

    class BranchMapReaderStrategyV8 : public Strategy {
    public:
      BranchMapReaderStrategyV8(TFile* file, int fileVersion);
      bool updateFile(TFile* file) override;
      bool updateEvent(Long_t eventEntry) override;
      bool updateLuminosityBlock(Long_t luminosityBlockEntry) override;
      bool updateRun(Long_t runEntry) override;
      bool updateMap() override;
      edm::BranchListIndexes const& branchListIndexes() const override { return dummyBranchListIndexes_; }

    private:
      edm::propagate_const<TBranch*> entryInfoBranch_;
      edm::EventEntryInfoVector eventEntryInfoVector_;
      edm::EventEntryInfoVector const* pEventEntryInfoVector_;
      edm::BranchListIndexes dummyBranchListIndexes_;
    };

    BranchMapReaderStrategyV8::BranchMapReaderStrategyV8(TFile* file, int fileVersion)
        : Strategy(file, fileVersion), eventEntryInfoVector_(), pEventEntryInfoVector_(&eventEntryInfoVector_) {
      updateFile(file);
    }

    bool BranchMapReaderStrategyV8::updateEvent(Long_t newevent) {
      // std::cout << "v8 updateevent " << newevent << std::endl;
      if (newevent != eventEntry_) {
        eventEntry_ = newevent;
        mapperFilled_ = false;
      }
      return true;
    }

    bool BranchMapReaderStrategyV8::updateLuminosityBlock(Long_t newLumi) {
      if (newLumi != luminosityBlockEntry_) {
        luminosityBlockEntry_ = newLumi;
        mapperFilled_ = false;
      }
      return true;
    }

    bool BranchMapReaderStrategyV8::updateRun(Long_t newRun) {
      if (newRun != runEntry_) {
        runEntry_ = newRun;
        mapperFilled_ = false;
      }
      return true;
    }

    bool BranchMapReaderStrategyV8::updateFile(TFile* file) {
      Strategy::updateFile(file);
      mapperFilled_ = false;
      entryInfoBranch_ = nullptr;
      TTree* metaDataTree = dynamic_cast<TTree*>(currentFile_->Get(edm::poolNames::eventMetaDataTreeName().c_str()));
      if (nullptr != metaDataTree) {
        entryInfoBranch_ = metaDataTree->GetBranch(BranchTypeToBranchEntryInfoBranchName(edm::InEvent).c_str());
        //         std::cout << "entryInfoBranch for " << BranchTypeToBranchEntryInfoBranchName(edm::InEvent) << " " << entryInfoBranch_ << std::endl;
      } else {
        return false;
      }
      pEventEntryInfoVector_ = &eventEntryInfoVector_;
      entryInfoBranch_->SetAddress(&pEventEntryInfoVector_);

      branchDescriptionMap_.clear();
      bDesc_.clear();

      edm::ProductRegistry reg;
      edm::ProductRegistry* pReg = &reg;
      TBranch* br = getBranchRegistry(&pReg);

      if (nullptr != br) {
        edm::ProductRegistry::ProductList& prodList = reg.productListUpdator();

        for (auto& item : prodList) {
          edm::BranchDescription& prod = item.second;
          if (edm::InEvent == prod.branchType()) {
            // call to regenerate branchName
            prod.init();
            branchDescriptionMap_.insert(bidToDesc::value_type(prod.branchID(), prod));
          }
        }
      }
      reg.setFrozen(false);
      return nullptr != br;
    }

    bool BranchMapReaderStrategyV8::updateMap() {
      if (mapperFilled_) {
        return true;
      }

      assert(entryInfoBranch_);

      entryInfoBranch_->GetEntry(eventEntry_);

      //      for(auto const& item : *pEventEntryInfoVector_) {
      //         eventInfoMap_.insert(item);
      //      }
      mapperFilled_ = true;
      return true;
    }

    class BranchMapReaderStrategyV11 : public Strategy {
    public:
      BranchMapReaderStrategyV11(TFile* file, int fileVersion);
      bool updateFile(TFile* file) override;
      bool updateEvent(Long_t eventEntry) override;
      bool updateLuminosityBlock(Long_t luminosityBlockEntry) override;
      bool updateRun(Long_t runEntry) override;
      bool updateMap() override;
      edm::BranchID productToBranchID(edm::ProductID const& pid) override;
      edm::BranchListIndexes const& branchListIndexes() const override { return history_.branchListIndexes(); }

    private:
      edm::propagate_const<std::unique_ptr<edm::BranchIDLists>> branchIDLists_;
      edm::propagate_const<TTree*> eventHistoryTree_;
      edm::History history_;
      edm::History const* pHistory_;
    };

    BranchMapReaderStrategyV11::BranchMapReaderStrategyV11(TFile* file, int fileVersion)
        : Strategy(file, fileVersion), eventHistoryTree_(nullptr), pHistory_(&history_) {
      updateFile(file);
    }

    bool BranchMapReaderStrategyV11::updateEvent(Long_t newevent) {
      //      std::cout << "v11 updateevent " << newevent << std::endl;
      if (newevent != eventEntry_) {
        eventEntry_ = newevent;
        mapperFilled_ = false;
      }
      return true;
    }

    bool BranchMapReaderStrategyV11::updateLuminosityBlock(Long_t newlumi) {
      if (newlumi != luminosityBlockEntry_) {
        luminosityBlockEntry_ = newlumi;
        mapperFilled_ = false;
      }
      return true;
    }

    bool BranchMapReaderStrategyV11::updateRun(Long_t newRun) {
      if (newRun != runEntry_) {
        runEntry_ = newRun;
        mapperFilled_ = false;
      }
      return true;
    }

    bool BranchMapReaderStrategyV11::updateFile(TFile* file) {
      Strategy::updateFile(file);
      mapperFilled_ = false;
      TTree* metaDataTree = dynamic_cast<TTree*>(currentFile_->Get(edm::poolNames::metaDataTreeName().c_str()));

      if (nullptr == metaDataTree) {
        throw edm::Exception(edm::errors::EventCorruption)
            << "No " << edm::poolNames::metaDataTreeName() << " TTree in file";
      }
      branchIDLists_ = std::make_unique<edm::BranchIDLists>();
      edm::BranchIDLists* branchIDListsPtr = branchIDLists_.get();
      if (metaDataTree->FindBranch(edm::poolNames::branchIDListBranchName().c_str()) != nullptr) {
        TBranch* b = metaDataTree->GetBranch(edm::poolNames::branchIDListBranchName().c_str());
        b->SetAddress(&branchIDListsPtr);
        b->GetEntry(0);
        //         std::cout << "--> " << branchIDLists_->size() << std::endl;
      } else {
        throw edm::Exception(edm::errors::EventCorruption) << "FindBranch of branchIDList failed";
        return false;
      }

      eventHistoryTree_ = dynamic_cast<TTree*>(currentFile_->Get(edm::poolNames::eventHistoryTreeName().c_str()));

      branchDescriptionMap_.clear();
      bDesc_.clear();

      edm::ProductRegistry reg;
      edm::ProductRegistry* pReg = &reg;
      TBranch* br = getBranchRegistry(&pReg);

      if (nullptr != br) {
        edm::ProductRegistry::ProductList& prodList = reg.productListUpdator();

        for (auto& item : prodList) {
          edm::BranchDescription& prod = item.second;
          if (edm::InEvent == prod.branchType()) {
            // call to regenerate branchName
            prod.init();
            branchDescriptionMap_.insert(bidToDesc::value_type(prod.branchID(), prod));
            //             std::cout << "v11 updatefile " << prod.branchID() << std::endl;
          }
        }
      }
      reg.setFrozen(false);
      return nullptr != br;
    }

    bool BranchMapReaderStrategyV11::updateMap() {
      if (!mapperFilled_) {
        TBranch* eventHistoryBranch = eventHistoryTree_->GetBranch(edm::poolNames::eventHistoryBranchName().c_str());
        if (!eventHistoryBranch) {
          throw edm::Exception(edm::errors::EventCorruption) << "Failed to find history branch in event history tree";
          return false;
        }
        // yes, SetAddress really does need to be called every time...
        eventHistoryBranch->SetAddress(&pHistory_);
        eventHistoryTree_->GetEntry(eventEntry_);
        mapperFilled_ = true;
      }
      return true;
    }

    edm::BranchID BranchMapReaderStrategyV11::productToBranchID(edm::ProductID const& pid) {
      updateMap();
      return edm::productIDToBranchID(pid, *branchIDLists_, history_.branchListIndexes());
    }

    class BranchMapReaderStrategyV17 : public Strategy {
    public:
      BranchMapReaderStrategyV17(TFile* file, int fileVersion);
      bool updateFile(TFile* file) override;
      bool updateEvent(Long_t eventEntry) override;
      bool updateLuminosityBlock(Long_t luminosityBlockEntry) override;
      bool updateRun(Long_t runEntry) override;
      bool updateMap() override;
      edm::BranchID productToBranchID(edm::ProductID const& pid) override;
      edm::BranchListIndexes const& branchListIndexes() const override { return branchListIndexes_; }

    private:
      edm::propagate_const<std::unique_ptr<edm::BranchIDLists>> branchIDLists_;
      edm::propagate_const<TTree*> eventsTree_;
      edm::BranchListIndexes branchListIndexes_;
      edm::propagate_const<edm::BranchListIndexes*> pBranchListIndexes_;
    };

    BranchMapReaderStrategyV17::BranchMapReaderStrategyV17(TFile* file, int fileVersion)
        : Strategy(file, fileVersion), eventsTree_(nullptr), pBranchListIndexes_(&branchListIndexes_) {
      updateFile(file);
    }

    bool BranchMapReaderStrategyV17::updateEvent(Long_t newevent) {
      //      std::cout << "v11 updateevent " << newevent << std::endl;
      if (newevent != eventEntry_) {
        eventEntry_ = newevent;
        mapperFilled_ = false;
      }
      return true;
    }

    bool BranchMapReaderStrategyV17::updateLuminosityBlock(Long_t newlumi) {
      if (newlumi != luminosityBlockEntry_) {
        luminosityBlockEntry_ = newlumi;
        mapperFilled_ = false;
      }
      return true;
    }

    bool BranchMapReaderStrategyV17::updateRun(Long_t newRun) {
      if (newRun != runEntry_) {
        runEntry_ = newRun;
        mapperFilled_ = false;
      }
      return true;
    }

    bool BranchMapReaderStrategyV17::updateFile(TFile* file) {
      Strategy::updateFile(file);
      mapperFilled_ = false;
      TTree* metaDataTree = dynamic_cast<TTree*>(currentFile_->Get(edm::poolNames::metaDataTreeName().c_str()));
      if (nullptr == metaDataTree) {
        throw edm::Exception(edm::errors::EventCorruption)
            << "No " << edm::poolNames::metaDataTreeName() << " TTree in file";
      }

      thinnedAssociationsHelper_ = std::make_unique<edm::ThinnedAssociationsHelper>();
      edm::ThinnedAssociationsHelper* thinnedAssociationsHelperPtr = thinnedAssociationsHelper_.get();
      if (metaDataTree->FindBranch(edm::poolNames::thinnedAssociationsHelperBranchName().c_str()) != nullptr) {
        TBranch* b = metaDataTree->GetBranch(edm::poolNames::thinnedAssociationsHelperBranchName().c_str());
        b->SetAddress(&thinnedAssociationsHelperPtr);
        b->GetEntry(0);
      }

      branchIDLists_ = std::make_unique<edm::BranchIDLists>();
      edm::BranchIDLists* branchIDListsPtr = branchIDLists_.get();
      if (metaDataTree->FindBranch(edm::poolNames::branchIDListBranchName().c_str()) != nullptr) {
        TBranch* b = metaDataTree->GetBranch(edm::poolNames::branchIDListBranchName().c_str());
        b->SetAddress(&branchIDListsPtr);
        b->GetEntry(0);
        //         std::cout << "--> " << branchIDLists_->size() << std::endl;
      } else {
        throw edm::Exception(edm::errors::EventCorruption) << "FindBranch of branchIDList failed";
        return false;
      }

      eventsTree_ = dynamic_cast<TTree*>(currentFile_->Get(edm::poolNames::eventTreeName().c_str()));

      branchDescriptionMap_.clear();
      bDesc_.clear();

      edm::ProductRegistry reg;
      edm::ProductRegistry* pReg = &reg;
      TBranch* br = getBranchRegistry(&pReg);

      if (nullptr != br) {
        edm::ProductRegistry::ProductList& prodList = reg.productListUpdator();

        for (auto& item : prodList) {
          edm::BranchDescription& prod = item.second;
          if (edm::InEvent == prod.branchType()) {
            // call to regenerate branchName
            prod.init();
            branchDescriptionMap_.insert(bidToDesc::value_type(prod.branchID(), prod));
            //             std::cout << "v11 updatefile " << prod.branchID() << std::endl;
          }
        }
      }
      reg.setFrozen(false);
      return nullptr != br;
    }

    bool BranchMapReaderStrategyV17::updateMap() {
      if (!mapperFilled_) {
        TBranch* branchListIndexesBranch =
            eventsTree_->GetBranch(edm::poolNames::branchListIndexesBranchName().c_str());
        if (!branchListIndexesBranch) {
          throw edm::Exception(edm::errors::EventCorruption)
              << "Failed to find branch list indexes branch in event tree";
          return false;
        }
        // yes, SetAddress really does need to be called every time...
        branchListIndexesBranch->SetAddress(&pBranchListIndexes_);
        branchListIndexesBranch->GetEntry(eventEntry_);
        mapperFilled_ = true;
      }
      return true;
    }

    edm::BranchID BranchMapReaderStrategyV17::productToBranchID(edm::ProductID const& pid) {
      updateMap();
      return edm::productIDToBranchID(pid, *branchIDLists_, branchListIndexes_);
    }
  }  // namespace internal

  //
  // constants, enums and typedefs
  //

  //
  // static data member definitions
  //

  //
  // constructors and destructor
  //

  BranchMapReader::BranchMapReader(TFile* file) : fileVersion_(-1) {
    if (nullptr == file) {
      throw cms::Exception("NoFile") << "The TFile pointer is null";
    }
    strategy_ = newStrategy(file, getFileVersion(file));
  }

  //
  // member functions
  //

  int BranchMapReader::getFileVersion(TFile* file) {
    TTree* metaDataTree = dynamic_cast<TTree*>(file->Get(edm::poolNames::metaDataTreeName().c_str()));
    if (nullptr == metaDataTree) {
      return 0;
    }

    edm::FileFormatVersion v;
    edm::FileFormatVersion* pV = &v;
    TBranch* bVer = metaDataTree->GetBranch(edm::poolNames::fileFormatVersionBranchName().c_str());
    bVer->SetAddress(&pV);
    bVer->GetEntry(0);
    fileVersion_ = v.value();
    return v.value();
  }

  bool BranchMapReader::updateEvent(Long_t newevent) { return strategy_->updateEvent(newevent); }

  bool BranchMapReader::updateLuminosityBlock(Long_t newlumi) { return strategy_->updateLuminosityBlock(newlumi); }

  bool BranchMapReader::updateRun(Long_t newRun) { return strategy_->updateRun(newRun); }

  bool BranchMapReader::updateFile(TFile* file) {
    if (nullptr == strategy_.get()) {
      strategy_ = newStrategy(file, getFileVersion(file));
      return true;
    }

    TFile* currentFile(strategy_->currentFile_);
    bool isNew(file != currentFile);

    if (!isNew) {
      //could still have a new TFile which just happens to share the same memory address as the previous file
      //will assume that if the Event tree's address and UUID are the same as before then we do not have
      // to treat this like a new file
      TTree* eventTreeTemp = dynamic_cast<TTree*>(currentFile->Get(edm::poolNames::eventTreeName().c_str()));
      isNew = eventTreeTemp != strategy_->eventTree_ || strategy_->fileUUID_ != currentFile->GetUUID();
    }
    if (isNew) {
      int fileVersion = getFileVersion(file);
      if (fileVersion != strategy_->fileVersion_) {
        strategy_ = newStrategy(file, fileVersion);
      } else {
        strategy_->updateFile(file);
      }
    }
    return isNew;
  }

  edm::BranchDescription const& BranchMapReader::productToBranch(edm::ProductID const& pid) {
    return strategy_->productToBranch(pid);
  }

  std::vector<edm::BranchDescription> const& BranchMapReader::getBranchDescriptions() {
    return strategy_->getBranchDescriptions();
  }

  std::unique_ptr<internal::BMRStrategy> BranchMapReader::newStrategy(TFile* file, int fileVersion) {
    std::unique_ptr<internal::BMRStrategy> s;

    if (fileVersion >= 17) {
      s = std::make_unique<internal::BranchMapReaderStrategyV17>(file, fileVersion);
    } else if (fileVersion >= 11) {
      s = std::make_unique<internal::BranchMapReaderStrategyV11>(file, fileVersion);
    } else if (fileVersion >= 8) {
      s = std::make_unique<internal::BranchMapReaderStrategyV8>(file, fileVersion);
    } else if (fileVersion >= 7) {
      s = std::make_unique<internal::BranchMapReaderStrategyV7>(file, fileVersion);
    } else {
      s = std::make_unique<internal::BranchMapReaderStrategyV1>(file, fileVersion);
    }
    return s;
  }
}  // namespace fwlite
