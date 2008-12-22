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
// $Id: BranchMapReader.cc,v 1.5 2008/12/18 06:19:37 wmtan Exp $
//

// system include files

#include "TFile.h"
#include "TTree.h"
#include "TBranch.h"
#include "TClass.h"
#include "Reflex/Type.h"
#include "TROOT.h"

// user include files
#include "FWCore/FWLite/interface/BranchMapReader.h"
#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Provenance/interface/ProductStatus.h"
#include "FWCore/Utilities/interface/WrappedClassName.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Provenance/interface/BranchIDList.h"
#include "DataFormats/Provenance/interface/ProductIDToBranchID.h"

namespace fwlite {
  namespace internal {
    class BranchMapReaderStrategyV1 : public fwlite::BranchMapReader::Strategy {
    public:
      BranchMapReaderStrategyV1(TFile* file, int fileVersion, BranchMapReader::bidToDesc& branchDescriptionMap);
      virtual bool updateFile(TFile* file);
      virtual bool updateMap();
    private:
    };

    BranchMapReaderStrategyV1::BranchMapReaderStrategyV1(TFile* file, int fileVersion,
      BranchMapReader::bidToDesc& branchDescriptionMap)
      : Strategy(file, fileVersion, branchDescriptionMap)
    {
      updateFile(file);
    }

    bool BranchMapReaderStrategyV1::updateFile(TFile* file)
    {
      if (BranchMapReader::Strategy::updateFile(file)) {
        mapperFilled_ = false;
        return true;        
      }
      return false;
    }
    
    bool BranchMapReaderStrategyV1::updateMap()
    {
      if (mapperFilled_) {
        return true;
      }

      branchDescriptionMap_.clear();
      
      edm::ProductRegistry reg;
      edm::ProductRegistry* pReg = &reg;
      TBranch* br = getBranchRegistry(&pReg);

      if (0 != br) {
        const edm::ProductRegistry::ProductList& prodList = reg.productList();

        for(edm::ProductRegistry::ProductList::const_iterator it = prodList.begin(), itEnd = prodList.end(); it != itEnd; ++it) {
          if (edm::InEvent == it->second.branchType()) {
//            edm::ProductStatus status = edm::productstatus::uninitialized();
            // call to regenerate branchName
            it->second.init();
//  	        edm::EventEntryInfo entry(it->second.branchID(), status, it->second.oldProductID());
//  	        eventInfoMap_.insert(entry);
            branchDescriptionMap_.insert(BranchMapReader::bidToDesc::value_type(it->second.branchID(), it->second));
            // std::cout << "v1 updatemap " << it->second.branchID() << std::endl;
	        }
	      }
        mapperFilled_ = true;
      }
      return 0 != br;
    }

    // v7 has differences in product status that are not implemented in BranchMapReader yet
    class BranchMapReaderStrategyV7 : public BranchMapReaderStrategyV1 {
    public:
      BranchMapReaderStrategyV7(TFile* file, int fileVersion, BranchMapReader::bidToDesc& branchDescriptionMap);
    };

    BranchMapReaderStrategyV7::BranchMapReaderStrategyV7(TFile* file, int fileVersion, BranchMapReader::bidToDesc& branchDescriptionMap)
    : BranchMapReaderStrategyV1(file, fileVersion, branchDescriptionMap)
    {
      updateFile(file);
    }

    class BranchMapReaderStrategyV8 : public fwlite::BranchMapReader::Strategy {
    public:
      BranchMapReaderStrategyV8(TFile* file, int fileVersion, BranchMapReader::bidToDesc& branchDescriptionMap);
      virtual bool updateFile(TFile* file);
      virtual bool updateEvent(Long_t eventEntry);
      virtual bool updateMap();
    private:
      TBranch* entryInfoBranch_;
      edm::EventEntryInfoVector  eventEntryInfoVector_;
      edm::EventEntryInfoVector* pEventEntryInfoVector_;
    };

    BranchMapReaderStrategyV8::BranchMapReaderStrategyV8(TFile* file, int fileVersion,
      BranchMapReader::bidToDesc& branchDescriptionMap)
    : Strategy(file, fileVersion, branchDescriptionMap),
      eventEntryInfoVector_(), pEventEntryInfoVector_(&eventEntryInfoVector_)
    {
      updateFile(file);
    }

    bool BranchMapReaderStrategyV8::updateEvent(Long_t newevent)
    {
      // std::cout << "v8 updateevent " << newevent << std::endl;
      if (newevent != eventEntry_) {
        eventEntry_ = newevent;
        mapperFilled_ = false;
      }
      return true;
    }

    bool BranchMapReaderStrategyV8::updateFile(TFile* file)
    {
      BranchMapReader::Strategy::updateFile(file);
      mapperFilled_ = false;
      entryInfoBranch_ = 0;
      TTree* metaDataTree = dynamic_cast<TTree*>(currentFile_->Get(edm::poolNames::eventMetaDataTreeName().c_str()) );
      // std::cout << "metaDataTree " << metaDataTree << std::endl;
      if (0 != metaDataTree) {
        entryInfoBranch_ = metaDataTree->GetBranch(BranchTypeToBranchEntryInfoBranchName(edm::InEvent).c_str());
        // std::cout << "entryInfoBranch for " << BranchTypeToBranchEntryInfoBranchName(edm::InEvent) << " " << entryInfoBranch_ << std::endl;
      } else {
        return false;
      }
      pEventEntryInfoVector_ = &eventEntryInfoVector_;
      entryInfoBranch_->SetAddress(&pEventEntryInfoVector_);

      branchDescriptionMap_.clear();
      
      edm::ProductRegistry reg;
      edm::ProductRegistry* pReg = &reg;
      TBranch *br = getBranchRegistry(&pReg);

      if (0 != br) {
        const edm::ProductRegistry::ProductList& prodList = reg.productList();

        for(edm::ProductRegistry::ProductList::const_iterator it = prodList.begin(), itEnd = prodList.end(); it != itEnd; ++it) {
          if (edm::InEvent == it->second.branchType()) {
            // call to regenerate branchName
            it->second.init();
            branchDescriptionMap_.insert(BranchMapReader::bidToDesc::value_type(it->second.branchID(), it->second));
            // std::cout << "v8 updatefile " << it->second.branchID() << std::endl;
	        }
	      }
      }
      return 0 != br;
    }

    bool BranchMapReaderStrategyV8::updateMap()
    {
      if (mapperFilled_) {
        return true;
      }

      assert (entryInfoBranch_);

      entryInfoBranch_->GetEntry(eventEntry_);

      for (std::vector<edm::EventEntryInfo>::const_iterator it = pEventEntryInfoVector_->begin(), 
           itEnd = pEventEntryInfoVector_->end();
           it != itEnd; ++it) {
//         eventInfoMap_.insert(*it);
        // std::cout << "v8 updatemap " << it->productID() << " " << it->branchID() << std::endl;
      }
      mapperFilled_ = true;
      return true;
    }

    class BranchMapReaderStrategyV11 : public fwlite::BranchMapReader::Strategy {
    public:
      BranchMapReaderStrategyV11(TFile* file, int fileVersion, BranchMapReader::bidToDesc& branchDescriptionMap);
      virtual bool updateFile(TFile* file);
      virtual bool updateEvent(Long_t eventEntry);
      virtual bool updateMap();
      virtual edm::BranchID productToBranchID(const edm::ProductID& pid);
    private:
      std::auto_ptr<edm::BranchIDLists> branchIDLists_;
    };

    BranchMapReaderStrategyV11::BranchMapReaderStrategyV11(TFile* file, int fileVersion, BranchMapReader::bidToDesc& branchDescriptionMap)
    : Strategy(file, fileVersion, branchDescriptionMap)
    {
      updateFile(file);
    }

    bool BranchMapReaderStrategyV11::updateEvent(Long_t newevent)
    {
      // std::cout << "v11 updateevent " << newevent << std::endl;
      if (newevent != eventEntry_) {
        eventEntry_ = newevent;
        mapperFilled_ = false;
      }
      return true;
    }

    bool BranchMapReaderStrategyV11::updateFile(TFile* file)
    {
      BranchMapReader::Strategy::updateFile(file);
      mapperFilled_ = false;
      TTree* metaDataTree = dynamic_cast<TTree*>(currentFile_->Get(edm::poolNames::metaDataTreeName().c_str()) );
      // std::cout << "metaDataTree " << metaDataTree << std::endl;

      branchIDLists_.reset(new edm::BranchIDLists);
      edm::BranchIDLists* branchIDListsPtr = branchIDLists_.get();
      if (metaDataTree->FindBranch(edm::poolNames::branchIDListBranchName().c_str()) != 0) {
        TBranch* b = metaDataTree->GetBranch(edm::poolNames::branchIDListBranchName().c_str());
        b->SetAddress(&branchIDListsPtr);
        b->GetEntry(0);
        // std::cout << "--> " << branchIDLists_->size() << std::endl;
      } else {
        // std::cout << "FindBranch of branchIDList failed" << std::endl;
        // throw??
        return false;
      }

      eventHistoryTree_ = dynamic_cast<TTree*>(currentFile_->Get(edm::poolNames::eventHistoryTreeName().c_str()));
      TBranch* eventHistoryBranch = eventHistoryTree_->GetBranch(edm::poolNames::eventHistoryBranchName().c_str());
      if (!eventHistoryBranch) {
        // std::cout << "GetBranch of eventHistoryBranch failed" << std::endl;
        return false;
//        throw edm::Exception(edm::errors::FatalRootError)
//          << "Failed to find history branch in event history tree";
      }
      eventHistoryBranch->SetAddress(&pHistory_);

      branchDescriptionMap_.clear();
      
      edm::ProductRegistry reg;
      edm::ProductRegistry* pReg = &reg;
      TBranch *br = getBranchRegistry(&pReg);

      if (0 != br) {
        const edm::ProductRegistry::ProductList& prodList = reg.productList();

        for(edm::ProductRegistry::ProductList::const_iterator it = prodList.begin(), itEnd = prodList.end(); it != itEnd; ++it) {
          if (edm::InEvent == it->second.branchType()) {
            // call to regenerate branchName
            it->second.init();
            branchDescriptionMap_.insert(BranchMapReader::bidToDesc::value_type(it->second.branchID(), it->second));
            // std::cout << "v11 updatefile " << it->second.branchID() << std::endl;
	        }
	      }
      }
      return 0 != br;
    }

    bool BranchMapReaderStrategyV11::updateMap()
    {
      if (!mapperFilled_) {
        // std::cout << "Calling GetEntry on eventHistoryTree" << std::endl;
        eventHistoryTree_->GetEntry(eventEntry_);
        mapperFilled_ = true;
      }
      return true;
    }

    edm::BranchID 
    BranchMapReaderStrategyV11::productToBranchID(const edm::ProductID& pid)
    {
      updateMap();
      return edm::productIDToBranchID(pid, *branchIDLists_, history_.branchListIndexes());
    }
  }

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//

BranchMapReader::BranchMapReader(TFile* file)
{
  strategy_ = newStrategy(file, getFileVersion(file));
}

BranchMapReader::Strategy::Strategy(TFile* file, int fileVersion, BranchMapReader::bidToDesc& branchDescriptionMap)
  : currentFile_(file), fileVersion_(fileVersion), eventEntry_(-1),
    branchDescriptionMap_(branchDescriptionMap), mapperFilled_(false), pHistory_(&history_)
{
  // do in derived obects
  // updateFile(file);
}

BranchMapReader::Strategy::~Strategy()
{
}

//
// member functions
//

bool BranchMapReader::Strategy::updateFile(TFile* file)
{
  currentFile_ = file;
  eventTree_ = dynamic_cast<TTree*>(currentFile_->Get(edm::poolNames::eventTreeName().c_str()));
  fileUUID_ = currentFile_->GetUUID();
  return 0 != eventTree_;
}

TBranch* BranchMapReader::Strategy::getBranchRegistry(edm::ProductRegistry** ppReg)
{
  TBranch* bReg(0);

  TTree* metaDataTree = dynamic_cast<TTree*>(currentFile_->Get(edm::poolNames::metaDataTreeName().c_str()) );
  if ( 0 != metaDataTree) {
    bReg = metaDataTree->GetBranch(edm::poolNames::productDescriptionBranchName().c_str());
    bReg->SetAddress(ppReg);
    bReg->GetEntry(0);
    (*ppReg)->setFrozen();
  }
  return bReg;
}

edm::BranchID 
BranchMapReader::Strategy::productToBranchID(const edm::ProductID& pid)
{
  throw edm::Exception(edm::errors::UnimplementedFeature) << "Unsupported EDM file version";
}



int BranchMapReader::getFileVersion(TFile* file) const
{
  TTree* metaDataTree = dynamic_cast<TTree*>(file->Get(edm::poolNames::metaDataTreeName().c_str()) );
  if (0==metaDataTree) {
    return 0;
  }
    
  edm::FileFormatVersion v;
  edm::FileFormatVersion* pV=&v;
  TBranch* bVer = metaDataTree->GetBranch(edm::poolNames::fileFormatVersionBranchName().c_str());
  bVer->SetAddress(&pV);
  bVer->GetEntry(0);
  return v.value_;
}

bool BranchMapReader::updateEvent(Long_t newevent)
{
  return strategy_->updateEvent(newevent);
}

bool BranchMapReader::updateFile(TFile* file)
{
  if (0 == strategy_.get()) {
    strategy_ = newStrategy(file, getFileVersion(file));
    bDesc_.clear();
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
    bDesc_.clear();
    int fileVersion = getFileVersion(file);
    if (fileVersion != strategy_->fileVersion_) {
      strategy_ = newStrategy(file, fileVersion);
    } else {
      strategy_->updateFile(file);
    }
  }
  return isNew;
}

const edm::BranchDescription
BranchMapReader::productToBranch(const edm::ProductID& pid)
{
  edm::BranchID bid = strategy_->productToBranchID(pid);
  bidToDesc::const_iterator bdi = branchDescriptionMap_.find(bid);
  if (branchDescriptionMap_.end() == bdi) {
    // std::cout << "productToBranch: product " << pid << " not found" << std::endl;
    return edm::BranchDescription();
  }
  // std::cout << "productToBranch: returning " << bdi->second << std::endl;
  return bdi->second;
}

const std::vector<edm::BranchDescription>&
BranchMapReader::getBranchDescriptions()
{
  if (bDesc_.empty()) {
    for (bidToDesc::const_iterator i = branchDescriptionMap_.begin(); i != branchDescriptionMap_.end(); ++i) {
      bDesc_.push_back(i->second);
    }
  }
  return bDesc_;
}


std::auto_ptr<BranchMapReader::Strategy>
BranchMapReader::newStrategy(TFile* file, int fileVersion)
{
  std::auto_ptr<Strategy> s;

  if (fileVersion >= 11) {
    s = std::auto_ptr<Strategy>(new internal::BranchMapReaderStrategyV11(file, fileVersion, branchDescriptionMap_));
  } else if (fileVersion >= 8) {
    s = std::auto_ptr<Strategy>(new internal::BranchMapReaderStrategyV8(file, fileVersion, branchDescriptionMap_));
  } else if (fileVersion >= 7) {
    s = std::auto_ptr<Strategy>(new internal::BranchMapReaderStrategyV7(file, fileVersion, branchDescriptionMap_));
  } else {
    s = std::auto_ptr<Strategy>(new internal::BranchMapReaderStrategyV1(file, fileVersion, branchDescriptionMap_));
  }
  return s;
}
}
