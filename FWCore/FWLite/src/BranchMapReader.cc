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
// $Id: BranchMapReader.cc,v 1.4.2.1 2008/12/17 04:40:53 wmtan Exp $
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
#include "FWCore/Utilities/interface/Exception.h"

namespace fwlite {
  static BranchMapReader::eeiMap emptyMapper;

  namespace internal {
    class BranchMapReaderStrategyV1 : public fwlite::BranchMapReader::Strategy {
    public:
      BranchMapReaderStrategyV1(TFile* file, int fileVersion, BranchMapReader::eeiMap& eventInfoMap, 
                                BranchMapReader::bidToDesc& branchDescriptionMap);
      virtual bool updateFile(TFile* file);
      virtual bool updateMap();
    private:
    };

    BranchMapReaderStrategyV1::BranchMapReaderStrategyV1(TFile* file, int fileVersion,
      BranchMapReader::eeiMap& eventInfoMap,
      BranchMapReader::bidToDesc& branchDescriptionMap)
      : Strategy(file, fileVersion, eventInfoMap, branchDescriptionMap)
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

      eventInfoMap_ = emptyMapper;
      branchDescriptionMap_.clear();
      
      edm::ProductRegistry reg;
      edm::ProductRegistry* pReg = &reg;
      TBranch* br = getBranchRegistry(&pReg);

      if (0 != br) {
        const edm::ProductRegistry::ProductList& prodList = reg.productList();

        for(edm::ProductRegistry::ProductList::const_iterator it = prodList.begin(), itEnd = prodList.end(); it != itEnd; ++it) {
          if (edm::InEvent == it->second.branchType()) {
            edm::ProductStatus status = edm::productstatus::uninitialized();
            // call to regenerate branchName
            it->second.init();
  	        edm::EventEntryInfo entry(it->second.branchID(), status, it->second.oldProductID());
  	        // eventInfoMap_.insert(entry); // Kludge to allow compilation
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
      BranchMapReaderStrategyV7(TFile* file, int fileVersion, BranchMapReader::eeiMap& eventInfoMap,
                                BranchMapReader::bidToDesc& branchDescriptionMap);
    };

    BranchMapReaderStrategyV7::BranchMapReaderStrategyV7(TFile* file, int fileVersion, BranchMapReader::eeiMap& eventInfoMap,
      BranchMapReader::bidToDesc& branchDescriptionMap)
    : BranchMapReaderStrategyV1(file, fileVersion, eventInfoMap, branchDescriptionMap)
    {
      updateFile(file);
    }

    class BranchMapReaderStrategyV8 : public fwlite::BranchMapReader::Strategy {
    public:
      BranchMapReaderStrategyV8(TFile* file, int fileVersion, BranchMapReader::eeiMap& eventInfoMap,
                                BranchMapReader::bidToDesc& branchDescriptionMap);
      virtual bool updateFile(TFile* file);
      virtual bool updateEvent(Long_t eventEntry);
      virtual bool updateMap();
    private:
      TBranch* entryInfoBranch_;
      edm::EventEntryInfoVector  eventEntryInfoVector_;
      edm::EventEntryInfoVector* pEventEntryInfoVector_;
    };

    BranchMapReaderStrategyV8::BranchMapReaderStrategyV8(TFile* file, int fileVersion,
      BranchMapReader::eeiMap& eventInfoMap, BranchMapReader::bidToDesc& branchDescriptionMap)
    : Strategy(file, fileVersion, eventInfoMap, branchDescriptionMap),
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

      eventInfoMap_ = emptyMapper;
      assert (entryInfoBranch_);

      entryInfoBranch_->GetEntry(eventEntry_);

      for (std::vector<edm::EventEntryInfo>::const_iterator it = pEventEntryInfoVector_->begin(), 
           itEnd = pEventEntryInfoVector_->end();
           it != itEnd; ++it) {
        // eventInfoMap_.insert(*it); // Kludge to allow compilation
        // std::cout << "v8 updatemap " << it->productID() << " " << it->branchID() << std::endl;
      }
      mapperFilled_ = true;
      return true;
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

BranchMapReader::Strategy::Strategy(TFile* file, int fileVersion, eeiMap& eventInfoMap,
  BranchMapReader::bidToDesc& branchDescriptionMap)
  : currentFile_(file), fileVersion_(fileVersion), eventEntry_(-1), eventInfoMap_(eventInfoMap),
    branchDescriptionMap_(branchDescriptionMap), mapperFilled_(false)
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

  if (fileVersion >= 8) {
    s = std::auto_ptr<Strategy>(new internal::BranchMapReaderStrategyV8(file, fileVersion, eventInfoMap_, branchDescriptionMap_));
  } else if (fileVersion >= 7) {
    s = std::auto_ptr<Strategy>(new internal::BranchMapReaderStrategyV7(file, fileVersion, eventInfoMap_, branchDescriptionMap_));
  } else {
    s = std::auto_ptr<Strategy>(new internal::BranchMapReaderStrategyV1(file, fileVersion, eventInfoMap_, branchDescriptionMap_));
  }
  return s;
}
}
