// -*- C++ -*-
//
// Package:     DataFormats/FWLite
// Class  :     LuminosityBlock
//
/**\class LuminosityBlock LuminosityBlock.h DataFormats/FWLite/interface/LuminosityBlock.h

   Description: <one line class summary>

   Usage:
   <usage>

*/
//
// Original Author:  Eric Vaandering
//         Created:  Wed Jan  13 15:01:20 EDT 2007
// $Id:
//

// system include files
#include <iostream>
#include "Reflex/Type.h"

// user include files
#include "DataFormats/FWLite/interface/LuminosityBlock.h"
#include "TFile.h"
#include "TTree.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/EDProductGetter.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"
#include "DataFormats/Provenance/interface/History.h"
#include "DataFormats/Provenance/interface/ProcessHistoryID.h"

#include "FWCore/FWLite/interface/setRefStreamer.h"

#include "FWCore/Utilities/interface/WrappedClassName.h"

#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ParameterSet/interface/ParameterSetConverter.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/FWLite/interface/LumiHistoryGetter.h"
#include "DataFormats/FWLite/interface/RunFactory.h"

//used for backwards compatability
#include "DataFormats/Provenance/interface/LuminosityBlockAux.h"

//
// constants, enums and typedefs
//
namespace fwlite {

//
// constructors and destructor
//
  LuminosityBlock::LuminosityBlock(TFile* iFile):
    branchMap_(new BranchMapReader(iFile)),
    pAux_(&aux_),
    pOldAux_(0),
    fileVersion_(-1),
    parameterSetRegistryFilled_(false),
    dataHelper_(branchMap_->getLuminosityBlockTree(),
                boost::shared_ptr<HistoryGetterBase>(new LumiHistoryGetter(this)),
                branchMap_)
  {
    if(0==iFile) {
      throw cms::Exception("NoFile")<<"The TFile pointer passed to the constructor was null";
    }

    if(0==branchMap_->getLuminosityBlockTree()) {
      throw cms::Exception("NoLumiTree")<<"The TFile contains no TTree named " <<edm::poolNames::luminosityBlockTreeName();
    }
    //need to know file version in order to determine how to read the basic product info
    fileVersion_ = branchMap_->getFileVersion(iFile);

    //got this logic from IOPool/Input/src/RootFile.cc

    TTree* luminosityBlockTree = branchMap_->getLuminosityBlockTree();
    if(fileVersion_ >= 3 ) {
      auxBranch_ = luminosityBlockTree->GetBranch(edm::BranchTypeToAuxiliaryBranchName(edm::InLumi).c_str());
      if(0==auxBranch_) {
        throw cms::Exception("NoLuminosityBlockAuxilliary")<<"The TTree "
        <<edm::poolNames::luminosityBlockTreeName()
        <<" does not contain a branch named 'LuminosityBlockAuxiliary'";
      }
      auxBranch_->SetAddress(&pAux_);
    } else {
      throw cms::Exception("OldFileVersion")<<"The FWLite Luminosity Block code des not support old file versions";
//       This code commented from fwlite::Event. May be portable if needed.
//       pOldAux_ = new edm::EventAux();
//       auxBranch_ = luminosityBlockTree->GetBranch(edm::BranchTypeToAuxBranchName(edm::InLuminosityBlock).c_str());
//       if(0==auxBranch_) {
//         throw cms::Exception("NoLuminosityBlockAux")<<"The TTree "
//           <<edm::poolNames::luminosityBlockTreeName()
//           <<" does not contain a branch named 'LuminosityBlockAux'";
//       }
//       auxBranch_->SetAddress(&pOldAux_);
    }
    branchMap_->updateLuminosityBlock(0);
    runFactory_ =  boost::shared_ptr<RunFactory>(new RunFactory());
}

  LuminosityBlock::LuminosityBlock(boost::shared_ptr<BranchMapReader> branchMap, boost::shared_ptr<RunFactory> runFactory):
    branchMap_(branchMap),
    pAux_(&aux_),
    pOldAux_(0),
    fileVersion_(-1),
    parameterSetRegistryFilled_(false),
    dataHelper_(branchMap_->getLuminosityBlockTree(),
                boost::shared_ptr<HistoryGetterBase>(new LumiHistoryGetter(this)),
                branchMap_),
    runFactory_(runFactory)
  {

    if(0==branchMap_->getLuminosityBlockTree()) {
      throw cms::Exception("NoLumiTree")<<"The TFile contains no TTree named " <<edm::poolNames::luminosityBlockTreeName();
    }
    //need to know file version in order to determine how to read the basic event info
    fileVersion_ = branchMap_->getFileVersion();
    //got this logic from IOPool/Input/src/RootFile.cc

    TTree* luminosityBlockTree = branchMap_->getLuminosityBlockTree();
    if(fileVersion_ >= 3 ) {
      auxBranch_ = luminosityBlockTree->GetBranch(edm::BranchTypeToAuxiliaryBranchName(edm::InLumi).c_str());
      if(0==auxBranch_) {
        throw cms::Exception("NoLuminosityBlockAuxilliary")<<"The TTree "
        <<edm::poolNames::luminosityBlockTreeName()
        <<" does not contain a branch named 'LuminosityBlockAuxiliary'";
      }
      auxBranch_->SetAddress(&pAux_);
    } else {
      throw cms::Exception("OldFileVersion")<<"The FWLite Luminosity Block code des not support old file versions";
/*      pOldAux_ = new edm::EventAux();
      auxBranch_ = luminosityBlockTree->GetBranch(edm::BranchTypeToAuxBranchName(edm::InLuminosityBlock).c_str());
      if(0==auxBranch_) {
        throw cms::Exception("NoLuminosityBlockAux")<<"The TTree "
          <<edm::poolNames::luminosityBlockTreeName()
          <<" does not contain a branch named 'LuminosityBlockAux'";
      }
      auxBranch_->SetAddress(&pOldAux_);*/
    }
    branchMap_->updateLuminosityBlock(0);

//     if(fileVersion_ >= 7 ) {
//       eventHistoryTree_ = dynamic_cast<TTree*>(iFile->Get(edm::poolNames::eventHistoryTreeName().c_str()));
//     }

}

LuminosityBlock::~LuminosityBlock()
{
  for(std::vector<const char*>::iterator it = labels_.begin(), itEnd=labels_.end();
      it != itEnd;
      ++it) {
    delete [] *it;
  }
  delete pOldAux_;
}

//
// member functions
//

const LuminosityBlock&
LuminosityBlock::operator++()
{
   Long_t luminosityBlockIndex = branchMap_->getLuminosityBlockEntry();
   if(luminosityBlockIndex < size())
   {
      branchMap_->updateLuminosityBlock(++luminosityBlockIndex);
   }
   return *this;
}


bool
LuminosityBlock::to (edm::RunNumber_t run, edm::LuminosityBlockNumber_t luminosityBlock)
{
   fillFileIndex();
   edm::FileIndex::const_iterator i =
      fileIndex_.findLumiPosition(run, luminosityBlock, true);
   if (fileIndex_.end() != i)
   {
      return branchMap_->updateLuminosityBlock(i->entry_);
   }
   return false;
}

void
LuminosityBlock::fillFileIndex() const
{
  if (fileIndex_.empty()) {
    TTree* meta = dynamic_cast<TTree*>(branchMap_->getFile()->Get(edm::poolNames::metaDataTreeName().c_str()));
    if (0==meta) {
      throw cms::Exception("NoMetaTree")<<"The TFile does not contain a TTree named "
        <<edm::poolNames::metaDataTreeName();
    }
    if (meta->FindBranch(edm::poolNames::fileIndexBranchName().c_str()) != 0) {
      edm::FileIndex* findexPtr = &fileIndex_;
      TBranch* b = meta->GetBranch(edm::poolNames::fileIndexBranchName().c_str());
      b->SetAddress(&findexPtr);
      b->GetEntry(0);
    } else {
      // TBD: fill the FileIndex for old file formats (prior to CMSSW 2_0_0)
      throw cms::Exception("NoFileIndexTree")<<"The TFile does not contain a TTree named "
        <<edm::poolNames::fileIndexBranchName();
    }
  }
  assert(!fileIndex_.empty());
}
const LuminosityBlock&
LuminosityBlock::toBegin()
{
   branchMap_->updateLuminosityBlock(0);
   return *this;
}

//
// const member functions
//
Long64_t
LuminosityBlock::size() const
{
  return branchMap_->getLuminosityBlockTree()->GetEntries();
}

bool
LuminosityBlock::isValid() const
{
  Long_t luminosityBlockIndex = branchMap_->getLuminosityBlockEntry();
  return luminosityBlockIndex!=-1 and luminosityBlockIndex < size();
}


LuminosityBlock::operator bool() const
{
  return isValid();
}

bool
LuminosityBlock::atEnd() const
{
  Long_t luminosityBlockIndex = branchMap_->getLuminosityBlockEntry();
  return luminosityBlockIndex==-1 or luminosityBlockIndex == size();
}


const std::string
LuminosityBlock::getBranchNameFor(const std::type_info& iInfo,
                  const char* iModuleLabel,
                  const char* iProductInstanceLabel,
                  const char* iProcessLabel) const
{
    return dataHelper_.getBranchNameFor(iInfo, iModuleLabel, iProductInstanceLabel, iProcessLabel);
}


bool
LuminosityBlock::getByLabel(const std::type_info& iInfo,
                  const char* iModuleLabel,
                  const char* iProductInstanceLabel,
                  const char* iProcessLabel,
                  void* oData) const {
    if(atEnd()) {
        throw cms::Exception("OffEnd")<<"You have requested data past the last lumi";
    }
    Long_t lumiIndex = branchMap_->getLuminosityBlockEntry();
    return dataHelper_.getByLabel(iInfo, iModuleLabel, iProductInstanceLabel, iProcessLabel, oData, lumiIndex);
}

edm::LuminosityBlockAuxiliary const&
LuminosityBlock::luminosityBlockAuxiliary() const
{
   Long_t luminosityBlockIndex = branchMap_->getLuminosityBlockEntry();
   updateAux(luminosityBlockIndex);
   return aux_;
}

void
LuminosityBlock::updateAux(Long_t luminosityBlockIndex) const
{
  if(auxBranch_->GetEntryNumber() != luminosityBlockIndex) {
    auxBranch_->GetEntry(luminosityBlockIndex);
    //handling dealing with old version
    if(0 != pOldAux_) {
      conversion(*pOldAux_,aux_);
    }
  }
}


const edm::ProcessHistory&
LuminosityBlock::history() const
{
  edm::ProcessHistoryID processHistoryID;

  bool newFormat = false;//(fileVersion_ >= 5);

  Long_t lumiIndex = branchMap_->getLuminosityBlockEntry();
  updateAux(lumiIndex);
  if (!newFormat) {
    processHistoryID = aux_.processHistoryID();
  }

  if(historyMap_.empty() || newFormat) {
    procHistoryNames_.clear();
    TTree *meta = dynamic_cast<TTree*>(branchMap_->getFile()->Get(edm::poolNames::metaDataTreeName().c_str()));
    if(0==meta) {
      throw cms::Exception("NoMetaTree")<<"The TFile does not appear to contain a TTree named "
      <<edm::poolNames::metaDataTreeName();
    }
    if (historyMap_.empty()) {
      if (fileVersion_ < 11) {
        edm::ProcessHistoryMap* pPhm=&historyMap_;
        TBranch* b = meta->GetBranch(edm::poolNames::processHistoryMapBranchName().c_str());
        b->SetAddress(&pPhm);
        b->GetEntry(0);
      } else {
        edm::ProcessHistoryVector historyVector;
        edm::ProcessHistoryVector* pPhv=&historyVector;
        TBranch* b = meta->GetBranch(edm::poolNames::processHistoryBranchName().c_str());
        b->SetAddress(&pPhv);
        b->GetEntry(0);
        for (edm::ProcessHistoryVector::const_iterator i = historyVector.begin(), e = historyVector.end();
            i != e; ++i) {
          historyMap_.insert(std::make_pair(i->id(), *i));
        }
      }
    }
//     if (newFormat) {
//       if (fileVersion_ >= 7) {
//         edm::History history;
//         edm::History* pHistory = &history;
//         TBranch* eventHistoryBranch = eventHistoryTree_->GetBranch(edm::poolNames::eventHistoryBranchName().c_str());
//         if (!eventHistoryBranch)
//           throw edm::Exception(edm::errors::FatalRootError)
//             << "Failed to find history branch in event history tree";
//         eventHistoryBranch->SetAddress(&pHistory);
//         eventHistoryTree_->GetEntry(lumiIndex);
//         processHistoryID = history.processHistoryID();
//       } else {
//         std::vector<edm::EventProcessHistoryID> *pEventProcessHistoryIDs = &eventProcessHistoryIDs_;
//         TBranch* b = meta->GetBranch(edm::poolNames::eventHistoryBranchName().c_str());
//         b->SetAddress(&pEventProcessHistoryIDs);
//         b->GetEntry(0);
//         edm::EventProcessHistoryID target(aux_.id(), edm::ProcessHistoryID());
//         processHistoryID = std::lower_bound(eventProcessHistoryIDs_.begin(), eventProcessHistoryIDs_.end(), target)->processHistoryID_;
//       }
//     }

  }
  return historyMap_[processHistoryID];
}


edm::EDProduct const*
LuminosityBlock::getByProductID(edm::ProductID const& iID) const {
  Long_t luminosityBlockIndex = branchMap_->getLuminosityBlockEntry();
  return dataHelper_.getByProductID(iID, luminosityBlockIndex);
}


//
// static member functions
//


namespace {
  struct NoDelete {
    void operator()(void*){}
  };
}

fwlite::Run const& LuminosityBlock::getRun() const {
  run_ = runFactory_->makeRun(boost::shared_ptr<BranchMapReader>(&*branchMap_,NoDelete()));
  edm::RunNumber_t run = luminosityBlockAuxiliary().run();
  run_->to(run);
  return *run_;
}

}
