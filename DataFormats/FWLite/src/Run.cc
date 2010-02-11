// -*- C++ -*-
//
// Package:     DataFormats/FWLite
// Class  :     Run
//
/**\class Run Run.h DataFormats/FWLite/interface/Run.h

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
#include "DataFormats/FWLite/interface/Run.h"
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
#include "DataFormats/FWLite/interface/RunHistoryGetter.h"

//used for backwards compatability
#include "DataFormats/Provenance/interface/RunAux.h"

//
// constants, enums and typedefs
//
namespace fwlite {

//
// constructors and destructor
//
  Run::Run(TFile* iFile):
    branchMap_(new BranchMapReader(iFile)),
    pAux_(&aux_),
    pOldAux_(0),
    fileVersion_(-1),
    parameterSetRegistryFilled_(false),
    dataHelper_(branchMap_->getRunTree(), boost::shared_ptr<HistoryGetterBase>(new RunHistoryGetter(this)))
  {
    if(0==iFile) {
      throw cms::Exception("NoFile")<<"The TFile pointer passed to the constructor was null";
    }

    if(0==branchMap_->getRunTree()) {
      throw cms::Exception("NoRunTree")<<"The TFile contains no TTree named " <<edm::poolNames::runTreeName();
    }
    //need to know file version in order to determine how to read the basic product info
    fileVersion_ = branchMap_->getFileVersion(iFile);

    //got this logic from IOPool/Input/src/RootFile.cc

    TTree* runTree = branchMap_->getRunTree();
    if(fileVersion_ >= 3 ) {
      auxBranch_ = runTree->GetBranch(edm::BranchTypeToAuxiliaryBranchName(edm::InRun).c_str());
      if(0==auxBranch_) {
        throw cms::Exception("NoRunAuxilliary")<<"The TTree "
        <<edm::poolNames::runTreeName()
        <<" does not contain a branch named 'RunAuxiliary'";
      }
      auxBranch_->SetAddress(&pAux_);
    } else {
      throw cms::Exception("OldFileVersion")<<"The FWLite Run code des not support old file versions";
//       This code commented from fwlite::Event. May be portable if needed.
//       pOldAux_ = new edm::EventAux();
//       auxBranch_ = runTree->GetBranch(edm::BranchTypeToAuxBranchName(edm::InRun).c_str());
//       if(0==auxBranch_) {
//         throw cms::Exception("NoRunAux")<<"The TTree "
//           <<edm::poolNames::runTreeName()
//           <<" does not contain a branch named 'RunAux'";
//       }
//       auxBranch_->SetAddress(&pOldAux_);
    }
    branchMap_->updateRun(0);
//     getter_ = boost::shared_ptr<edm::EDProductGetter>(new ProductGetter(this));
}

  Run::Run(boost::shared_ptr<BranchMapReader> branchMap):
    branchMap_(branchMap),
    pAux_(&aux_),
    pOldAux_(0),
    fileVersion_(-1),
    parameterSetRegistryFilled_(false),
    dataHelper_(branchMap_->getRunTree(), boost::shared_ptr<HistoryGetterBase>(new RunHistoryGetter(this)))
  {
    if(0==branchMap_->getRunTree()) {
      throw cms::Exception("NoRunTree")<<"The TFile contains no TTree named " <<edm::poolNames::runTreeName();
    }
    //need to know file version in order to determine how to read the basic event info
    fileVersion_ = branchMap_->getFileVersion();
    //got this logic from IOPool/Input/src/RootFile.cc

    TTree* runTree = branchMap_->getRunTree();
    if(fileVersion_ >= 3 ) {
      auxBranch_ = runTree->GetBranch(edm::BranchTypeToAuxiliaryBranchName(edm::InRun).c_str());
      if(0==auxBranch_) {
        throw cms::Exception("NoRunAuxilliary")<<"The TTree "
        <<edm::poolNames::runTreeName()
        <<" does not contain a branch named 'RunAuxiliary'";
      }
      auxBranch_->SetAddress(&pAux_);
    } else {
      throw cms::Exception("OldFileVersion")<<"The FWLite Run code des not support old file versions";
/*      pOldAux_ = new edm::EventAux();
      auxBranch_ = runTree->GetBranch(edm::BranchTypeToAuxBranchName(edm::InRun).c_str());
      if(0==auxBranch_) {
        throw cms::Exception("NoRunAux")<<"The TTree "
          <<edm::poolNames::runTreeName()
          <<" does not contain a branch named 'RunAux'";
      }
      auxBranch_->SetAddress(&pOldAux_);*/
    }
    branchMap_->updateRun(0);

//     if(fileVersion_ >= 7 ) {
//       eventHistoryTree_ = dynamic_cast<TTree*>(iFile->Get(edm::poolNames::eventHistoryTreeName().c_str()));
//     }

//     getter_ = boost::shared_ptr<edm::EDProductGetter>(new ProductGetter(this));
}

Run::~Run()
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

const Run&
Run::operator++()
{
   Long_t runIndex = branchMap_->getRunEntry();
   if(runIndex < size())
   {
      branchMap_->updateRun(++runIndex);
   }
   return *this;
}


bool
Run::to (edm::RunNumber_t run)
{
   fillFileIndex();
   edm::FileIndex::const_iterator i =
      fileIndex_.findRunPosition(run, true);
   if (fileIndex_.end() != i)
   {
      return branchMap_->updateRun(i->entry_);
   }
   return false;
}

void
Run::fillFileIndex() const
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
const Run&
Run::toBegin()
{
   branchMap_->updateRun(0);
   return *this;
}

//
// const member functions
//
Long64_t
Run::size() const
{
  return branchMap_->getRunTree()->GetEntries();
}

bool
Run::isValid() const
{
  Long_t runIndex = branchMap_->getRunEntry();
  return runIndex!=-1 and runIndex < size();
}


Run::operator bool() const
{
  return isValid();
}

bool
Run::atEnd() const
{
  Long_t runIndex = branchMap_->getRunEntry();
  return runIndex==-1 or runIndex == size();
}


bool
Run::getByLabel(const std::type_info& iInfo,
                  const char* iModuleLabel,
                  const char* iProductInstanceLabel,
                  const char* iProcessLabel,
                  void* oData) const
{
    if(atEnd()) {
        throw cms::Exception("OffEnd")<<"You have requested data past the last run";
    }
    Long_t runIndex = branchMap_->getRunEntry();
    return dataHelper_.getByLabel(iInfo, iModuleLabel, iProductInstanceLabel, iProcessLabel, oData, runIndex);
}

edm::RunAuxiliary const&
Run::runAuxiliary() const
{
   Long_t runIndex = branchMap_->getRunEntry();
   updateAux(runIndex);
   return aux_;
}

void
Run::updateAux(Long_t runIndex) const
{
  if(auxBranch_->GetEntryNumber() != runIndex) {
    auxBranch_->GetEntry(runIndex);
    //handling dealing with old version
    if(0 != pOldAux_) {
      conversion(*pOldAux_,aux_);
    }
  }
}

const edm::ProcessHistory&
Run::history() const
{
  edm::ProcessHistoryID processHistoryID;

  bool newFormat = false;//(fileVersion_ >= 5);

  Long_t runIndex = branchMap_->getRunEntry();
  updateAux(runIndex);
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
//         eventHistoryTree_->GetEntry(runIndex);
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
Run::getByProductID(edm::ProductID const& iID) const
{
  Long_t runIndex = branchMap_->getRunEntry();
  return dataHelper_.getByProductID(iID, runIndex);
}


//
// static member functions
//
void
Run::throwProductNotFoundException(const std::type_info& iType, const char* iModule, const char* iProduct, const char* iProcess)
{
    edm::TypeID type(iType);
  throw edm::Exception(edm::errors::ProductNotFound)<<"A branch was found for \n  type ='"<<type.className()<<"'\n  module='"<<iModule
    <<"'\n  productInstance='"<<((0!=iProduct)?iProduct:"")<<"'\n  process='"<<((0!=iProcess)?iProcess:"")<<"'\n"
    "but no data is available for this Run";
}
}
