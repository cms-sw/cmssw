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

//used for backwards compatability
#include "DataFormats/Provenance/interface/LuminosityBlockAux.h"

//
// constants, enums and typedefs
//
namespace fwlite {
//
// static data member definitions
//
  namespace internalLS {
    const char* const DataKey::kEmpty="";

    class ProductGetter : public edm::EDProductGetter {
public:
      ProductGetter(LuminosityBlock* iLuminosityBlock) : luminosityBlock_(iLuminosityBlock) {}

      edm::EDProduct const*
      getIt(edm::ProductID const& iID) const {
        return luminosityBlock_->getByProductID(iID);
      }
private:
      LuminosityBlock* luminosityBlock_;

    };
  }
  typedef std::map<internalLS::DataKey, boost::shared_ptr<internalLS::Data> > DataMap;
  // empty object used to signal that the branch requested was not found
  static internalLS::Data branchNotFound;

//
// constructors and destructor
//
  LuminosityBlock::LuminosityBlock(TFile* iFile):
    branchMap_(new BranchMapReader(iFile)),
    pAux_(&aux_),
    pOldAux_(0),
    fileVersion_(-1),
    parameterSetRegistryFilled_(false)
  {
    if(0==iFile) {
      throw cms::Exception("NoFile")<<"The TFile pointer passed to the constructor was null";
    }

    if(0==branchMap_->getLuminosityBlockTree()) {
      throw cms::Exception("NoLumiTree")<<"The TFile contains no TTree named " <<edm::poolNames::luminosityBlockTreeName();
    }
    //need to know file version in order to determine how to read the basic event info
    fileVersion_ = branchMap_->getFileVersion(iFile);

    //got this logic from IOPool/Input/src/RootFile.cc

    TTree* luminosityBlockTree = branchMap_->getLuminosityBlockTree();
//    if(fileVersion_ >= 3 ) {
      auxBranch_ = luminosityBlockTree->GetBranch(edm::BranchTypeToAuxiliaryBranchName(edm::InLumi).c_str());
      if(0==auxBranch_) {
        throw cms::Exception("NoLuminosityBlockAuxilliary")<<"The TTree "
        <<edm::poolNames::luminosityBlockTreeName()
        <<" does not contain a branch named 'LuminosityBlockAuxiliary'";
      }
      auxBranch_->SetAddress(&pAux_);
/*    } else {
      pOldAux_ = new edm::EventAux();
      auxBranch_ = luminosityBlockTree->GetBranch(edm::BranchTypeToAuxBranchName(edm::InLuminosityBlock).c_str());
      if(0==auxBranch_) {
        throw cms::Exception("NoLuminosityBlockAux")<<"The TTree "
          <<edm::poolNames::luminosityBlockTreeName()
          <<" does not contain a branch named 'LuminosityBlockAux'";
      }
      auxBranch_->SetAddress(&pOldAux_);
    }*/
    branchMap_->updateLuminosityBlock(0);

//     if(fileVersion_ >= 7 ) {
//       eventHistoryTree_ = dynamic_cast<TTree*>(iFile->Get(edm::poolNames::eventHistoryTreeName().c_str()));
//     }

    getter_ = boost::shared_ptr<edm::EDProductGetter>(new internalLS::ProductGetter(this));
}

  LuminosityBlock::LuminosityBlock(boost::shared_ptr<BranchMapReader> branchMap):
    branchMap_(branchMap),
    pAux_(&aux_),
    pOldAux_(0),
    fileVersion_(-1),
    parameterSetRegistryFilled_(false)
  {

    if(0==branchMap_->getLuminosityBlockTree()) {
      throw cms::Exception("NoLumiTree")<<"The TFile contains no TTree named " <<edm::poolNames::luminosityBlockTreeName();
    }
    //need to know file version in order to determine how to read the basic event info
    fileVersion_ = branchMap_->getFileVersion();
    //got this logic from IOPool/Input/src/RootFile.cc

    TTree* luminosityBlockTree = branchMap_->getLuminosityBlockTree();
//    if(fileVersion_ >= 3 ) {
      auxBranch_ = luminosityBlockTree->GetBranch(edm::BranchTypeToAuxiliaryBranchName(edm::InLumi).c_str());
      if(0==auxBranch_) {
        throw cms::Exception("NoLuminosityBlockAuxilliary")<<"The TTree "
        <<edm::poolNames::luminosityBlockTreeName()
        <<" does not contain a branch named 'LuminosityBlockAuxiliary'";
      }
      auxBranch_->SetAddress(&pAux_);
/*    } else {
      pOldAux_ = new edm::EventAux();
      auxBranch_ = luminosityBlockTree->GetBranch(edm::BranchTypeToAuxBranchName(edm::InLuminosityBlock).c_str());
      if(0==auxBranch_) {
        throw cms::Exception("NoLuminosityBlockAux")<<"The TTree "
          <<edm::poolNames::luminosityBlockTreeName()
          <<" does not contain a branch named 'LuminosityBlockAux'";
      }
      auxBranch_->SetAddress(&pOldAux_);
    }*/
    branchMap_->updateLuminosityBlock(0);

//     if(fileVersion_ >= 7 ) {
//       eventHistoryTree_ = dynamic_cast<TTree*>(iFile->Get(edm::poolNames::eventHistoryTreeName().c_str()));
//     }

    getter_ = boost::shared_ptr<edm::EDProductGetter>(new internalLS::ProductGetter(this));
}



// Event::Event(const Event& rhs)
// {
//    // do actual copying here;
// }

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

/*
void
Event::getByBranchName(const std::type_info& iInfo, const char* iName, void*& oData) const
{
  oData=0;
  std::cout <<iInfo.name()<<std::endl;
}
*/

static
TBranch* findBranch(TTree* iTree, const std::string& iMainLabels, const std::string& iProcess) {
  std::string branchName(iMainLabels);
  branchName+=iProcess;
  //branchName+=".obj";
  branchName+=".";
  return iTree->GetBranch(branchName.c_str());
}


static
void getBranchData(edm::EDProductGetter* iGetter,
                   Long64_t iLuminosityBlockIndex,
                   internalLS::Data& iData)
{
  GetterOperate op(iGetter);

  //WORK AROUND FOR ROOT!!
  //Create a new instance so that we can clear any cache the object uses
  //this slows the code down
  Reflex::Object obj = iData.obj_;
  iData.obj_ = iData.obj_.TypeOf().Construct();
  iData.pObj_ = iData.obj_.Address();
  iData.branch_->SetAddress(&(iData.pObj_));
  //If a REF to this was requested in the past, we might as well do the work now
  if(0!=iData.pProd_) {
    //The type is the same so the offset will be the same
    void* p = iData.pProd_;
    iData.pProd_ = reinterpret_cast<edm::EDProduct*>(static_cast<char*>(iData.obj_.Address())+(static_cast<char*>(p)-static_cast<char*>(obj.Address())));
  }
  obj.Destruct();
  //END OF WORK AROUND

  iData.branch_->GetEntry(iLuminosityBlockIndex);
  iData.lastLuminosityBlock_=iLuminosityBlockIndex;
}


internalLS::Data&
LuminosityBlock::getBranchDataFor(const std::type_info& iInfo,
                  const char* iModuleLabel,
                  const char* iProductInstanceLabel,
                  const char* iProcessLabel) const
{
  //std::cout <<iInfo.name()<<" '"<<iModuleLabel<<"' '"<< (( 0!=iProductInstanceLabel)?iProductInstanceLabel:"")<<"' '"
  //<<((0!=iProcessLabel)?iProcessLabel:"")<<"'"<<std::endl;
  //std::cout <<iInfo.name()<<std::endl;
  edm::TypeID type(iInfo);
  internalLS::DataKey key(type, iModuleLabel, iProductInstanceLabel, iProcessLabel);

  boost::shared_ptr<internalLS::Data> theData;
  DataMap::iterator itFind = data_.find(key);
  if(itFind == data_.end() ) {
    //std::cout <<"did not find the key"<<std::endl;
    //see if such a branch actually exists
    const std::string sep("_");
    //CHANGE: If this fails, need to lookup the the friendly name which was used to write the file
    std::string name(type.friendlyClassName());
    name +=sep+std::string(key.module());
    name +=sep+std::string(key.product())+sep;

    //if we have to lookup the process label, remember it and register the product again
    std::string foundProcessLabel;
    TBranch* branch = 0;
    TTree* luminosityBlockTree = branchMap_->getLuminosityBlockTree();
//    std::cout <<" iPl "<<iProcessLabel << " :: " << internalLS::DataKey::kEmpty <<std::endl; //EWV

    if (0==iProcessLabel || iProcessLabel==internalLS::DataKey::kEmpty ||
        strlen(iProcessLabel)==0)
    {
      const std::string* lastLabel=0;
      //have to search in reverse order since newest are on the bottom
      const edm::ProcessHistory& h = LuminosityBlock::history();
      for (edm::ProcessHistory::const_reverse_iterator iproc = h.rbegin(), eproc = h.rend();
           iproc != eproc;
           ++iproc) {

        lastLabel = &(iproc->processName());
        branch=findBranch(luminosityBlockTree,name,iproc->processName());
        if(0!=branch) {
          break;
        }
      }
      if(0==branch) {
        return branchNotFound;
      }
      //do we already have this one?
      if(0!=lastLabel) {
        //std::cout <<" process name "<<*lastLabel<<std::endl;
        internalLS::DataKey fullKey(type,iModuleLabel,iProductInstanceLabel,lastLabel->c_str());
        itFind = data_.find(fullKey);
        if(itFind != data_.end()) {
          //remember the data we've found
          //std::cout <<"  key already exists"<<std::endl;
          theData = itFind->second;
        } else {
          //only set this if we don't already have it
          // since it this string is not empty we re-register
          //std::cout <<"  key does not already exists"<<std::endl;
          foundProcessLabel = *lastLabel;
        }
      }
    }else {
      //we have all the pieces
      branch = findBranch(luminosityBlockTree,name,key.process());
      if(0==branch){
        return branchNotFound;
      }
    }

    //cache the info
    char* newModule = new char[strlen(iModuleLabel)+1];
    std::strcpy(newModule,iModuleLabel);
    labels_.push_back(newModule);

    char* newProduct = const_cast<char*>(key.product());
    if(newProduct[0] != 0) {
      newProduct = new char[strlen(newProduct)+1];
      std::strcpy(newProduct,key.product());
      labels_.push_back(newProduct);
    }
    char* newProcess = const_cast<char*>(key.process());
    if(newProcess[0]!=0) {
      newProcess = new char[strlen(newProcess)+1];
      std::strcpy(newProcess,key.process());
      labels_.push_back(newProcess);
    }
    internalLS::DataKey newKey(edm::TypeID(iInfo),newModule,newProduct,newProcess);

    if(0 == theData.get() ) {
      //We do not already have this data as another key

      //Use Reflex to create an instance of the object to be used as a buffer
      Reflex::Type rType = Reflex::Type::ByTypeInfo(iInfo);
      if(rType == Reflex::Type()) {
        throw cms::Exception("UnknownType")<<"No Reflex dictionary exists for type "<<iInfo.name();
      }
      Reflex::Object obj = rType.Construct();

      if(obj.Address() == 0) {
        throw cms::Exception("ConstructionFailed")<<"failed to construct an instance of "<<rType.Name();
      }
      boost::shared_ptr<internalLS::Data> newData(new internalLS::Data() );
      newData->branch_ = branch;
      newData->obj_ = obj;
      newData->lastLuminosityBlock_=-1;
      newData->pObj_ = obj.Address();
      newData->pProd_ = 0;
      branch->SetAddress(&(newData->pObj_));
      theData = newData;
    }
    itFind = data_.insert(std::make_pair(newKey, theData)).first;

    if(foundProcessLabel.size()) {
      //also remember it with the process label
      newProcess = new char[foundProcessLabel.size()+1];
      std::strcpy(newProcess,foundProcessLabel.c_str());
      labels_.push_back(newProcess);
      internalLS::DataKey newKey(edm::TypeID(iInfo),newModule,newProduct,newProcess);

      data_.insert(std::make_pair(newKey,theData));
    }
  }
  return *(itFind->second);
}

const std::string
LuminosityBlock::getBranchNameFor(const std::type_info& iInfo,
                  const char* iModuleLabel,
                  const char* iProductInstanceLabel,
                  const char* iProcessLabel) const
{
  internalLS::Data& theData =
    LuminosityBlock::getBranchDataFor(iInfo, iModuleLabel, iProductInstanceLabel, iProcessLabel);

  if (0 != theData.branch_) {
    return std::string(theData.branch_->GetName());
  }
  return std::string("");
}

bool
LuminosityBlock::getByLabel(const std::type_info& iInfo,
                  const char* iModuleLabel,
                  const char* iProductInstanceLabel,
                  const char* iProcessLabel,
                  void* oData) const
{
  if(atEnd()) {
    throw cms::Exception("OffEnd")<<"You have requested data past the last lumi";
  }
  void** pOData = reinterpret_cast<void**>(oData);
  *pOData = 0;

  internalLS::Data& theData =
    LuminosityBlock::getBranchDataFor(iInfo, iModuleLabel, iProductInstanceLabel, iProcessLabel);

  if (0 != theData.branch_) {
    Long_t lumiIndex = branchMap_->getLuminosityBlockEntry();
    if(lumiIndex != theData.lastLuminosityBlock_) {
      //haven't gotten the data for this event
      //std::cout <<" getByLabel getting data"<<std::endl;
      getBranchData(getter_.get(), lumiIndex, theData);
    }
    *pOData = theData.obj_.Address();
  }

  if ( 0 == *pOData ) return false;
  else return true;
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

//EWV: Should be able to use LuminosityBlock auxillary ProcessHistoryID and processHistory branch to get process histories rather than event history branch.

//const edm::ProcessHistory& LuminosityBlock::history() const { return edm::ProcessHistory();}

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
LuminosityBlock::getByProductID(edm::ProductID const& iID) const
{
  //std::cout <<"getByProductID"<<std::endl;
  std::map<edm::ProductID,boost::shared_ptr<internalLS::Data> >::const_iterator itFound = idToData_.find(iID);
  if(itFound == idToData_.end() ) {
    //std::cout <<" not found"<<std::endl;
    edm::BranchDescription bDesc = branchMap_->productToBranch(iID);

    if (!bDesc.branchID().isValid()) {
      return 0;
    }

    //std::cout <<"  get Type for class"<<std::endl;
    //Calculate the key from the branch description
    Reflex::Type type( Reflex::Type::ByName(edm::wrappedClassName(bDesc.fullClassName())));
    assert( Reflex::Type() != type) ;

    //std::cout <<"  build key"<<std::endl;
    //Only the product instance label may be empty
    const char* pIL = bDesc.productInstanceName().c_str();
    if(pIL[0] == 0) {
      pIL = 0;
    }
    internalLS::DataKey k(edm::TypeID(type.TypeInfo()),
                        bDesc.moduleLabel().c_str(),
                        pIL,
                        bDesc.processName().c_str());

    //has this already been gotten?
    KeyToDataMap::iterator itData = data_.find(k);
    if(data_.end() == itData) {
      //ask for the data
      void* dummy = 0;
      getByLabel(type.TypeInfo(),
                 k.module(),
                 k.product(),
                 k.process(),
                 &dummy);
      //std::cout <<"  called"<<std::endl;
      if (0 == dummy) {
        return 0;
      }
      itData = data_.find(k);
      assert(itData != data_.end());
      //assert(0!=dummy);
      assert(dummy == itData->second->obj_.Address());
    }
    itFound = idToData_.insert(std::make_pair(iID,itData->second)).first;
  }
  Long_t luminosityBlockIndex = branchMap_->getLuminosityBlockEntry();
  if(luminosityBlockIndex != itFound->second->lastLuminosityBlock_) {
    //haven't gotten the data for this event
    getBranchData(getter_.get(), luminosityBlockIndex, *(itFound->second));
  }
  if(0==itFound->second->pProd_) {
    //std::cout <<"  need to convert"<<std::endl;
    //need to convert pointer to proper type
    static Reflex::Type sEDProd( Reflex::Type::ByTypeInfo(typeid(edm::EDProduct)));
    //assert( sEDProd != Reflex::Type() );
    Reflex::Object edProdObj = itFound->second->obj_.CastObject( sEDProd );

    itFound->second->pProd_ = reinterpret_cast<edm::EDProduct*>(edProdObj.Address());

    //std::cout <<" type "<<typeid(itFound->second->pProd_).name()<<std::endl;
    if(0==itFound->second->pProd_) {
      cms::Exception("FailedConversion")
      <<"failed to convert a '"<<itFound->second->obj_.TypeOf().Name()<<"' to a edm::EDProduct";
    }
  }
  //std::cout <<"finished getByProductID"<<std::endl;
  return itFound->second->pProd_;
}


//
// static member functions
//
void
LuminosityBlock::throwProductNotFoundException(const std::type_info& iType, const char* iModule, const char* iProduct, const char* iProcess)
{
    edm::TypeID type(iType);
  throw edm::Exception(edm::errors::ProductNotFound)<<"A branch was found for \n  type ='"<<type.className()<<"'\n  module='"<<iModule
    <<"'\n  productInstance='"<<((0!=iProduct)?iProduct:"")<<"'\n  process='"<<((0!=iProcess)?iProcess:"")<<"'\n"
    "but no data is available for this Lumi";
}
}
