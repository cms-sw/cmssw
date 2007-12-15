// -*- C++ -*-
//
// Package:     FWLite
// Class  :     Event
// 
// Implementation:
//     <Notes on implementation>
//
// Original Author:  Chris Jones
//         Created:  Tue May  8 15:07:03 EDT 2007
// $Id: Event.cc,v 1.12 2007/12/07 23:22:53 wmtan Exp $
//

// system include files
#include <iostream>
#include "Reflex/Type.h"

// user include files
#include "DataFormats/FWLite/interface/Event.h"
#include "TFile.h"
#include "TTree.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Provenance/interface/BranchType.h"
#include "DataFormats/Common/interface/EDProduct.h"
#include "DataFormats/Common/interface/EDProductGetter.h"

#include "DataFormats/Provenance/interface/ProductRegistry.h"
#include "DataFormats/Provenance/interface/FileFormatVersion.h"

#include "FWCore/FWLite/interface/setRefStreamer.h"

#include "FWCore/Utilities/interface/WrappedClassName.h"

//used for backwards compatability
#include "DataFormats/Provenance/interface/EventAux.h"

//
// constants, enums and typedefs
//
namespace fwlite {
//
// static data member definitions
//
  namespace internal {
    const char* const DataKey::kEmpty="";
    
    class ProductGetter : public edm::EDProductGetter {
public:
      ProductGetter(Event* iEvent) : event_(iEvent) {}
      
      edm::EDProduct const*
      getIt(edm::ProductID const& iID) const {
        return event_->getByProductID(iID);
      }
private:
      Event* event_;
      
    };
  }
  typedef std::map<internal::DataKey, boost::shared_ptr<internal::Data> > DataMap;

//
// constructors and destructor
//
  Event::Event(TFile* iFile):
  file_(iFile),
  eventTree_(0),
  eventIndex_(-1),
  pAux_(&aux_),
  pOldAux_(0),
  fileVersion_(-1),
  prodReg_(0)
{
    if(0==file_) {
      throw cms::Exception("NoFile")<<"The TFile pointer passed to the constructor was null";
    }
    
    eventTree_ = dynamic_cast<TTree*>(iFile->Get(edm::poolNames::eventTreeName().c_str()));
    if(0==eventTree_) {
      throw cms::Exception("NoEventTree")<<"The TFile contains no TTree named "<<edm::poolNames::eventTreeName();
    }
    //need to know file version in order to determine how to read the basic event info
    {
      TTree* metaDataTree = dynamic_cast<TTree*>(file_->Get(edm::poolNames::metaDataTreeName().c_str()) );
      assert(0!=metaDataTree);
      
      edm::FileFormatVersion v;
      edm::FileFormatVersion* pV=&v;
      TBranch* bVer = metaDataTree->GetBranch(edm::poolNames::fileFormatVersionBranchName().c_str());
      bVer->SetAddress(&pV);
      bVer->GetEntry(0);
      fileVersion_ = v.value_;
    }
    //got this logic from IOPool/Input/src/RootFile.cc
    
    if(fileVersion_ >= 3 ) {
      auxBranch_ = eventTree_->GetBranch(edm::BranchTypeToAuxiliaryBranchName(edm::InEvent).c_str());
      if(0==auxBranch_) {
        throw cms::Exception("NoEventAuxilliary")<<"The TTree "
        <<edm::poolNames::eventTreeName()
        <<" does not contain a branch named 'EventAuxiliary'";
      }
      auxBranch_->SetAddress(&pAux_);
    } else {
      pOldAux_ = new edm::EventAux();
      auxBranch_ = eventTree_->GetBranch(edm::BranchTypeToAuxBranchName(edm::InEvent).c_str());
      if(0==auxBranch_) {
        throw cms::Exception("NoEventAux")<<"The TTree "
          <<edm::poolNames::eventTreeName()
          <<" does not contain a branch named 'EventAux'";
      }
      auxBranch_->SetAddress(&pOldAux_);
    }
    eventIndex_=0;
    
    getter_ = std::auto_ptr<edm::EDProductGetter>(new internal::ProductGetter(this));
}

// Event::Event(const Event& rhs)
// {
//    // do actual copying here;
// }

Event::~Event()
{
  for(std::vector<const char*>::iterator it = labels_.begin(), itEnd=labels_.end();
      it != itEnd;
      ++it) {
    delete [] *it;
  }
  delete prodReg_;
  delete pOldAux_;
}

//
// assignment operators
//
// const Event& Event::operator=(const Event& rhs)
// {
//   //An exception safe implementation is
//   Event temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
const Event& 
Event::operator++()
{
  if(eventIndex_ < size()) {
    ++eventIndex_;
  }
  return *this;
}

const Event& 
Event::to(Long64_t iEntry)
{
  eventIndex_ = iEntry;
  return *this;
}

const Event& 
Event::toBegin()
{
  eventIndex_ = 0;
  return *this;
}

//
// const member functions
//
Long64_t
Event::size() const 
{
  return eventTree_->GetEntries();
}

bool
Event::isValid() const
{
  return eventIndex_!=-1 and eventIndex_ < size(); 
}


Event::operator bool() const
{
  return isValid();
}

bool
Event::atEnd() const
{
  return eventIndex_==-1 or eventIndex_ == size();
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
                   Long64_t iEventIndex,
                   internal::Data& iData)
{
  GetterOperate op(iGetter);
  
  //WORK AROUND FOR ROOT!!
  //Create a new instance so that we can clear any cache the object uses
  //this slows the code down 
  ROOT::Reflex::Object obj = iData.obj_;
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
  
  iData.branch_->GetEntry(iEventIndex);
  iData.lastEvent_=iEventIndex;  
}

void 
Event::getByLabel(const std::type_info& iInfo,
                  const char* iModuleLabel,
                  const char* iProductInstanceLabel,
                  const char* iProcessLabel,
                  void* oData) const 
{
  if(atEnd()) {
    throw cms::Exception("OffEnd")<<"You have requested to get data after having gone passed the last event";
  }
  
  //std::cout <<iInfo.name()<<" '"<<iModuleLabel<<"' '"<< (( 0!=iProductInstanceLabel)?iProductInstanceLabel:"")<<"' '"
  //<<((0!=iProcessLabel)?iProcessLabel:"")<<"'"<<std::endl;
  void** pOData = reinterpret_cast<void**>(oData);
  *pOData = 0;
  //std::cout <<iInfo.name()<<std::endl;
  edm::TypeID type(iInfo);
  internal::DataKey key(type, iModuleLabel, iProductInstanceLabel, iProcessLabel);
  
  boost::shared_ptr<internal::Data> theData;
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
    if (0==iProcessLabel || iProcessLabel==internal::DataKey::kEmpty ||
        strlen(iProcessLabel)==0) 
    {
      const std::string* lastLabel=0;
      //have to search in reverse order since newest are on the bottom
      const edm::ProcessHistory& h = history();
      for (edm::ProcessHistory::const_reverse_iterator iproc = h.rbegin(),
	   eproc = h.rend();
           iproc != eproc;
           ++iproc) {
        lastLabel = &(iproc->processName());
        branch=findBranch(eventTree_,name,iproc->processName());
        if(0!=branch) { break; }
      }
      if(0==branch) {
        throw cms::Exception("NoBranch")<<"The file does not contain a branch beginning with '"<<name<<"'";
      }
      //do we already have this one?
      if(0!=lastLabel) {
        //std::cout <<" process name "<<*lastLabel<<std::endl;
        internal::DataKey fullKey(type,iModuleLabel,iProductInstanceLabel,lastLabel->c_str());
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
      branch = findBranch(eventTree_,name,key.process());
      if(0==branch){
        throw cms::Exception("NoBranch")<<"The file does not contain a branch named '"<<name<<key.process()<<"'";
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
    internal::DataKey newKey(edm::TypeID(iInfo),newModule,newProduct,newProcess);
    
    if(0== theData.get() ) {
      //We do not already have this data as another key
      
      //Use Reflex to create an instance of the object to be used as a buffer
      ROOT::Reflex::Type rType = ROOT::Reflex::Type::ByTypeInfo(iInfo);
      if(rType == ROOT::Reflex::Type()) {
        throw cms::Exception("UnknownType")<<"No Reflex dictionary exists for type "<<iInfo.name();
      }
      ROOT::Reflex::Object obj = rType.Construct();
      
      if(obj.Address() == 0) {
        throw cms::Exception("ConstructionFailed")<<"failed to construct an instance of "<<rType.Name();
      }
      boost::shared_ptr<internal::Data> newData(new internal::Data() );
      newData->branch_ = branch;
      newData->obj_ = obj;
      newData->lastEvent_=-1;
      newData->pObj_ = obj.Address();
      newData->pProd_=0;
      branch->SetAddress(&(newData->pObj_));
      theData = newData;
    }
    itFind = data_.insert(std::make_pair(newKey, theData)).first;
    
    if(foundProcessLabel.size()) {
      //also remember it with the process label
      newProcess = new char[foundProcessLabel.size()+1];
      std::strcpy(newProcess,foundProcessLabel.c_str());
      labels_.push_back(newProcess);
      internal::DataKey newKey(edm::TypeID(iInfo),newModule,newProduct,newProcess);

      data_.insert(std::make_pair(newKey,theData));
    }
  }
  if(eventIndex_ != itFind->second->lastEvent_) {
    //haven't gotten the data for this event
    //std::cout <<" getByLabel getting data"<<std::endl;
    getBranchData(getter_.get(), eventIndex_, *(itFind->second));
  }
  *pOData = itFind->second->obj_.Address();

}

const edm::ProcessHistory& 
Event::history() const
{
  edm::ProcessHistoryID processHistoryID;

  bool newFormat = (fileVersion_ >= 5);

  if(auxBranch_->GetEntryNumber() != eventIndex_) {
    auxBranch_->GetEntry(eventIndex_);
    //handling dealing with old version
    if(0 != pOldAux_) {
      conversion(*pOldAux_,aux_);
    }
  }
  if (!newFormat) {
    processHistoryID = aux_.processHistoryID();
  }
  if(historyMap_.empty() || newFormat) {
    TTree *meta = dynamic_cast<TTree*>(file_->Get(edm::poolNames::metaDataTreeName().c_str()));
    if(0==meta) {
      throw cms::Exception("NoMetaTree")<<"The TFile does not appear to contain a TTree named "
      <<edm::poolNames::metaDataTreeName();
    }
    if (historyMap_.empty()) {
      edm::ProcessHistoryMap* pPhm=&historyMap_;
      TBranch* b = meta->GetBranch(edm::poolNames::processHistoryMapBranchName().c_str());
      b->SetAddress(&pPhm);
      b->GetEntry(0);
    }
    if (newFormat) {
      std::vector<edm::EventProcessHistoryID> *pEventProcessHistoryIDs = &eventProcessHistoryIDs_;
      TBranch* b = meta->GetBranch(edm::poolNames::eventHistoryBranchName().c_str());
      b->SetAddress(&pEventProcessHistoryIDs);
      b->GetEntry(0);
      edm::EventProcessHistoryID target(aux_.id(), edm::ProcessHistoryID());
      processHistoryID = std::lower_bound(eventProcessHistoryIDs_.begin(), eventProcessHistoryIDs_.end(), target)->processHistoryID_;
    }
  }
  
  return historyMap_[processHistoryID];
}

edm::EDProduct const* 
Event::getByProductID(edm::ProductID const& iID) const
{
  //std::cout <<"getByProductID"<<std::endl;
  std::map<edm::ProductID,boost::shared_ptr<internal::Data> >::const_iterator itFound = idToData_.find(iID);
  if(itFound == idToData_.end() ) {
    //std::cout <<" not found"<<std::endl;
    if(0==prodReg_) {
      //std::cout <<" getting product registry"<<std::endl;
      prodReg_ = new edm::ProductRegistry;

      TTree* metaDataTree = dynamic_cast<TTree*>(file_->Get(edm::poolNames::metaDataTreeName().c_str()) );
      assert(0!=metaDataTree);

      TBranch* bReg = metaDataTree->GetBranch(edm::poolNames::productDescriptionBranchName().c_str());
      bReg->SetAddress(&prodReg_);
      bReg->GetEntry(0);
      prodReg_->setFrozen();
      
      //std::cout <<"  caching"<<std::endl;
      //cache some info
      const edm::ProductRegistry::ProductList& prodList = prodReg_->productList();
      for(edm::ProductRegistry::ProductList::const_iterator itProd = prodList.begin(),
	  itProdEnd = prodList.end();
          itProd != itProdEnd;
          ++itProd) {
        //this has to be called since 'branchName' is not stored and the 'init' method is supposed to
        // regenerate it
        itProd->second.init();
        idToBD_[itProd->second.productID()] = &(itProd->second);
      }
      //std::cout <<"  cached"<<std::endl;
    }
    //std::cout <<"  find branch description"<<std::endl;
    std::map<edm::ProductID,const edm::BranchDescription*>::iterator itCacheFind = idToBD_.find(iID);
    if(idToBD_.end() == itCacheFind) {
      throw cms::Exception("ProductNotFound")<<"No data item with productID "<<iID.id()<<" was found in the file.";
    }

    //std::cout <<"  get Type for class"<<std::endl;
    //Calculate the key from the branch description
    ROOT::Reflex::Type type( ROOT::Reflex::Type::ByName(edm::wrappedClassName(itCacheFind->second->fullClassName())));
    assert( ROOT::Reflex::Type() != type) ;

    //std::cout <<"  build key"<<std::endl;
    //Only the product instance label may be empty
    const char* pIL = itCacheFind->second->productInstanceName().c_str();
    if(pIL[0] == 0) {
      pIL = 0;
    }
    internal::DataKey k(edm::TypeID(type.TypeInfo()), 
                        itCacheFind->second->moduleLabel().c_str(),
                        pIL,
                        itCacheFind->second->processName().c_str());
    
    //has this already been gotten?
    KeyToDataMap::iterator itData = data_.find(k);
    if(data_.end() == itData) {
      //std::cout <<" calling getByLabel"<<std::endl;
      //ask for the data
      void* dummy = 0;
      getByLabel(type.TypeInfo(),
                 k.module(),
                 k.product(),
                 k.process(),
                 &dummy);
      //std::cout <<"  called"<<std::endl;
      itData = data_.find(k);
      assert(itData != data_.end());
      assert(0!=dummy);
      assert(dummy == itData->second->obj_.Address());
    }
    itFound = idToData_.insert(std::make_pair(iID,itData->second)).first;
  }
  if(eventIndex_ != itFound->second->lastEvent_) {
    //haven't gotten the data for this event
    getBranchData(getter_.get(), eventIndex_, *(itFound->second));
  }  
  if(0==itFound->second->pProd_) {
    //std::cout <<"  need to convert"<<std::endl;
    //need to convert pointer to proper type
    static ROOT::Reflex::Type sEDProd( ROOT::Reflex::Type::ByTypeInfo(typeid(edm::EDProduct)));
    //assert( sEDProd != ROOT::Reflex::Type() );
    ROOT::Reflex::Object edProdObj = itFound->second->obj_.CastObject( sEDProd );
        
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
Event::throwProductNotFoundException(const std::type_info& iType, const char* iModule, const char* iProduct, const char* iProcess)
{
    edm::TypeID type(iType);
  throw edm::Exception(edm::errors::ProductNotFound)<<"A branch was found for \n  type ='"<<type.className()<<"'\n  module='"<<iModule
    <<"'\n  productInstance='"<<((0!=iProduct)?iProduct:"")<<"'\n  process='"<<((0!=iProcess)?iProcess:"")<<"'\n"
    "but no data is available for this Event";
}
}
