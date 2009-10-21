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
// $Id: Event.cc,v 1.29 2009/09/04 21:34:20 wdd Exp $
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
#include "DataFormats/Provenance/interface/History.h"

#include "FWCore/FWLite/interface/setRefStreamer.h"

#include "FWCore/Utilities/interface/WrappedClassName.h"

#include "DataFormats/Provenance/interface/ParameterSetBlob.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "FWCore/ParameterSet/interface/ParameterSetConverter.h"

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
  // empty object used to signal that the branch requested was not found
  static internal::Data branchNotFound;

//
// constructors and destructor
//
  Event::Event(TFile* iFile):
//  file_(iFile),
//  eventTree_(0),
  eventHistoryTree_(0),
//  eventIndex_(-1),
  branchMap_(iFile),
  pAux_(&aux_),
  pOldAux_(0),
  fileVersion_(-1),
  parameterSetRegistryFilled_(false)
{
    if(0==iFile) {
      throw cms::Exception("NoFile")<<"The TFile pointer passed to the constructor was null";
    }
    
    if(0==branchMap_.getEventTree()) {
      throw cms::Exception("NoEventTree")<<"The TFile contains no TTree named "<<edm::poolNames::eventTreeName();
    }
    //need to know file version in order to determine how to read the basic event info
    fileVersion_ = branchMap_.getFileVersion(iFile);

    //got this logic from IOPool/Input/src/RootFile.cc
    
    TTree* eventTree = branchMap_.getEventTree();
    if(fileVersion_ >= 3 ) {
      auxBranch_ = eventTree->GetBranch(edm::BranchTypeToAuxiliaryBranchName(edm::InEvent).c_str());
      if(0==auxBranch_) {
        throw cms::Exception("NoEventAuxilliary")<<"The TTree "
        <<edm::poolNames::eventTreeName()
        <<" does not contain a branch named 'EventAuxiliary'";
      }
      auxBranch_->SetAddress(&pAux_);
    } else {
      pOldAux_ = new edm::EventAux();
      auxBranch_ = eventTree->GetBranch(edm::BranchTypeToAuxBranchName(edm::InEvent).c_str());
      if(0==auxBranch_) {
        throw cms::Exception("NoEventAux")<<"The TTree "
          <<edm::poolNames::eventTreeName()
          <<" does not contain a branch named 'EventAux'";
      }
      auxBranch_->SetAddress(&pOldAux_);
    }
    branchMap_.updateEvent(0);

    if(fileVersion_ >= 7 ) {
      eventHistoryTree_ = dynamic_cast<TTree*>(iFile->Get(edm::poolNames::eventHistoryTreeName().c_str()));
    }
    
    getter_ = boost::shared_ptr<edm::EDProductGetter>(new internal::ProductGetter(this));
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
   Long_t eventIndex = branchMap_.getEventEntry();
   if(eventIndex < size()) 
   {
      branchMap_.updateEvent(++eventIndex);
   }
   return *this;
}

bool
Event::to (Long64_t iEntry)
{
   if (iEntry < size())
   {
      // this is a valid entry
      return branchMap_.updateEvent(iEntry);
   }
   // if we're here, then iEntry was not valid
   return false;
}

bool
Event::to (edm::RunNumber_t run, edm::EventNumber_t event)
{
   fillFileIndex();
   edm::FileIndex::const_iterator i = 
      fileIndex_.findEventPosition(run, 0, event, true);
   if (fileIndex_.end() != i) 
   {
      return branchMap_.updateEvent(i->entry_);
   }
   return false;
}

bool
Event::to (const edm::EventID &id)
{
   return to (id.run(), id.event());
}

void
Event::fillFileIndex() const
{
  if (fileIndex_.empty()) {
    TTree* meta = dynamic_cast<TTree*>(branchMap_.getFile()->Get(edm::poolNames::metaDataTreeName().c_str()));
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
const Event& 
Event::toBegin()
{
   branchMap_.updateEvent(0);
   return *this;
}

//
// const member functions
//
Long64_t
Event::size() const 
{
  return branchMap_.getEventTree()->GetEntries();
}

bool
Event::isValid() const
{
  Long_t eventIndex = branchMap_.getEventEntry();
  return eventIndex!=-1 and eventIndex < size(); 
}


Event::operator bool() const
{
  return isValid();
}

bool
Event::atEnd() const
{
  Long_t eventIndex = branchMap_.getEventEntry();
  return eventIndex==-1 or eventIndex == size();
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
  
  iData.branch_->GetEntry(iEventIndex);
  iData.lastEvent_=iEventIndex;  
}

const std::vector<std::string>&
Event::getProcessHistory() const
{
  if (procHistoryNames_.empty()) {
    // std::cout << "Getting new process history" << std::endl;
    const edm::ProcessHistory& h = history();
    for (edm::ProcessHistory::const_iterator iproc = h.begin(), eproc = h.end();
         iproc != eproc; ++iproc) {
      procHistoryNames_.push_back(iproc->processName());
      // std::cout << iproc->processName() << std::endl;
    }
  }
  return procHistoryNames_;
}

internal::Data&
Event::getBranchDataFor(const std::type_info& iInfo,
                  const char* iModuleLabel,
                  const char* iProductInstanceLabel,
                  const char* iProcessLabel) const
{
  //std::cout <<iInfo.name()<<" '"<<iModuleLabel<<"' '"<< (( 0!=iProductInstanceLabel)?iProductInstanceLabel:"")<<"' '"
  //<<((0!=iProcessLabel)?iProcessLabel:"")<<"'"<<std::endl;
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
    TTree* eventTree = branchMap_.getEventTree();
    if (0==iProcessLabel || iProcessLabel==internal::DataKey::kEmpty ||
        strlen(iProcessLabel)==0) 
    {
      const std::string* lastLabel=0;
      //have to search in reverse order since newest are on the bottom
      const edm::ProcessHistory& h = history();
      for (edm::ProcessHistory::const_reverse_iterator iproc = h.rbegin(), eproc = h.rend();
           iproc != eproc;
           ++iproc) {
        lastLabel = &(iproc->processName());
        branch=findBranch(eventTree,name,iproc->processName());
        if(0!=branch) { break; }
      }
      if(0==branch) {
        return branchNotFound;
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
      branch = findBranch(eventTree,name,key.process());
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
    internal::DataKey newKey(edm::TypeID(iInfo),newModule,newProduct,newProcess);
    
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
      boost::shared_ptr<internal::Data> newData(new internal::Data() );
      newData->branch_ = branch;
      newData->obj_ = obj;
      newData->lastEvent_=-1;
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
      internal::DataKey newKey(edm::TypeID(iInfo),newModule,newProduct,newProcess);

      data_.insert(std::make_pair(newKey,theData));
    }
  }
  return *(itFind->second);
}

const std::string 
Event::getBranchNameFor(const std::type_info& iInfo,
                  const char* iModuleLabel,
                  const char* iProductInstanceLabel,
                  const char* iProcessLabel) const
{
  internal::Data& theData = 
    Event::getBranchDataFor(iInfo, iModuleLabel, iProductInstanceLabel, iProcessLabel);

  if (0 != theData.branch_) {
    return std::string(theData.branch_->GetName());
  }
  return std::string("");
}

bool 
Event::getByLabel(const std::type_info& iInfo,
                  const char* iModuleLabel,
                  const char* iProductInstanceLabel,
                  const char* iProcessLabel,
                  void* oData) const 
{
  if(atEnd()) {
    throw cms::Exception("OffEnd")<<"You have requested data past the last event";
  }
  void** pOData = reinterpret_cast<void**>(oData);
  *pOData = 0;


  internal::Data& theData = 
    Event::getBranchDataFor(iInfo, iModuleLabel, iProductInstanceLabel, iProcessLabel);
                      
  if (0 != theData.branch_) {
    Long_t eventIndex = branchMap_.getEventEntry();
    if(eventIndex != theData.lastEvent_) {
      //haven't gotten the data for this event
      //std::cout <<" getByLabel getting data"<<std::endl;
      getBranchData(getter_.get(), eventIndex, theData);
    }
    *pOData = theData.obj_.Address();
  }
  if ( 0 == *pOData ) return false;
  else return true;
}

edm::EventAuxiliary const& 
Event::eventAuxiliary() const
{
   Long_t eventIndex = branchMap_.getEventEntry();
   updateAux(eventIndex);
   return aux_;
}

void
Event::updateAux(Long_t eventIndex) const
{
  if(auxBranch_->GetEntryNumber() != eventIndex) {
    auxBranch_->GetEntry(eventIndex);
    //handling dealing with old version
    if(0 != pOldAux_) {
      conversion(*pOldAux_,aux_);
    }
  }
}

const edm::ProcessHistory& 
Event::history() const
{
  edm::ProcessHistoryID processHistoryID;

  bool newFormat = (fileVersion_ >= 5);

  Long_t eventIndex = branchMap_.getEventEntry();
  updateAux(eventIndex);
  if (!newFormat) {
    processHistoryID = aux_.processHistoryID();
  }
  if(historyMap_.empty() || newFormat) {
    procHistoryNames_.clear();
    TTree *meta = dynamic_cast<TTree*>(branchMap_.getFile()->Get(edm::poolNames::metaDataTreeName().c_str()));
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
    if (newFormat) {
      if (fileVersion_ >= 7) {
        edm::History history;
        edm::History* pHistory = &history;
        TBranch* eventHistoryBranch = eventHistoryTree_->GetBranch(edm::poolNames::eventHistoryBranchName().c_str());
        if (!eventHistoryBranch)
          throw edm::Exception(edm::errors::FatalRootError)
            << "Failed to find history branch in event history tree";
        eventHistoryBranch->SetAddress(&pHistory);
        eventHistoryTree_->GetEntry(eventIndex);
        processHistoryID = history.processHistoryID();
      } else {
        std::vector<edm::EventProcessHistoryID> *pEventProcessHistoryIDs = &eventProcessHistoryIDs_;
        TBranch* b = meta->GetBranch(edm::poolNames::eventHistoryBranchName().c_str());
        b->SetAddress(&pEventProcessHistoryIDs);
        b->GetEntry(0);
        edm::EventProcessHistoryID target(aux_.id(), edm::ProcessHistoryID());
        processHistoryID = std::lower_bound(eventProcessHistoryIDs_.begin(), eventProcessHistoryIDs_.end(), target)->processHistoryID_;
      } 
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
    edm::BranchDescription bDesc = branchMap_.productToBranch(iID);

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
    internal::DataKey k(edm::TypeID(type.TypeInfo()), 
                        bDesc.moduleLabel().c_str(),
                        pIL,
                        bDesc.processName().c_str());
    
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
  Long_t eventIndex = branchMap_.getEventEntry();
  if(eventIndex != itFound->second->lastEvent_) {
    //haven't gotten the data for this event
    getBranchData(getter_.get(), eventIndex, *(itFound->second));
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

TriggerNames const&
Event::triggerNames(edm::TriggerResults const& triggerResults)
{
  TriggerNames const* names = triggerNames_(triggerResults);
  if (names != 0) return *names;

  if (!parameterSetRegistryFilled_) {
    fillParameterSetRegistry();
    names = triggerNames_(triggerResults);
  }
  if (names != 0) return *names;

  throw cms::Exception("TriggerNamesNotFound")
    << "TriggerNames not found in ParameterSet registry";
  return *names;
}

void
Event::fillParameterSetRegistry()
{
  if (parameterSetRegistryFilled_) return;
  parameterSetRegistryFilled_ = true;

  TTree* meta = dynamic_cast<TTree*>(branchMap_.getFile()->Get(edm::poolNames::metaDataTreeName().c_str()));
  if (0==meta) {
    throw cms::Exception("NoMetaTree") << "The TFile does not contain a TTree named "
      << edm::poolNames::metaDataTreeName();
  }

  edm::FileFormatVersion fileFormatVersion;  
  edm::FileFormatVersion *fftPtr = &fileFormatVersion;
  if(meta->FindBranch(edm::poolNames::fileFormatVersionBranchName().c_str()) != 0) {
    TBranch *fft = meta->GetBranch(edm::poolNames::fileFormatVersionBranchName().c_str());
    fft->SetAddress(&fftPtr);
    fft->GetEntry(0);
  }

  if (meta->FindBranch(edm::poolNames::parameterSetMapBranchName().c_str()) != 0) {
    typedef std::map<edm::ParameterSetID, edm::ParameterSetBlob> PsetMap;
    PsetMap psetMap;
    PsetMap *psetMapPtr = &psetMap;
    TBranch* b = meta->GetBranch(edm::poolNames::parameterSetMapBranchName().c_str());
    b->SetAddress(&psetMapPtr);
    b->GetEntry(0);

    edm::ParameterSetConverter::ParameterSetIdConverter psetIdConverter;
    if(!fileFormatVersion.triggerPathsTracked()) {
      edm::ParameterSetConverter converter(psetMap, psetIdConverter, fileFormatVersion.parameterSetsByReference());
    } else {
      // Merge into the parameter set registry.
      edm::pset::Registry& psetRegistry = *edm::pset::Registry::instance();
      for(PsetMap::const_iterator i = psetMap.begin(), iEnd = psetMap.end();
          i != iEnd; ++i) {
        edm::ParameterSet pset(i->second.pset_);
        pset.setID(i->first);
        pset.setFullyTracked();
        psetRegistry.insertMapped(pset);
      } 
    }
  }
  else {
    throw cms::Exception("NoParameterSetMapBranch")
      << "The TTree does not contain a TBranch named "
      << edm::poolNames::parameterSetMapBranchName();
  }
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
