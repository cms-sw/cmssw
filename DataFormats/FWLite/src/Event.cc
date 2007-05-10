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
// $Id$
//

// system include files
#include <iostream>
#include "Reflex/Type.h"
#include "Reflex/Object.h"

// user include files
#include "DataFormats/FWLite/interface/Event.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Provenance/interface/BranchType.h"

//
// constants, enums and typedefs
//
namespace fwlite {
//
// static data member definitions
//
  namespace internal {
    const char* const DataKey::kEmpty="";
    
  }
  typedef std::map<internal::DataKey, internal::Data> DataMap;

//
// constructors and destructor
//
  Event::Event(TFile* iFile):
  file_(iFile),
  eventTree_(0),
  eventIndex_(-1),
  pAux_(&aux_)
{
    if(0==file_) {
      throw cms::Exception("NoFile")<<"The TFile pointer passed to the constructor was null";
    }
    
    eventTree_ = dynamic_cast<TTree*>(iFile->Get(edm::poolNames::eventTreeName().c_str()));
    if(0==eventTree_) {
      throw cms::Exception("NoEventTree")<<"The TFile contains no TTree named "<<edm::poolNames::eventTreeName();
    }
    auxBranch_ = eventTree_->GetBranch("EventAuxiliary");
    if(0==auxBranch_) {
      throw cms::Exception("NoEventAuxilliary")<<"The TTree "
      <<edm::poolNames::eventTreeName()
      <<" does not contain a branch named 'EventAuxiliary'";
    }
    auxBranch_->SetAddress(&pAux_);
    eventTree_->GetEntry();
    eventIndex_=0;
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
  for(DataMap::iterator it=data_.begin(), itEnd=data_.end();
      it != itEnd;
      ++it) {
    it->second.obj_.Destruct();
  //  ROOT::Reflex::Type::ByName(it->first.typeID().className()).Destruct(it->second.data_);
  }
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
  branchName+=".obj";
  return iTree->GetBranch(branchName.c_str());
}
void 
Event::getByLabel(const std::type_info& iInfo,
                  const char* iModuleLabel,
                  const char* iProductInstanceLabel,
                  const char* iProcessLabel,
                  void*& oData) const 
{
  oData = 0;
  std::cout <<iInfo.name()<<std::endl;
  edm::TypeID type(iInfo);
  internal::DataKey key(type, iModuleLabel, iProductInstanceLabel, iProcessLabel);
  
  DataMap::iterator itFind = data_.find(key);
  if(itFind == data_.end() ) {
    //see if such a branch actually exists
    const std::string sep("_");
    //CHANGE: Need to lookup the the friendly name which was used to write the file
    std::string name(type.friendlyClassName());
    name +=sep+std::string(key.module());
    name +=sep+std::string(key.product())+sep;
    TBranch* branch = 0;
    if (0==iProcessLabel || iProcessLabel==internal::DataKey::kEmpty ||
        strlen(iProcessLabel)==0) 
    {
      //have to search in reverse order since newest are on the bottom
      const edm::ProcessHistory& h = history();
      for (edm::ProcessHistory::const_reverse_iterator iproc = h.rbegin(),
	   eproc = h.rend();
           iproc != eproc;
           ++iproc) {
        branch=findBranch(eventTree_,name,iproc->processName());
        if(0!=branch) { break; }
      }
      if(0==branch) {
        throw cms::Exception("NoBranch")<<"The file does not contain a branch beginning with '"<<name<<"'";
      }
    }else {
      //we have all the pieces
      branch = findBranch(eventTree_,name,key.process());
      if(0==branch){
        throw cms::Exception("NoBranch")<<"The file does not contain a branch named '"<<name<<key.process()<<"'";
      }
    }
    //Use Reflex to create an instance of the object to be used as a buffer
    ROOT::Reflex::Type rType = ROOT::Reflex::Type::ByTypeInfo(iInfo);
    if(rType == ROOT::Reflex::Type()) {
      throw cms::Exception("UnknownType")<<"No Reflex dictionary exists for type "<<iInfo.name();
    }
    ROOT::Reflex::Object obj = rType.Construct();
    
    if(obj.Address() == 0) {
      throw cms::Exception("ConstructionFailed")<<"failed to construct an instance of "<<rType.Name();
    }
    //cache the info
    char* newModule = new char[strlen(iModuleLabel)+1];
    std::strcpy(newModule,iModuleLabel);
    labels_.push_back(newModule);
    
    char* newProduct = const_cast<char*>(key.product());
    if(newProduct[0] != 0) {
      std::strcpy(newProduct,key.product());
      labels_.push_back(newProduct);
    }
    char* newProcess = const_cast<char*>(key.process());
    if(newProcess[0]!=0) {
      std::strcpy(newProcess,key.process());
      labels_.push_back(newProcess);
    }
    internal::DataKey newKey(edm::TypeID(iInfo),newModule,newProduct,newProcess);
    internal::Data newData;
    newData.branch_ = branch;
    newData.obj_ = obj;
    newData.lastEvent_=-1;
    itFind = data_.insert(std::make_pair(newKey, newData)).first;
    branch->SetAddress(itFind->second.obj_.Address());
  }
  if(eventIndex_ != itFind->second.lastEvent_) {
    //haven't gotten the data for this event
    itFind->second.branch_->GetEntry(eventIndex_);
    itFind->second.lastEvent_=eventIndex_;
  }
  oData = itFind->second.obj_.Address();
}


const edm::ProcessHistory& 
Event::history() const
{
  if(historyMap_.empty()) {
    TTree* meta = dynamic_cast<TTree*>(file_->Get(edm::poolNames::metaDataTreeName().c_str()));
    if(0==meta) {
      throw cms::Exception("NoMetaTree")<<"The TFile does not appear to contain a TTree named "
      <<edm::poolNames::metaDataTreeName();
    }

    edm::ProcessHistoryMap* pPhm=&historyMap_;
    TBranch* b = meta->GetBranch(edm::poolNames::processHistoryMapBranchName().c_str());
    b->SetAddress(&pPhm);
    b->GetEntry(0);
  }

  if(auxBranch_->GetEntryNumber() != eventIndex_) {
    auxBranch_->GetEntry(eventIndex_);
  }
  
  return historyMap_[aux_.processHistoryID()];
}

//
// static member functions
//
}
