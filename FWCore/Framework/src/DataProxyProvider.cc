// -*- C++ -*-
//
// Package:     Framework
// Class  :     DataProxyProvider
// 
// Implementation:
//     <Notes on implementation>
//
// Author:      Chris Jones
// Created:     Mon Mar 28 15:07:54 EST 2005
//

// system include files

// user include files
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

namespace edm {
   namespace eventsetup {
//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
DataProxyProvider::DataProxyProvider() : recordProxies_(), description_()
{
}

// DataProxyProvider::DataProxyProvider(const DataProxyProvider& rhs)
// {
//    // do actual copying here;
// }

DataProxyProvider::~DataProxyProvider()
{
}

//
// assignment operators
//
// const DataProxyProvider& DataProxyProvider::operator=(const DataProxyProvider& rhs)
// {
//   //An exception safe implementation is
//   DataProxyProvider temp(rhs);
//   swap(rhs);
//
//   return *this;
// }

//
// member functions
//
void 
DataProxyProvider::usingRecordWithKey(const EventSetupRecordKey& iKey)
{
   recordProxies_[iKey];
   //keys_.push_back(iKey);
}

void 
DataProxyProvider::invalidateProxies(const EventSetupRecordKey& iRecordKey) 
{
   KeyedProxies& proxyList((*(recordProxies_.find(iRecordKey))).second) ;
   KeyedProxies::iterator finished(proxyList.end()) ;
   for (KeyedProxies::iterator keyedProxy(proxyList.begin()) ;
         keyedProxy != finished ;
         ++keyedProxy) {
      (*((*keyedProxy).second)).invalidate() ;
   }
   
}

void 
DataProxyProvider::resetProxies(const EventSetupRecordKey& iRecordKey)
{
  invalidateProxies(iRecordKey);
}
      
void 
DataProxyProvider::resetProxiesIfTransient(const EventSetupRecordKey& iRecordKey) 
{
   KeyedProxies& proxyList((*(recordProxies_.find(iRecordKey))).second) ;
   KeyedProxies::iterator finished(proxyList.end()) ;
   for (KeyedProxies::iterator keyedProxy(proxyList.begin()) ;
        keyedProxy != finished ;
        ++keyedProxy) {
      (*((*keyedProxy).second)).resetIfTransient() ;
   }
   
}
      
void
DataProxyProvider::setAppendToDataLabel(const edm::ParameterSet& iToAppend)
{
  std::string oldValue( appendToDataLabel_);
  //this can only be changed once and the default value is the empty string
  assert(0 == oldValue.size());
  
  const std::string kParamName("appendToDataLabel");
  if(iToAppend.exists(kParamName) ) {    
    appendToDataLabel_ = iToAppend.getParameter<std::string>(kParamName);
  }
}
//
// const member functions
//
bool 
DataProxyProvider::isUsingRecord(const EventSetupRecordKey& iKey) const
{
   return recordProxies_.end() != recordProxies_.find(iKey);
}

std::set<EventSetupRecordKey> 
DataProxyProvider::usingRecords() const
{
   std::set<EventSetupRecordKey> returnValue;
   for(RecordProxies::const_iterator itRecProxies = recordProxies_.begin(),
        itRecProxiesEnd = recordProxies_.end();
        itRecProxies != itRecProxiesEnd;
        ++itRecProxies) {
      returnValue.insert(returnValue.end(), itRecProxies->first);
   }
   //copy_all(keys_, std::inserter(returnValue, returnValue.end()));
   return returnValue;
}   

const DataProxyProvider::KeyedProxies& 
DataProxyProvider::keyedProxies(const EventSetupRecordKey& iRecordKey) const
{
   RecordProxies::const_iterator itFind = recordProxies_.find(iRecordKey);
   assert(itFind != recordProxies_.end());
   
   if(itFind->second.empty()) {
      //delayed registration
      KeyedProxies& proxies = const_cast<KeyedProxies&>(itFind->second);
      const_cast<DataProxyProvider*>(this)->registerProxies(iRecordKey,
                                                            proxies);

      bool mustChangeLabels = (0 != appendToDataLabel_.size());
      for(KeyedProxies::iterator itProxy = proxies.begin(), itProxyEnd = proxies.end();
          itProxy != itProxyEnd;
          ++itProxy) {
        itProxy->second->setProviderDescription(&description());
        if( mustChangeLabels ) {
          //Using swap is fine since
          // 1) the data structure is not a map and so we have not sorted on the keys
          // 2) this is the first time filling this so no outside agency has yet seen
          //   the label and therefore can not be dependent upon its value
          std::string temp(std::string(itProxy->first.name().value())+appendToDataLabel_);
          DataKey newKey(itProxy->first.type(),temp.c_str());
          swap(itProxy->first,newKey);
        }
      }
   }
   
   return itFind->second;
}

//
// static member functions
//
   }
}

