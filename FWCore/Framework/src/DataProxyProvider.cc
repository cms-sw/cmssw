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
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/DataProxyProvider.h"
#include "FWCore/Framework/interface/DataProxy.h"
#include "FWCore/Framework/interface/RecordDependencyRegister.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include <cassert>

namespace edm {
   namespace eventsetup {

DataProxyProvider::DataProxyProvider()
{
}

DataProxyProvider::~DataProxyProvider() noexcept(false)
{
}

void DataProxyProvider::updateLookup(eventsetup::ESRecordsToProxyIndices const&) {}

void 
DataProxyProvider::usingRecordWithKey(const EventSetupRecordKey& iKey)
{
   recordProxies_[iKey];
}

void
DataProxyProvider::fillRecordsNotAllowingConcurrentIOVs(std::set<EventSetupRecordKey>& recordsNotAllowingConcurrentIOVs) const {
  for (auto const& it : recordProxies_) {
    const EventSetupRecordKey& key = it.first;
    if (!allowConcurrentIOVs(key)) {
      recordsNotAllowingConcurrentIOVs.insert(recordsNotAllowingConcurrentIOVs.end(), key);
    }
  }
}

void
DataProxyProvider::resizeKeyedProxiesVector(EventSetupRecordKey const& key, unsigned int nConcurrentIOVs) {
  recordProxies_[key].resize(nConcurrentIOVs);
}

void
DataProxyProvider::setAppendToDataLabel(const edm::ParameterSet& iToAppend)
{
  std::string oldValue( appendToDataLabel_);
  //this can only be changed once and the default value is the empty string
  assert(oldValue.empty());
  
  const std::string kParamName("appendToDataLabel");
  if(iToAppend.exists(kParamName) ) {    
    appendToDataLabel_ = iToAppend.getParameter<std::string>(kParamName);
  }
}

bool 
DataProxyProvider::isUsingRecord(const EventSetupRecordKey& iKey) const
{
   return recordProxies_.end() != recordProxies_.find(iKey);
}

std::set<EventSetupRecordKey>
DataProxyProvider::usingRecords() const
{
   std::set<EventSetupRecordKey> returnValue;
   for (auto const& it : recordProxies_) {
      returnValue.insert(returnValue.end(), it.first);
   }
   return returnValue;
}

DataProxyProvider::KeyedProxies&
DataProxyProvider::keyedProxies(const EventSetupRecordKey& iRecordKey, unsigned int iovIndex)
{
   RecordProxies::iterator itFind = recordProxies_.find(iRecordKey);
   assert(itFind != recordProxies_.end());
   assert(iovIndex < itFind->second.size());
   KeyedProxies& keyedProxies = itFind->second[iovIndex];

   if (keyedProxies.empty()) {
      //delayed registration
      registerProxies(iRecordKey, keyedProxies, iovIndex);

      bool mustChangeLabels = (!appendToDataLabel_.empty());
      for (auto& itProxy : keyedProxies) {
         itProxy.second->setProviderDescription(&description());
         if (mustChangeLabels) {
            //Using swap is fine since
            // 1) the data structure is not a map and so we have not sorted on the keys
            // 2) this is the first time filling this so no outside agency has yet seen
            //   the label and therefore can not be dependent upon its value
            std::string temp(std::string(itProxy.first.name().value()) + appendToDataLabel_);
            DataKey newKey(itProxy.first.type(), temp.c_str());
            swap(itProxy.first, newKey);
         }
      }
   }
   return keyedProxies;
}

static const std::string kAppendToDataLabel("appendToDataLabel");

void
DataProxyProvider::prevalidate(ConfigurationDescriptions& iDesc)
{
   if(iDesc.defaultDescription()) {
     if (iDesc.defaultDescription()->isLabelUnused(kAppendToDataLabel)) {
       iDesc.defaultDescription()->add<std::string>(kAppendToDataLabel, std::string(""));
     }
   }
   for(auto& v: iDesc) {
     if (v.second.isLabelUnused(kAppendToDataLabel)) {
       v.second.add<std::string>(kAppendToDataLabel, std::string(""));
     }
   }
}

   }
}
