// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     SharedResourcesRegistry
// 
// Implementation:
//     [Notes on implementation]
//
// Original Author:  Chris Jones
//         Created:  Sun, 06 Oct 2013 15:48:50 GMT
//

// system include files
#include <cassert>

// user include files
#include "SharedResourcesRegistry.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"

namespace edm {

  const std::string SharedResourcesRegistry::kLegacyModuleResourceName{"__legacy__"};
  
  SharedResourcesRegistry*
  SharedResourcesRegistry::instance() {
    static SharedResourcesRegistry s_instance;
    return &s_instance;
  }
  
  void
  SharedResourcesRegistry::registerSharedResource(const std::string& iString){
    auto& values = resourceMap_[iString];
    
    if(values.second ==1) {
      //only need to make the resource if more than 1 module wants it
      values.first = std::shared_ptr<std::recursive_mutex>( new std::recursive_mutex );
    }
    ++(values.second);
  }
  
  SharedResourcesAcquirer
  SharedResourcesRegistry::createAcquirerForSourceDelayedReader() {
    if(not resourceForDelayedReader_) {
      resourceForDelayedReader_.reset(new std::recursive_mutex{});
    }
    std::vector<std::recursive_mutex*> mutexes = {resourceForDelayedReader_.get()};

    return SharedResourcesAcquirer(std::move(mutexes));
  }

  
  SharedResourcesAcquirer
  SharedResourcesRegistry::createAcquirer(std::vector<std::string> const &  iNames) const {
    //Sort by how often used and then by name
    std::map<std::pair<unsigned int, std::string>, std::recursive_mutex*> sortedResources;
    
    for(auto const& name: iNames) {
      auto itFound = resourceMap_.find(name);
      assert(itFound != resourceMap_.end());
      //If only one module wants it, it really isn't shared
      if(itFound->second.second>1) {
        sortedResources.insert(std::make_pair(std::make_pair(itFound->second.second, itFound->first),itFound->second.first.get()));
      }
    }
    std::vector<std::recursive_mutex*> mutexes;
    mutexes.reserve(sortedResources.size());
    for(auto const& resource: sortedResources) {
      mutexes.push_back(resource.second);
    }
    
    return SharedResourcesAcquirer(std::move(mutexes));
  }
  
}
