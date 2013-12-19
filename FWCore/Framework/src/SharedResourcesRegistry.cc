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
#include <algorithm>
#include <cassert>

// user include files
#include "SharedResourcesRegistry.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"

namespace edm {

  const std::string SharedResourcesRegistry::kLegacyModuleResourceName{"__legacy__"};

  SharedResourcesRegistry::SharedResourcesRegistry() : legacyRegistered_(false) {
  }

  SharedResourcesRegistry*
  SharedResourcesRegistry::instance() {
    static SharedResourcesRegistry s_instance;
    return &s_instance;
  }
  
  void
  SharedResourcesRegistry::registerSharedResource(const std::string& iString){
    if(iString == kLegacyModuleResourceName) {
      if(!legacyRegistered_) {
        legacyRegistered_ = true;
        for(auto & resource : resourceMap_) {
          if(!resource.second.first) {
            resource.second.first.reset(new std::recursive_mutex);
          }
        }
      }
    } else {
      if(resourceMap_.find(iString) == resourceMap_.end()) {
        nonLegacyResources_.push_back(iString);
      }
    }

    auto& values = resourceMap_[iString];

    if(values.second == 1 || legacyRegistered_) {
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
  SharedResourcesRegistry::createAcquirer(std::vector<std::string> const& iNames) const {

    std::vector<std::string> const* theNames = &iNames;

    // Do not let legacy modules run concurrently with each other or
    // any other module that has declared a shared resource.  To accomplish
    // this, we say the legacy modules declare they depend on
    // all other declared shared resources. In the special case where
    // the only resource ever declared is legacy, then just let them all
    // declare that.
    if(legacyRegistered_ &&
       std::find(iNames.begin(), iNames.end(), kLegacyModuleResourceName) != iNames.end() &&
       !nonLegacyResources_.empty()) {

      theNames = &nonLegacyResources_;
    }

    //Sort by how often used and then by name
    std::map<std::pair<unsigned int, std::string>, std::recursive_mutex*> sortedResources;
    
    for(auto const& name: *theNames) {
      auto itFound = resourceMap_.find(name);
      assert(itFound != resourceMap_.end());
      //If only one module wants it, it really isn't shared
      if(itFound->second.second > 1 || legacyRegistered_) {
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
