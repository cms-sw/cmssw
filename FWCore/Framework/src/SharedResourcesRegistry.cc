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

#include <algorithm>
#include <cassert>

#include "FWCore/Framework/interface/SharedResourcesRegistry.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"

namespace edm {

  SharedResourcesRegistry* SharedResourcesRegistry::instance() {
    static SharedResourcesRegistry s_instance;
    return &s_instance;
  }

  SharedResourcesRegistry::SharedResourcesRegistry() {}

  void SharedResourcesRegistry::registerSharedResource(const std::string& resourceName) {
    auto& queueAndCounter = resourceMap_[resourceName];

    // count the number of times the resource was registered
    ++queueAndCounter.second;

    // If registering a resource the second time
    // we know we will need the queue so go ahead and create it.
    if (queueAndCounter.second == 2) {
      queueAndCounter.first = std::make_shared<SerialTaskQueue>();
    }
  }

  std::pair<SharedResourcesAcquirer, std::shared_ptr<std::recursive_mutex>>
  SharedResourcesRegistry::createAcquirerForSourceDelayedReader() {
    if (not resourceForDelayedReader_) {
      resourceForDelayedReader_ =
          std::make_shared<std::recursive_mutex>();  // propagate_const<T> has no reset() function
      queueForDelayedReader_ = std::make_shared<SerialTaskQueue>();
    }

    std::vector<std::shared_ptr<SerialTaskQueue>> queues = {get_underlying(queueForDelayedReader_)};
    return std::make_pair(SharedResourcesAcquirer(std::move(queues)), get_underlying(resourceForDelayedReader_));
  }

  SharedResourcesAcquirer SharedResourcesRegistry::createAcquirer(std::vector<std::string> const& resourceNames) const {
    // The acquirer will acquire the shared resources declared by a module
    // so that only it can use those resources while it runs. The other
    // modules using the same resource will not be run until the module
    // that acquired the resources completes its task.

    // Sort by how often used and then by name
    // Consistent sorting avoids deadlocks and this particular order optimizes performance
    std::map<std::pair<unsigned int, std::string>, std::shared_ptr<SerialTaskQueue>> sortedResources;

    for (auto const& name : resourceNames) {
      auto resource = resourceMap_.find(name);
      assert(resource != resourceMap_.end());
      // If only one module wants it, it really isn't shared
      if (resource->second.second > 1) {
        sortedResources.insert(
            std::make_pair(std::make_pair(resource->second.second, resource->first), resource->second.first));
      }
    }

    std::vector<std::shared_ptr<SerialTaskQueue>> queues;
    queues.reserve(sortedResources.size());
    for (auto const& resource : sortedResources) {
      queues.push_back(resource.second);
    }
    if (queues.empty()) {
      //Calling code is depending on there being at least one shared queue
      queues.reserve(1);
      queues.push_back(std::make_shared<SerialTaskQueue>());
    }

    return SharedResourcesAcquirer(std::move(queues));
  }
}  // namespace edm
