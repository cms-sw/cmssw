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
#include "FWCore/Framework/interface/SharedResourcesRegistry.h"
#include "FWCore/Framework/interface/SharedResourcesAcquirer.h"

namespace edm {

  const std::string SharedResourcesRegistry::kLegacyModuleResourceName{"__legacy__"};

  SharedResourcesRegistry* SharedResourcesRegistry::instance() {
    static SharedResourcesRegistry s_instance;
    return &s_instance;
  }

  SharedResourcesRegistry::SharedResourcesRegistry() : nLegacy_(0) {}

  void SharedResourcesRegistry::registerSharedResource(const std::string& resourceName) {
    auto& queueAndCounter = resourceMap_[resourceName];

    if (resourceName == kLegacyModuleResourceName) {
      ++nLegacy_;
      for (auto& resource : resourceMap_) {
        if (!resource.second.first) {
          resource.second.first = std::make_shared<SerialTaskQueue>();
        }
        ++resource.second.second;
      }
    } else {
      // count the number of times the resource was registered
      ++queueAndCounter.second;

      // When first registering a nonlegacy resource, we have to
      // account for any legacy resource registrations already made.
      if (queueAndCounter.second == 1) {
        if (nLegacy_ > 0U) {
          queueAndCounter.first = std::make_shared<SerialTaskQueue>();
          queueAndCounter.second += nLegacy_;
        }
        // If registering a nonlegacy resource the second time and
        // the legacy resource has not been registered yet,
        // we know we will need the queue so go ahead and create it.
      } else if (queueAndCounter.second == 2) {
        queueAndCounter.first = std::make_shared<SerialTaskQueue>();
      }
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

    // The legacy shared resource is special.
    // Legacy modules cannot run concurrently with each other or
    // any other module that has declared any shared resource. Treat
    // one modules that call usesResource with no argument in the
    // same way.

    // Sort by how often used and then by name
    // Consistent sorting avoids deadlocks and this particular order optimizes performance
    std::map<std::pair<unsigned int, std::string>, std::shared_ptr<SerialTaskQueue>> sortedResources;

    // Is this acquirer for a module that depends on the legacy shared resource?
    if (std::find(resourceNames.begin(), resourceNames.end(), kLegacyModuleResourceName) != resourceNames.end()) {
      for (auto const& resource : resourceMap_) {
        // It's redundant to declare legacy if the legacy modules
        // all declare all the other resources, so just skip it.
        // But if the only shared resource is the legacy resource don't skip it.
        if (resource.first == kLegacyModuleResourceName && resourceMap_.size() > 1)
          continue;
        //legacy modules are not allowed to depend on ES shared resources
        if (resource.first.substr(0, 3) == "es_")
          continue;
        //If only one module wants it, it really isn't shared
        if (resource.second.second > 1) {
          sortedResources.insert(
              std::make_pair(std::make_pair(resource.second.second, resource.first), resource.second.first));
        }
      }
      // Handle cases where the module does not declare the legacy resource
    } else {
      for (auto const& name : resourceNames) {
        auto resource = resourceMap_.find(name);
        assert(resource != resourceMap_.end());
        //If only one module wants it, it really isn't shared
        if (resource->second.second > 1) {
          sortedResources.insert(
              std::make_pair(std::make_pair(resource->second.second, resource->first), resource->second.first));
        }
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
