#ifndef FWCore_Framework_SharedResourcesRegistry_h
#define FWCore_Framework_SharedResourcesRegistry_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     SharedResourcesRegistry
// 
/**\class SharedResourcesRegistry SharedResourcesRegistry.h "SharedResourcesRegistry.h"

 Description: Manages the Acquirers used to take temporary control of a resource shared between modules

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sun, 06 Oct 2013 15:48:44 GMT
//

// system include files
#include <vector>
#include <string>
#include <map>
#include <mutex>
#include <memory>

// user include files

// forward declarations
class testSharedResourcesRegistry;

namespace edm {
  class SharedResourcesAcquirer;
  
  class SharedResourcesRegistry
  {
    
  public:
    //needed for testing
    friend class ::testSharedResourcesRegistry;
    
    // ---------- const member functions ---------------------
    SharedResourcesAcquirer createAcquirer(std::vector<std::string> const&) const;
    
    SharedResourcesAcquirer createAcquirerForSourceDelayedReader();
    // ---------- static member functions --------------------
    static SharedResourcesRegistry* instance();
    
    ///All legacy modules share this resource
    static const std::string kLegacyModuleResourceName;
    
    // ---------- member functions ---------------------------
    ///A resource name must be registered before it can be used in the createAcquirer call
    void registerSharedResource(const std::string&);
    
  private:
    SharedResourcesRegistry()=default;
    ~SharedResourcesRegistry()=default;

    SharedResourcesRegistry(const SharedResourcesRegistry&) = delete; // stop default
    
    const SharedResourcesRegistry& operator=(const SharedResourcesRegistry&) = delete; // stop default
    
    // ---------- member data --------------------------------
    std::map<std::string, std::pair<std::shared_ptr<std::recursive_mutex>,unsigned int>> resourceMap_;
    
    std::shared_ptr<std::recursive_mutex> resourceForDelayedReader_;
    
  };
}

#endif
