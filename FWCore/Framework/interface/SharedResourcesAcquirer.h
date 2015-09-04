#ifndef Subsystem_Package_SharedResourcesAcquirer_h
#define Subsystem_Package_SharedResourcesAcquirer_h
// -*- C++ -*-
//
// Package:     Subsystem/Package
// Class  :     SharedResourcesAcquirer
// 
/**\class SharedResourcesAcquirer SharedResourcesAcquirer.h "SharedResourcesAcquirer.h"

 Description: Handles acquiring and releasing a group of resources shared between modules

 Usage:
    <usage>

*/
//
// Original Author:  Chris Jones
//         Created:  Sun, 06 Oct 2013 19:43:26 GMT
//

// system include files

// user include files
#include <vector>
#include <mutex>
#include <memory>

// forward declarations
class testSharedResourcesRegistry;
namespace edm {
  class SharedResourcesAcquirer
  {
  public:
    friend class ::testSharedResourcesRegistry;
    
    SharedResourcesAcquirer() = default;
    explicit SharedResourcesAcquirer(std::vector<std::recursive_mutex*>&& iResources):
    m_resources(iResources){}
    
    SharedResourcesAcquirer(SharedResourcesAcquirer&&) = default;
    SharedResourcesAcquirer(const SharedResourcesAcquirer&) = default;
    SharedResourcesAcquirer& operator=(const SharedResourcesAcquirer&) = default;
    
    ~SharedResourcesAcquirer() = default;
    
    // ---------- member functions ---------------------------
    void lock();
    void unlock();
    
    ///Used by the framework to temporarily unlock a resource in the case where a module is temporarily suspended,
    /// e.g. when a Event::getByLabel call launches a Producer via unscheduled processing
    template<typename FUNC>
    void temporaryUnlock(FUNC iFunc) {
      auto iThis = this;
      std::shared_ptr<void> guard(nullptr,[iThis](void* ) {iThis->lock();});
      this->unlock();
      iFunc();
    }
    
    ///The number returned may be less than the number of resources requested if a resource is only used by one module and therefore is not being shared.
    size_t numberOfResources() const { return m_resources.size();}
  private:
    
    // ---------- member data --------------------------------
    std::vector<std::recursive_mutex*> m_resources;
  };
}


#endif
