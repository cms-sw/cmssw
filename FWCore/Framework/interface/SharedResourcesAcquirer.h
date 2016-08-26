#ifndef FWCore_Framework_SharedResourcesAcquirer_h
#define FWCore_Framework_SharedResourcesAcquirer_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
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
#include "FWCore/Concurrency/interface/SerialTaskQueueChain.h"

// user include files

// forward declarations
class testSharedResourcesRegistry;

namespace edm {
  class SerialTaskQueueChain;
  class SerialTaskQueue;

  class SharedResourcesAcquirer
  {
  public:
    friend class ::testSharedResourcesRegistry;
    
    SharedResourcesAcquirer() = default;
    explicit SharedResourcesAcquirer(std::vector<std::shared_ptr<SerialTaskQueue>>  iQueues):
    m_queues(std::move(iQueues)){}
    
    SharedResourcesAcquirer(SharedResourcesAcquirer&&) = default;
    SharedResourcesAcquirer(const SharedResourcesAcquirer&) = delete;
    SharedResourcesAcquirer& operator=(const SharedResourcesAcquirer&) = delete;
    
    SharedResourcesAcquirer& operator=(SharedResourcesAcquirer&&) = default;
    ~SharedResourcesAcquirer() = default;
    
    // ---------- member functions ---------------------------
    
    ///The number returned may be less than the number of resources requested if a resource is only used by one module and therefore is not being shared.
    size_t numberOfResources() const { return m_queues.numberOfQueues();}
    
    SerialTaskQueueChain& serialQueueChain() const { return m_queues; }
  private:
    
    // ---------- member data --------------------------------
    mutable SerialTaskQueueChain m_queues;
  };
}


#endif
