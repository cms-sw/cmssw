#ifndef FWCore_Framework_ConsumesCollector_h
#define FWCore_Framework_ConsumesCollector_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::ConsumesCollector
// 
/**\class edm::ConsumesCollector ConsumesCollector.h "FWCore/Framework/interface/ConsumesCollector.h"

 Description: Helper class to gather consumes information for EDConsumerBase class.

 Usage:
    The constructor of a module can get an instance of edm::ConsumesCollector by calling its
consumesCollector() method. This instance can then be passed to helper classes in order to register
the data the helper will request from an Event, LuminosityBlock or Run on behalf of the module.

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 07 Jun 2013 12:44:47 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/EDConsumerBase.h"

// forward declarations
namespace edm {
  class EDConsumerBase;
  
  class ConsumesCollector
  {
    
  public:
    //virtual ~ConsumesCollector();
    ConsumesCollector(ConsumesCollector&& iOther): m_consumer(iOther.m_consumer){}
    
    // ---------- member functions ---------------------------
    template <typename ProductType, BranchType B=InEvent>
    EDGetTokenT<ProductType> consumes(edm::InputTag const& tag) {
      return m_consumer->consumes<ProductType,B>(tag);
    }
    
    EDGetToken consumes(const TypeToGet& id, edm::InputTag const& tag) {
      return m_consumer->consumes(id,tag);
    }
    
    template <BranchType B>
    EDGetToken consumes(TypeToGet const& id, edm::InputTag const& tag) {
      return m_consumer->consumes<B>(id,tag);
    }
    
    template <typename ProductType, BranchType B=InEvent>
    EDGetTokenT<ProductType> mayConsume(edm::InputTag const& tag) {
      return m_consumer->mayConsume<ProductType,B>(tag);
    }
    
    
    EDGetToken mayConsume(const TypeToGet& id, edm::InputTag const& tag) {
      return m_consumer->mayConsume(id,tag);
    }
    
    template <BranchType B>
    EDGetToken mayConsume(const TypeToGet& id, edm::InputTag const& tag) {
      return m_consumer->mayConsume<B>(id,tag);
    }
    
    template <typename ProductType, BranchType B=InEvent>
    void consumesMany() {
      m_consumer->consumesMany<ProductType,B>();
    }
    
    
    void consumesMany(const TypeToGet& id) {
      m_consumer->consumesMany(id);
    }
    
    template <BranchType B>
    void consumesMany(const TypeToGet& id) {
      m_consumer->consumesMany<B>(id);
    }
    

  private:
    //only EDConsumerBase is allowed to make an instance of this class
    friend class EDConsumerBase;
    
    ConsumesCollector(EDConsumerBase* iConsumer):
    m_consumer(iConsumer) {}

    ConsumesCollector() = delete;
    ConsumesCollector(const ConsumesCollector&) = delete; // stop default
    
    const ConsumesCollector& operator=(const ConsumesCollector&) = delete; // stop default
    
    // ---------- member data --------------------------------
    EDConsumerBase* m_consumer;
    
  };
}


#endif
