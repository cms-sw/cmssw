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

     WARNING: The ConsumesCollector should be used during the time that modules are being
constructed. It should not be saved and used later. It will not work if it is used to call
the consumes function during beginJob, beginRun, beginLuminosity block, event processing or
at any later time. It can be used while the module constructor is running or be contained in
a functor passed to the Framework with a call to callWhenNewProductsRegistered.

*/
//
// Original Author:  Chris Jones
//         Created:  Fri, 07 Jun 2013 12:44:47 GMT
//

// system include files

// user include files
#include "FWCore/Framework/interface/EDConsumerBase.h"
#include "FWCore/Utilities/interface/propagate_const.h"

// forward declarations
namespace edm {
  class EDConsumerBase;
  template <Transition TR>
  class ConsumesCollectorESAdaptor;
  template <Transition TR>
  class ConsumesCollectorWithTagESAdaptor;
  template <BranchType B>
  class ConsumesCollectorAdaptor;

  class ConsumesCollector {
  public:
    ConsumesCollector() = delete;
    ConsumesCollector(ConsumesCollector const&);
    ConsumesCollector(ConsumesCollector&&) = default;
    ConsumesCollector& operator=(ConsumesCollector const&);
    ConsumesCollector& operator=(ConsumesCollector&&) = default;

    // ---------- member functions ---------------------------
    template <typename ProductType, BranchType B = InEvent>
    EDGetTokenT<ProductType> consumes(edm::InputTag const& tag) {
      return m_consumer->consumes<ProductType, B>(tag);
    }

    template <BranchType B = InEvent>
    [[nodiscard]] ConsumesCollectorAdaptor<B> consumes(edm::InputTag tag) {
      return ConsumesCollectorAdaptor<B>(*this, std::move(tag));
    }

    EDGetToken consumes(const TypeToGet& id, edm::InputTag const& tag) { return m_consumer->consumes(id, tag); }

    template <BranchType B>
    EDGetToken consumes(TypeToGet const& id, edm::InputTag const& tag) {
      return m_consumer->consumes<B>(id, tag);
    }

    template <typename ProductType, BranchType B = InEvent>
    EDGetTokenT<ProductType> mayConsume(edm::InputTag const& tag) {
      return m_consumer->mayConsume<ProductType, B>(tag);
    }

    EDGetToken mayConsume(const TypeToGet& id, edm::InputTag const& tag) { return m_consumer->mayConsume(id, tag); }

    template <BranchType B>
    EDGetToken mayConsume(const TypeToGet& id, edm::InputTag const& tag) {
      return m_consumer->mayConsume<B>(id, tag);
    }

    // For consuming event-setup products
    template <typename ESProduct, typename ESRecord, Transition Tr = Transition::Event>
    auto esConsumes() {
      return esConsumes<ESProduct, ESRecord, Tr>(ESInputTag{});
    }

    template <typename ESProduct, typename ESRecord, Transition Tr = Transition::Event>
    auto esConsumes(ESInputTag const& tag) {
      return m_consumer->esConsumes<ESProduct, ESRecord, Tr>(tag);
    }

    template <typename ESProduct, Transition Tr = Transition::Event>
    auto esConsumes(eventsetup::EventSetupRecordKey const& key, ESInputTag const& tag) {
      return m_consumer->esConsumes<ESProduct, Tr>(key, tag);
    }

    template <Transition Tr = Transition::Event>
    [[nodiscard]] constexpr auto esConsumes() noexcept {
      return ConsumesCollectorESAdaptor<Tr>(*this);
    }

    template <Transition Tr = Transition::Event>
    [[nodiscard]] auto esConsumes(ESInputTag tag) noexcept {
      return ConsumesCollectorWithTagESAdaptor<Tr>(*this, std::move(tag));
    }

  private:
    //only EDConsumerBase is allowed to make an instance of this class
    friend class EDConsumerBase;

    ConsumesCollector(EDConsumerBase* iConsumer) : m_consumer(iConsumer) {}

    // ---------- member data --------------------------------
    edm::propagate_const<EDConsumerBase*> m_consumer;
  };

  template <Transition TR>
  class ConsumesCollectorESAdaptor {
  public:
    template <typename TYPE, typename REC>
    ESGetToken<TYPE, REC> consumes() {
      return m_consumer.template esConsumes<TYPE, REC, TR>();
    }

  private:
    //only ConsumesCollector is allowed to make an instance of this class
    friend class ConsumesCollector;

    explicit ConsumesCollectorESAdaptor(ConsumesCollector iBase) : m_consumer(std::move(iBase)) {}

    ConsumesCollector m_consumer;
  };

  template <Transition TR>
  class ConsumesCollectorWithTagESAdaptor {
  public:
    template <typename TYPE, typename REC>
    ESGetToken<TYPE, REC> consumes() {
      return m_consumer.template esConsumes<TYPE, REC, TR>(m_tag);
    }

  private:
    //only ConsumesCollector is allowed to make an instance of this class
    friend class ConsumesCollector;

    ConsumesCollectorWithTagESAdaptor(ConsumesCollector iBase, ESInputTag iTag)
        : m_consumer(std::move(iBase)), m_tag(std::move(iTag)) {}

    ConsumesCollector m_consumer;
    ESInputTag const m_tag;
  };

  template <BranchType B>
  class ConsumesCollectorAdaptor {
  public:
    template <typename TYPE>
    EDGetTokenT<TYPE> consumes() {
      return m_consumer.template consumes<TYPE, B>(m_tag);
    }

  private:
    //only ConsumesCollector is allowed to make an instance of this class
    friend class ConsumesCollector;

    ConsumesCollectorAdaptor(ConsumesCollector iBase, edm::InputTag iTag) : m_consumer(iBase), m_tag(std::move(iTag)) {}

    ConsumesCollector m_consumer;
    edm::InputTag const m_tag;
  };

}  // namespace edm

#endif
