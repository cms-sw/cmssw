#ifndef FWCore_Framework_ESConsumesCollector_h
#define FWCore_Framework_ESConsumesCollector_h
// -*- C++ -*-
//
// Package:     FWCore/Framework
// Class  :     edm::ConsumesCollector
//
/**\class edm::ESConsumesCollector ESConsumesCollector.h "FWCore/Framework/interface/ESConsumesCollector.h"

 Description: Helper class to gather consumes information for the EventSetup.

 Usage: The constructor of an ESProducer module can get an instance of
        edm::ESConsumesCollector by calling consumesCollector()
        method. This instance can then be passed to helper classes in
        order to register the event-setup data the helper will request
        from an Event, LuminosityBlock or Run on behalf of the module.

 Caveat: The ESConsumesCollector should be used during the time that
         modules are being constructed. It should not be saved and
         used later. It will not work if it is used to call the
         consumes function during beginJob, beginRun, beginLuminosity
         block, event processing or at any later time. As of now, an
         ESConsumesCollector is provided for only ESProducer
         subclasses--i.e. those that call setWhatProduced(this, ...).
*/
//
// Original Author:  Kyle Knoepfel
//         Created:  Fri, 02 Oct 2018 12:44:47 GMT
//

#include "FWCore/Framework/interface/EventSetupRecordKey.h"
#include "FWCore/Framework/interface/DataKey.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/Transition.h"

#include <vector>
#include <memory>
namespace edm {
  using ESConsumesInfo =
      std::vector<std::tuple<edm::eventsetup::EventSetupRecordKey, edm::eventsetup::DataKey, std::string> >;

  class ESConsumesCollector {
  public:
    ESConsumesCollector() = delete;
    ESConsumesCollector(ESConsumesCollector const&) = default;
    ESConsumesCollector(ESConsumesCollector&&) = default;
    ESConsumesCollector& operator=(ESConsumesCollector const&) = default;
    ESConsumesCollector& operator=(ESConsumesCollector&&) = default;

    // ---------- member functions ---------------------------
    template <typename Product, typename Record>
    auto consumesFrom(ESInputTag const& tag) {
      using namespace edm::eventsetup;
      ESTokenIndex index{static_cast<ESTokenIndex::Value_t>(m_consumer->size())};
      m_consumer->emplace_back(EventSetupRecordKey::makeKey<Record>(),
                               DataKey(DataKey::makeTypeTag<Product>(), tag.data().c_str()),
                               tag.module());
      //even though m_consumer may expand, the address for
      // name().value() remains the same since it is 'moved'.
      return ESGetToken<Product, Record>{m_transitionID, index, std::get<1>(m_consumer->back()).name().value()};
    }

  protected:
    explicit ESConsumesCollector(ESConsumesInfo* const iConsumer, unsigned int iTransitionID)
        : m_consumer{iConsumer}, m_transitionID{iTransitionID} {}

  private:
    // ---------- member data --------------------------------
    edm::propagate_const<ESConsumesInfo*> m_consumer{nullptr};
    unsigned int m_transitionID{0};
  };

  template <typename RECORD>
  class ESConsumesCollectorT : public ESConsumesCollector {
  public:
    ESConsumesCollectorT() = delete;
    ESConsumesCollectorT(ESConsumesCollectorT<RECORD> const&) = default;
    ESConsumesCollectorT(ESConsumesCollectorT<RECORD>&&) = default;
    ESConsumesCollectorT<RECORD>& operator=(ESConsumesCollectorT<RECORD> const&) = default;
    ESConsumesCollectorT<RECORD>& operator=(ESConsumesCollectorT<RECORD>&&) = default;

    // ---------- member functions ---------------------------

    template <typename Product>
    auto consumes(ESInputTag const& tag) {
      return consumesFrom<Product, RECORD>(tag);
    }

  private:
    //only ESProducer is allowed to make an instance of this class
    friend class ESProducer;

    explicit ESConsumesCollectorT(ESConsumesInfo* const iConsumer, unsigned int iTransitionID)
        : ESConsumesCollector(iConsumer, iTransitionID) {}
  };

}  // namespace edm

#endif
