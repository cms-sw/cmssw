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
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/propagate_const.h"
#include "FWCore/Utilities/interface/Transition.h"

namespace edm {
  class ESConsumesCollector {
  public:

    ESConsumesCollector() = delete;
    ESConsumesCollector(ESConsumesCollector const&) = default;
    ESConsumesCollector(ESConsumesCollector&&) = default;
    ESConsumesCollector& operator=(ESConsumesCollector const&) = default;
    ESConsumesCollector& operator=(ESConsumesCollector&&) = default;

    // ---------- member functions ---------------------------
    template <typename Product>
    auto consumes(ESInputTag const& tag) {
      return ESGetTokenT<Product>{tag};
    }

    template <typename Product, typename Record>
    auto consumes(ESInputTag const& tag) {
      return ESGetTokenT<Product>{tag};
    }

    template <typename Product>
    auto consumes(eventsetup::EventSetupRecordKey const&, ESInputTag const& tag) {
      return ESGetTokenT<Product>{tag};
    }

  private:
    //only ESProducer is allowed to make an instance of this class
    friend class ESProducer;

    explicit ESConsumesCollector(ESProducer* const iConsumer) :
      m_consumer{iConsumer}
    {}

    // ---------- member data --------------------------------
    edm::propagate_const<ESProducer*> m_consumer{nullptr};
  };
}


#endif
