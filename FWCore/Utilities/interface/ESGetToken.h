#ifndef FWCore_Utilities_ESGetToken_h
#define FWCore_Utilities_ESGetToken_h
// -*- C++ -*-
//
// Package:     FWCore/Utilities
// Class  :     ESGetToken
//
/**\class ESGetToken ESGetToken.h "FWCore/Utilities/interface/ESGetToken.h"

 Description: A token used to get data from the event setup system

 Usage:
    An ESGetToken is created by calls to 'esConsumes' from an ED module
    or via a ConsumesCollector::consumes.
*/

#include "FWCore/Utilities/interface/ESInputTag.h"

namespace edm {
  class EDConsumerBase;
  class ESProducer;
  class ESConsumesCollector;
  class EventSetup;
  class EventSetupImpl;
  namespace eventsetup {
    class EventSetupRecord;
  }

  // A ESGetToken is created by calls to 'esConsumes' from an EDM
  // module.
  template<typename ESProduct, typename ESRecord>
  class ESGetToken {
    friend class EDConsumerBase;
    friend class ESProducer;
    friend class ESConsumesCollector;
    friend class EventSetup;
    friend class EventSetupImpl;
    friend class eventsetup::EventSetupRecord;

  public:
    ESGetToken() = default;

  private:
    explicit ESGetToken(ESInputTag const& tag) : m_tag{tag} {}

    ESInputTag m_tag{};
  };

}

#endif
