#include "EventFilter/L1TRawToDigi/interface/L1TRawToDigi.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"

namespace l1t {
   UnpackerCollections::UnpackerCollections(edm::Event& event) :
      event_(event),
      jets_(new JetBxCollection()),
      taus_(new TauBxCollection()),
      egammas_(new EGammaBxCollection()),
      etsums_(new EtSumBxCollection()),
      calotowers_(new CaloTowerBxCollection())
   {
   }

   UnpackerCollections::~UnpackerCollections()
   {
      // For every member:
      event_.put(jets_);
      event_.put(taus_);
      event_.put(egammas_);
      event_.put(etsums_);
      event_.put(calotowers_);
   }

   void
   UnpackerCollections::registerCollections(L1TRawToDigi *prod)
   {
      // For every member:
      prod->produces<JetBxCollection>();
      prod->produces<TauBxCollection>();
      prod->produces<EGammaBxCollection>();
      prod->produces<EtSumBxCollection>();
      prod->produces<CaloTowerBxCollection>();
   }
}
