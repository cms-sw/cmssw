#include "EventFilter/L1TRawToDigi/interface/L1TRawToDigi.h"
#include "EventFilter/L1TRawToDigi/interface/UnpackerCollections.h"

namespace l1t {
   UnpackerCollections::UnpackerCollections(edm::Event& event) :
      event_(event),
      jets_(new JetBxCollection()),
      taus_(new TauBxCollection())
   {
   }

   UnpackerCollections::~UnpackerCollections()
   {
      // For every member:
      event_.put(jets_);
      event_.put(taus_);
   }

   void
   UnpackerCollections::registerCollections(L1TRawToDigi *prod)
   {
      // For every member:
      prod->produces<JetBxCollection>();
      prod->produces<TauBxCollection>();
   }
}
