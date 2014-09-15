#include "FWCore/Framework/interface/one/EDProducerBase.h"
#include "FWCore/Framework/interface/Event.h"

#include "L1TCollections.h"

namespace l1t {
   L1TCollections::~L1TCollections()
   {
      event_.put(towers_);
      event_.put(egammas_);
      event_.put(etsums_);
      event_.put(jets_);
      event_.put(taus_);

      event_.put(mp_etsums_, "MP");
      event_.put(mp_jets_, "MP");
   }

   L1TCollectionsProduces::L1TCollectionsProduces(edm::one::EDProducerBase& prod) : UnpackerCollectionsProduces(prod)
   {
      prod.produces<CaloTowerBxCollection>();
      prod.produces<EGammaBxCollection>();
      prod.produces<EtSumBxCollection>();
      prod.produces<JetBxCollection>();
      prod.produces<TauBxCollection>();

      prod.produces<EtSumBxCollection>("MP");
      prod.produces<JetBxCollection>("MP");
   }
}

DEFINE_L1TUNPACKER_COLLECTION(l1t::L1TCollections);
DEFINE_L1TUNPACKER_COLLECTION_PRODUCES(l1t::L1TCollectionsProduces);
