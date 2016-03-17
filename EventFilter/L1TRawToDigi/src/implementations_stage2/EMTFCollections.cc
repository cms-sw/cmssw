#include "FWCore/Framework/interface/Event.h"
#include "EMTFCollections.h"

namespace l1t {
  namespace stage2 {
    EMTFCollections::~EMTFCollections()
    {
      event_.put(regionalMuonCands_);
      event_.put(EMTFOutputs_);
    }
  }
}
