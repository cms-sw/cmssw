#include "FWCore/Framework/interface/Event.h"
#include "EMTFCollections.h"

namespace l1t {
  namespace stage2 {
    EMTFCollections::~EMTFCollections()
    {
      // std::cout << "Inside EMTFCollections.cc: ~EMTFCollections" << std::endl;

      // Sort by processor to match uGMT unpacked order
      L1TMuonEndCap::sort_uGMT_muons(*regionalMuonCands_);

      event_.put(std::move(regionalMuonCands_));
      event_.put(std::move(EMTFDaqOuts_));
      event_.put(std::move(EMTFHits_));
      event_.put(std::move(EMTFTracks_));
      event_.put(std::move(EMTFLCTs_));
      event_.put(std::move(EMTFCPPFs_));
    }
  }
}
