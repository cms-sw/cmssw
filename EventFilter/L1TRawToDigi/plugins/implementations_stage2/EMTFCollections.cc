#include "FWCore/Framework/interface/Event.h"
#include "EMTFCollections.h"

namespace l1t {
  namespace stage2 {
    EMTFCollections::~EMTFCollections()
    {
      // std::cout << "Inside EMTFCollections.cc: ~EMTFCollections" << std::endl;
      event_.put(std::move(regionalMuonCands_));
      event_.put(std::move(EMTFDaqOuts_));
      event_.put(std::move(EMTFHits_));
      event_.put(std::move(EMTFTracks_));
      event_.put(std::move(EMTFHit2016s_));
      event_.put(std::move(EMTFTrack2016s_));
      event_.put(std::move(EMTFLCTs_));
    }
  }
}
