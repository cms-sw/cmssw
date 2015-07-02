#ifndef FastSimulation_Tracking_SimTrackIdProducer_h
#define FastSimulation_Tracking_SimTrackIdProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/TrackReco/interface/Track.h"


#include <vector>
#include <string>

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}

class SimTrackIdProducer final : public edm::stream::EDProducer <>
 {
   public:

   using QualityMaskCollection = std::vector<unsigned char>;

      explicit SimTrackIdProducer(const edm::ParameterSet& conf);

      virtual ~SimTrackIdProducer() {}

      virtual void produce(edm::Event& e, const edm::EventSetup& es) override;


private:

      // consumes 
      edm::EDGetTokenT<reco::TrackCollection>  trackToken;
      edm::EDGetTokenT<QualityMaskCollection> srcQuals;
      double maxChi2_;
      bool filterTracks_ = false;
      reco::TrackBase::TrackQuality trackQuality_;
     
};

#endif
