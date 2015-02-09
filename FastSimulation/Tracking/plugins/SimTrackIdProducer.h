#ifndef FastSimulation_Tracking_SimTrackIdProducer_h
#define FastSimulation_Tracking_SimTrackIdProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"


#include <vector>
#include <string>

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}

class SimTrackIdProducer : public edm::stream::EDProducer <>
 {
   public:

      explicit SimTrackIdProducer(const edm::ParameterSet& conf);

      virtual ~SimTrackIdProducer() {}

      virtual void produce(edm::Event& e, const edm::EventSetup& es) override;

      std::vector<int> SimTrackIds;

private:

      // consumes 
      edm::EDGetTokenT<reco::TrackCollection>  trackToken;

};

#endif
