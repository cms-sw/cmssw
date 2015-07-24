#ifndef FastSimulation_Tracking_FastTrackingMaskProducer_h
#define FastSimulation_Tracking_FastTrackingMaskProducer_h

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

class FastTrackingMaskProducer : public edm::stream::EDProducer <>
{
 public:

  explicit FastTrackingMaskProducer(const edm::ParameterSet& conf);

  virtual ~FastTrackingMaskProducer() {}

  virtual void produce(edm::Event& e, const edm::EventSetup& es) override;


 private:



  // consumes                                                                                                                 
  edm::EDGetTokenT<reco::TrackCollection>  trackToken_;
  edm::EDGetTokenT<std::vector<bool> > hitMasksToken_;
  edm::EDGetTokenT<std::vector<bool> > hitCombinationMasksToken_;
  edm::EDGetTokenT<edm::ValueMap<int> > trkQualsToken_;
  // set value in constructor
  bool oldHitMasks_exists_;
  bool oldHitCombinationMasks_exists_;
  bool overRideTrkQuals_;
  bool filterTracks_;
  reco::TrackBase::TrackQuality trackQuality_;

};

#endif






