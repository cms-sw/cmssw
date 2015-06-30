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
  edm::EDGetTokenT<reco::TrackCollection>  trackToken;
  edm::EDGetTokenT<std::vector<bool> > hitMasksToken;
  edm::EDGetTokenT<std::vector<bool> > hitCombinationMasksToken;  
  std::vector< edm::EDGetTokenT<edm::ValueMap<int> > > overrideTrkQuals_;
  // set value in constructor
  bool filterTracks_ = false;
  bool oldHitMasks_exist = false;
  bool oldHitCombinationMasks_exist = false;
  reco::TrackBase::TrackQuality trackQuality_;

};

#endif






