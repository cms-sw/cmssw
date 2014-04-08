#ifndef FastSimulation_Tracking_FastTrackMerger_h
#define FastSimulation_Tracking_FastTrackMerger_h

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"

#include <vector>
#include <string>

namespace edm { 
  class ParameterSet;
  class Event;
  class EventSetup;
}

//namespace reco { 
//  class Track;
//}


class FastTrackMerger : public edm::EDProducer
{
 public:
  
  explicit FastTrackMerger(const edm::ParameterSet& conf);
  
  virtual ~FastTrackMerger() {}
  
  virtual void produce(edm::Event& e, const edm::EventSetup& es);
  
 private:

  int findId(const reco::Track& aTrack) const;

 private:

  std::vector<edm::InputTag> trackProducers;
  std::vector<edm::InputTag> removeTrackProducers;
  bool tracksOnly;
  bool promoteQuality;
  double pTMin2;
  unsigned minHits;
  unsigned trackAlgo;
  std::string qualityStr;
  unsigned theMinimumNumberOfHits;
  unsigned theMaxLostHits;
  unsigned theMaxConsecutiveLostHits;

  // consumes
  std::vector<edm::EDGetTokenT<reco::TrackCollection> > removeTrackTokens;
  std::vector<edm::EDGetTokenT<reco::TrackCollection> > trackTokens;
  std::vector<edm::EDGetTokenT<std::vector<Trajectory> > > trajectoryTokens;
  std::vector<edm::EDGetTokenT<TrajTrackAssociationCollection> > assoMapTokens;

};

#endif
