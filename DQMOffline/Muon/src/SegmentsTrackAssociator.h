#ifndef SegmentsTrackAssociator_H
#define SegmentsTrackAssociator_H

//standard include
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/InputTag.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h" 
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

namespace edm {class ParameterSet; class Event; class EventSetup;}

class SegmentsTrackAssociator{
  
public:
  
  //constructor
  SegmentsTrackAssociator (const edm::ParameterSet& );
  //distructor 
  virtual ~SegmentsTrackAssociator() {} 
  //function associate
  MuonTransientTrackingRecHit::MuonRecHitContainer associate(const edm::Event&, const edm::EventSetup&, const reco::Track& );
							     //reco::TrackCollection::const_iterator);
  
  

private:

  int NrecHit;
  int NrecHitDT;
  int NrecHitCSC;
  int Nseg;
  
  edm::InputTag theDTSegmentLabel;
  edm::InputTag theCSCSegmentLabel;
  edm::InputTag theSegmentContainerName;

  std::string metname;
 

  
};

#endif
