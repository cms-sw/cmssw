#ifndef SegmentsTrackAssociator_H
#define SegmentsTrackAssociator_H


/** \class SegmentsTrackAssociator
 *
 *  tool which take as input a muon track and return a vector 
 *  with the segments used to fit it
 *
 *  \author C. Botta, G. Mila - INFN Torino
 */


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"
#include "TrackingTools/PatternTools/interface/TrajectoryMeasurement.h" 
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/PatternTools/interface/TrajMeasLessEstim.h"
#include "RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"



namespace edm {class ParameterSet; class Event; class EventSetup;}


class SegmentsTrackAssociator{
  
public:
  
  /// Constructor
  SegmentsTrackAssociator (const edm::ParameterSet& ,edm::ConsumesCollector& iC);
  
  /// Destructor 
  virtual ~SegmentsTrackAssociator();

  /// Get the analysis
  MuonTransientTrackingRecHit::MuonRecHitContainer associate(const edm::Event&, const edm::EventSetup&, const reco::Track& );
  
  

private:

  // the counters
  int numRecHit;
  int numRecHitDT;
  int numRecHitCSC;
  
  // collection label
  edm::InputTag theDTSegmentLabel;
  edm::InputTag theCSCSegmentLabel;
  edm::InputTag theSegmentContainerName;

  edm::EDGetTokenT<DTRecSegment4DCollection> dtSegmentsToken;
  edm::EDGetTokenT<CSCSegmentCollection> cscSegmentsToken;


  std::string metname;
 

  
};

#endif
