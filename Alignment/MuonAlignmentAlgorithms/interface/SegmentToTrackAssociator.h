#ifndef MuonAlignmentAlgorithms_SegmentToTrackAssociator_H
#define MuonAlignmentAlgorithms_SegmentToTrackAssociator_H

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

typedef std::vector< std::vector<int> > intDVector;


class SegmentToTrackAssociator
{
  
public:
  
  //constructor
  SegmentToTrackAssociator ( const edm::ParameterSet& );

  //destructor 
  virtual ~SegmentToTrackAssociator() {} 

  //Associate
  MuonTransientTrackingRecHit::MuonRecHitContainer associate( const edm::Event&, const edm::EventSetup&, const reco::Track&, std::string  );
  
  //Clear the vector
  void clear();
  

private:

  intDVector indexCollectionDT;
  intDVector indexCollectionCSC;
 
  edm::InputTag theDTSegmentLabel;
  edm::InputTag theCSCSegmentLabel;


};

#endif
