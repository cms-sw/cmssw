#ifndef MuonAlignmentAlgorithms_SegmentToTrackAssociator_H
#define MuonAlignmentAlgorithms_SegmentToTrackAssociator_H

#include <vector>

//standard include
#include "FWCore/Framework/interface/ConsumesCollector.h"
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
#include "DataFormats/DTRecHit/interface/DTRecSegment4DCollection.h"
#include "DataFormats/CSCRecHit/interface/CSCSegmentCollection.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "Geometry/Records/interface/GlobalTrackingGeometryRecord.h"
#include "Geometry/CommonDetUnit/interface/GlobalTrackingGeometry.h"

namespace edm {
  class EventSetup;
}  // namespace edm

class SegmentToTrackAssociator {
public:
  typedef std::vector<std::vector<int> > intDVector;

  //constructor
  SegmentToTrackAssociator(const edm::ParameterSet&,
                           const GlobalTrackingGeometry* GlobalTrackingGeometry,
                           edm::ConsumesCollector&);

  //destructor
  virtual ~SegmentToTrackAssociator() = default;

  //Associate
  MuonTransientTrackingRecHit::MuonRecHitContainer associate(const edm::Event&,
                                                             const edm::EventSetup&,
                                                             const reco::Track&,
                                                             std::string);

  //Clear the vector
  void clear();

private:
  intDVector indexCollectionDT;
  intDVector indexCollectionCSC;

  const GlobalTrackingGeometry* globalTrackingGeometry_;

  const edm::InputTag theDTSegmentLabel;
  const edm::InputTag theCSCSegmentLabel;
  const edm::EDGetTokenT<DTRecSegment4DCollection> tokenDTSegment_;
  const edm::EDGetTokenT<CSCSegmentCollection> tokenCSCSegment_;
};

#endif
