#ifndef TRACKTORPC_H
#define TRACKTORPC_H

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"
#include "DataFormats/MuonReco/interface/MuonTrackLinks.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"

#include "TrackingTools/PatternTools/interface/Trajectory.h"
#include "TrackingTools/PatternTools/interface/TrajTrackAssociation.h"
#include "TrackingTools/GeomPropagators/interface/Propagator.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "TrackingTools/Records/interface/TrackingComponentsRecord.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TrajectoryState/interface/FreeTrajectoryState.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

#include "RecoMuon/DetLayers/interface/MuonDetLayerGeometry.h"
#include "RecoMuon/Records/interface/MuonRecoGeometryRecord.h"
#include "RecoMuon/MeasurementDet/interface/MuonDetLayerMeasurements.h"
#include "RecoMuon/TransientTrackingRecHit/interface/MuonTransientTrackingRecHit.h"

#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHitCollection.h"
#include "DataFormats/Common/interface/Ref.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/RPCGeometry/interface/RPCGeometry.h"
#include "Geometry/CommonDetUnit/interface/GeomDetType.h"
#include "Geometry/RPCGeometry/interface/RPCRoll.h"
#include "Geometry/RPCGeometry/interface/RPCGeomServ.h"
#include "Geometry/DTGeometry/interface/DTLayer.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"
#include "Geometry/CommonTopologies/interface/RectangularStripTopology.h"
#include "Geometry/CommonTopologies/interface/TrapezoidalStripTopology.h"

#include "TrackingTools/TrackRefitter/interface/TrackTransformerForCosmicMuons.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformerBase.h"
#include "TrackingTools/TrackRefitter/interface/TrackTransformer.h"

#include <memory>

class RPCGeometry;
class DTGeometry;
class DTObjectMap;
class CSCGeometry;
class CSCObjectMap;
class MuonGeometryRecord;
class Propagator;
class TrackingComponentsRecord;

using reco::MuonCollection;
using reco::TrackCollection;
typedef std::vector<Trajectory> Trajectories;

class TracktoRPC {
public:
  TracktoRPC(const edm::ParameterSet& iConfig, const edm::InputTag& tracklabel, edm::ConsumesCollector iC);
  std::unique_ptr<RPCRecHitCollection> thePoints(reco::TrackCollection const* alltracks,
                                                 edm::EventSetup const& iSetup,
                                                 bool debug);

private:
  bool ValidRPCSurface(RPCDetId rpcid, LocalPoint LocalP, const RPCGeometry* rpcGeo);

  edm::ESGetToken<RPCGeometry, MuonGeometryRecord> rpcGeoToken_;
  edm::ESGetToken<DTGeometry, MuonGeometryRecord> dtGeoToken_;
  edm::ESGetToken<DTObjectMap, MuonGeometryRecord> dtMapToken_;
  edm::ESGetToken<CSCGeometry, MuonGeometryRecord> cscGeoToken_;
  edm::ESGetToken<CSCObjectMap, MuonGeometryRecord> cscMapToken_;
  edm::ESGetToken<Propagator, TrackingComponentsRecord> propagatorToken_;
  std::unique_ptr<TrackTransformerBase> theTrackTransformer;
};

#endif
