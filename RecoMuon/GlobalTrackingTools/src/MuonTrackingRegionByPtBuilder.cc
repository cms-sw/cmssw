#include "RecoMuon/GlobalTrackingTools/interface/MuonTrackingRegionByPtBuilder.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"
#include "RecoTracker/Record/interface/TrackerMultipleScatteringRecord.h"
#include "RecoTracker/TkMSParametrization/interface/MultipleScatteringParametrisationMaker.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"

//
// constructor
//
void MuonTrackingRegionByPtBuilder::build(const edm::ParameterSet& par, edm::ConsumesCollector& iC) {
  // Adjust errors on dz
  theNsigmaDz = par.getParameter<double>("Rescale_Dz");

  // Flag to switch to use Vertices instead of BeamSpot
  useVertex = par.getParameter<bool>("UseVertex");

  // Flag to use fixed limits for Eta, Phi, Z, pT
  useFixedZ = par.getParameter<bool>("Z_fixed");
  useFixedPt = par.getParameter<bool>("Pt_fixed");

  // Minimum value for pT
  thePtMin = par.getParameter<double>("Pt_min");

  // The static region size along the Z direction
  theHalfZ = par.getParameter<double>("DeltaZ");

  // The transverse distance of the region from the BS/PV
  theDeltaR = par.getParameter<double>("DeltaR");

  // Parameterized ROIs depending on the pt
  ptRanges_ = par.getParameter<std::vector<double>>("ptRanges");
  if (ptRanges_.size() < 2) {
    edm::LogError("MuonTrackingRegionByPtBuilder") << "Size of ptRanges does not be less than 2." << std::endl;
  }

  deltaEtas_ = par.getParameter<std::vector<double>>("deltaEtas");
  if (deltaEtas_.size() != ptRanges_.size() - 1) {
    edm::LogError("MuonTrackingRegionByPtBuilder")
        << "Size of deltaEtas does not match number of pt bins." << std::endl;
  }

  deltaPhis_ = par.getParameter<std::vector<double>>("deltaPhis");
  if (deltaPhis_.size() != ptRanges_.size() - 1) {
    edm::LogError("MuonTrackingRegionByPtBuilder")
        << "Size of deltaPhis does not match number of pt bins." << std::endl;
  }

  // Maximum number of regions to build when looping over Muons
  theMaxRegions = par.getParameter<int>("maxRegions");

  // Flag to use precise??
  thePrecise = par.getParameter<bool>("precise");

  // perigee reference point
  theOnDemand = RectangularEtaPhiTrackingRegion::intToUseMeasurementTracker(par.getParameter<int>("OnDemand"));
  if (theOnDemand != RectangularEtaPhiTrackingRegion::UseMeasurementTracker::kNever) {
    theMeasurementTrackerToken =
        iC.consumes<MeasurementTrackerEvent>(par.getParameter<edm::InputTag>("MeasurementTrackerName"));
  }

  // Vertex collection and Beam Spot
  beamSpotToken = iC.consumes<reco::BeamSpot>(par.getParameter<edm::InputTag>("beamSpot"));
  vertexCollectionToken = iC.consumes<reco::VertexCollection>(par.getParameter<edm::InputTag>("vertexCollection"));

  // Input muon collection
  inputCollectionToken = iC.consumes<reco::TrackCollection>(par.getParameter<edm::InputTag>("input"));

  bfieldToken = iC.esConsumes();
  if (thePrecise) {
    msmakerToken = iC.esConsumes();
  }
}

//
// Member function to be compatible with TrackingRegionProducerFactory: create many ROI for many tracks
//
std::vector<std::unique_ptr<TrackingRegion>> MuonTrackingRegionByPtBuilder::regions(const edm::Event& ev,
                                                                                    const edm::EventSetup& es) const {
  std::vector<std::unique_ptr<TrackingRegion>> result;

  edm::Handle<reco::TrackCollection> tracks;
  ev.getByToken(inputCollectionToken, tracks);

  int nRegions = 0;
  for (auto it = tracks->cbegin(), ed = tracks->cend(); it != ed && nRegions < theMaxRegions; ++it) {
    result.push_back(region(*it, ev, es));
    nRegions++;
  }

  return result;
}

//
// Call region on Track from TrackRef
//
std::unique_ptr<RectangularEtaPhiTrackingRegion> MuonTrackingRegionByPtBuilder::region(
    const reco::TrackRef& track) const {
  return region(*track);
}

//
// ToDo: Not sure if this is needed?
//
void MuonTrackingRegionByPtBuilder::setEvent(const edm::Event& event, const edm::EventSetup& es) {
  theEvent = &event;
  theEventSetup = &es;
}

//
//	Main member function called to create the ROI
//
std::unique_ptr<RectangularEtaPhiTrackingRegion> MuonTrackingRegionByPtBuilder::region(
    const reco::Track& staTrack, const edm::Event& ev, const edm::EventSetup& es) const {
  // get track momentum/direction at vertex
  const math::XYZVector& mom = staTrack.momentum();
  GlobalVector dirVector(mom.x(), mom.y(), mom.z());
  double pt = staTrack.pt();

  // Fix for StandAlone tracks with low momentum
  const math::XYZVector& innerMomentum = staTrack.innerMomentum();
  GlobalVector forSmallMomentum(innerMomentum.x(), innerMomentum.y(), innerMomentum.z());
  if (staTrack.p() <= 1.5) {
    pt = std::abs(forSmallMomentum.perp());
  }

  // initial vertex position - in the following it is replaced with beamspot/vertexing
  GlobalPoint vertexPos(0.0, 0.0, 0.0);
  // standard 15.9, if useVertex than use error from  vertex
  double deltaZ = theHalfZ;

  // retrieve beam spot information
  edm::Handle<reco::BeamSpot> bs;
  bool bsHandleFlag = ev.getByToken(beamSpotToken, bs);

  // check the validity, otherwise vertexing
  if (bsHandleFlag && bs.isValid() && !useVertex) {
    vertexPos = GlobalPoint(bs->x0(), bs->y0(), bs->z0());
    deltaZ = useFixedZ ? theHalfZ : bs->sigmaZ() * theNsigmaDz;
  } else {
    // get originZPos from list of reconstructed vertices (first or all)
    edm::Handle<reco::VertexCollection> vertexCollection;
    bool vtxHandleFlag = ev.getByToken(vertexCollectionToken, vertexCollection);
    // check if there exists at least one reconstructed vertex
    if (vtxHandleFlag && !vertexCollection->empty()) {
      // use the first vertex in the collection and assume it is the primary event vertex
      reco::VertexCollection::const_iterator vtx = vertexCollection->begin();
      if (!vtx->isFake() && vtx->isValid()) {
        vertexPos = GlobalPoint(vtx->x(), vtx->y(), vtx->z());
        deltaZ = useFixedZ ? theHalfZ : vtx->zError() * theNsigmaDz;
      }
    }
  }

  // set delta eta and delta phi depending on the pt
  auto region_dEta = deltaEtas_.at(0);
  auto region_dPhi = deltaPhis_.at(0);
  if (pt < ptRanges_.back()) {
    auto lowEdge = std::upper_bound(ptRanges_.begin(), ptRanges_.end(), pt);
    region_dEta = deltaEtas_.at(lowEdge - ptRanges_.begin() - 1);
    region_dPhi = deltaPhis_.at(lowEdge - ptRanges_.begin() - 1);
  }

  float deltaR = theDeltaR;
  double minPt = useFixedPt ? thePtMin : std::max(thePtMin, pt * 0.6);

  const MeasurementTrackerEvent* measurementTracker = nullptr;
  if (!theMeasurementTrackerToken.isUninitialized()) {
    edm::Handle<MeasurementTrackerEvent> hmte;
    ev.getByToken(theMeasurementTrackerToken, hmte);
    measurementTracker = hmte.product();
  }

  const auto& bfield = es.getData(bfieldToken);
  const MultipleScatteringParametrisationMaker* msmaker = nullptr;
  if (thePrecise) {
    msmaker = &es.getData(msmakerToken);
  }

  auto region = std::make_unique<RectangularEtaPhiTrackingRegion>(dirVector,
                                                                  vertexPos,
                                                                  minPt,
                                                                  deltaR,
                                                                  deltaZ,
                                                                  region_dEta,
                                                                  region_dPhi,
                                                                  bfield,
                                                                  msmaker,
                                                                  thePrecise,
                                                                  theOnDemand,
                                                                  measurementTracker);

  LogDebug("MuonTrackingRegionByPtBuilder")
      << "the region parameters are:\n"
      << "\n dirVector: " << dirVector << "\n vertexPos: " << vertexPos << "\n minPt: " << minPt
      << "\n deltaR:" << deltaR << "\n deltaZ:" << deltaZ << "\n region_dEta:" << region_dEta
      << "\n region_dPhi:" << region_dPhi << "\n on demand parameter: " << static_cast<int>(theOnDemand);

  return region;
}

void MuonTrackingRegionByPtBuilder::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<double>("DeltaR", 0.2);
  desc.add<edm::InputTag>("beamSpot", edm::InputTag(""));
  desc.add<int>("OnDemand", -1);
  desc.add<edm::InputTag>("vertexCollection", edm::InputTag(""));
  desc.add<edm::InputTag>("MeasurementTrackerName", edm::InputTag(""));
  desc.add<bool>("UseVertex", false);
  desc.add<double>("Rescale_Dz", 3.0);
  desc.add<bool>("Pt_fixed", false);
  desc.add<bool>("Z_fixed", true);
  desc.add<double>("Pt_min", 1.5);
  desc.add<double>("DeltaZ", 15.9);
  desc.add<std::vector<double>>("ptRanges", {0., 1.e9});
  desc.add<std::vector<double>>("deltaEtas", {0.2});
  desc.add<std::vector<double>>("deltaPhis", {0.15});
  desc.add<int>("maxRegions", 1);
  desc.add<bool>("precise", true);
  desc.add<edm::InputTag>("input", edm::InputTag(""));
  descriptions.add("MuonTrackingRegionByPtBuilder", desc);

  descriptions.setComment(
      "Build a TrackingRegion around a standalone muon. Options to define region around beamspot or primary vertex and "
      "dynamic regions are included.");
}
