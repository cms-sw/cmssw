/** 
 *  Class: MuonTrackingRegionBuilder
 *
 *  Build a TrackingRegion around a standalone muon 
 *
 *  \author N. Neumeister   Purdue University
 *  \author A. Everett      Purdue University
 */

#include "RecoMuon/GlobalTrackingTools/interface/MuonTrackingRegionBuilder.h"

//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "RecoTracker/MeasurementDet/interface/MeasurementTrackerEvent.h"

#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"

//
// constructor
//
void MuonTrackingRegionBuilder::build(const edm::ParameterSet& par, edm::ConsumesCollector& iC) {

  // Adjust errors on Eta, Phi, Z
  theNsigmaEta  = par.getParameter<double>("Rescale_eta");
  theNsigmaPhi  = par.getParameter<double>("Rescale_phi");
  theNsigmaDz   = par.getParameter<double>("Rescale_Dz");

  // Upper limits parameters
  theEtaRegionPar1 = par.getParameter<double>("EtaR_UpperLimit_Par1"); 
  theEtaRegionPar2 = par.getParameter<double>("EtaR_UpperLimit_Par2");
  thePhiRegionPar1 = par.getParameter<double>("PhiR_UpperLimit_Par1");
  thePhiRegionPar2 = par.getParameter<double>("PhiR_UpperLimit_Par2");

  // Flag to switch to use Vertices instead of BeamSpot
  useVertex = par.getParameter<bool>("UseVertex");

  // Flag to use fixed limits for Eta, Phi, Z, pT
  useFixedZ = par.getParameter<bool>("Z_fixed");
  useFixedPt = par.getParameter<bool>("Pt_fixed");
  useFixedPhi = par.getParameter<bool>("Phi_fixed");
  useFixedEta = par.getParameter<bool>("Eta_fixed");

  // Minimum value for pT
  thePtMin = par.getParameter<double>("Pt_min");

  // Minimum value for Phi
  thePhiMin = par.getParameter<double>("Phi_min");

  // Minimum value for Eta
  theEtaMin = par.getParameter<double>("Eta_min");

  // The static region size along the Z direction
  theHalfZ  = par.getParameter<double>("DeltaZ");

  // The transverse distance of the region from the BS/PV
  theDeltaR = par.getParameter<double>("DeltaR");

  // The static region size in Eta
  theDeltaEta = par.getParameter<double>("DeltaEta");

  // The static region size in Phi
  theDeltaPhi = par.getParameter<double>("DeltaPhi");

  // Maximum number of regions to build when looping over Muons
  theMaxRegions = par.getParameter<int>("maxRegions");

  // Flag to use precise??
  thePrecise = par.getParameter<bool>("precise"); 

  // perigee reference point ToDo: Check this
  theOnDemand = RectangularEtaPhiTrackingRegion::intToUseMeasurementTracker(par.getParameter<int>("OnDemand"));
  if (theOnDemand != RectangularEtaPhiTrackingRegion::UseMeasurementTracker::kNever) {
    theMeasurementTrackerToken = iC.consumes<MeasurementTrackerEvent>(par.getParameter<edm::InputTag>("MeasurementTrackerName"));
  }

  // Vertex collection and Beam Spot
  beamSpotToken = iC.consumes<reco::BeamSpot>(par.getParameter<edm::InputTag>("beamSpot"));
  vertexCollectionToken = iC.consumes<reco::VertexCollection>(par.getParameter<edm::InputTag>("vertexCollection"));

  // Input muon collection
  inputCollectionToken = iC.consumes<reco::TrackCollection>(par.getParameter<edm::InputTag>("input"));
}


//
// Member function to be compatible with TrackingRegionProducerFactory: create many ROI for many tracks
//
std::vector<std::unique_ptr<TrackingRegion>> MuonTrackingRegionBuilder::regions(const edm::Event& ev, const edm::EventSetup& es) const {
	
  std::vector<std::unique_ptr<TrackingRegion>> result;

  edm::Handle<reco::TrackCollection> tracks;
  ev.getByToken(inputCollectionToken, tracks);

  int nRegions = 0;
  for (auto it = tracks->cbegin(), ed = tracks->cend(); it != ed && nRegions < theMaxRegions; ++it) {
    result.push_back(region(*it,ev));
    nRegions++; 
  }

  return result;

}


//
// Call region on Track from TrackRef
//
std::unique_ptr<RectangularEtaPhiTrackingRegion> MuonTrackingRegionBuilder::region(const reco::TrackRef& track) const {
  return region(*track);
}


//
// ToDo: Not sure if this is needed?
//
void MuonTrackingRegionBuilder::setEvent(const edm::Event& event) {
  theEvent = &event;
}


//
//	Main member function called to create the ROI
//
std::unique_ptr<RectangularEtaPhiTrackingRegion> MuonTrackingRegionBuilder::region(const reco::Track& staTrack, const edm::Event& ev) const {

  // get track momentum/direction at vertex
  const math::XYZVector& mom = staTrack.momentum();
  GlobalVector dirVector(mom.x(),mom.y(),mom.z());
  double pt = staTrack.pt();

  // Fix for StandAlone tracks with low momentum
  const math::XYZVector& innerMomentum = staTrack.innerMomentum();
  GlobalVector forSmallMomentum(innerMomentum.x(),innerMomentum.y(),innerMomentum.z());
  if ( staTrack.p() <= 1.5 ) {
    pt = std::abs(forSmallMomentum.perp());
  }

  // initial vertex position - in the following it is replaced with beamspot/vertexing
  GlobalPoint vertexPos(0.0,0.0,0.0);
  // standard 15.9, if useVertex than use error from  vertex
  double deltaZ = theHalfZ;

  // retrieve beam spot information
  edm::Handle<reco::BeamSpot> bs;
  bool bsHandleFlag = ev.getByToken(beamSpotToken, bs);

  // check the validity, otherwise vertexing
  if ( bsHandleFlag && bs.isValid() && !useVertex ) {
    vertexPos = GlobalPoint(bs->x0(), bs->y0(), bs->z0());
    deltaZ = useFixedZ ? theHalfZ : bs->sigmaZ() * theNsigmaDz;
  } else {
    // get originZPos from list of reconstructed vertices (first or all)
    edm::Handle<reco::VertexCollection> vertexCollection;
    bool vtxHandleFlag = ev.getByToken(vertexCollectionToken, vertexCollection);
    // check if there exists at least one reconstructed vertex
    if ( vtxHandleFlag && !vertexCollection->empty() ) {
      // use the first vertex in the collection and assume it is the primary event vertex 
      reco::VertexCollection::const_iterator vtx = vertexCollection->begin();
      if (!vtx->isFake() && vtx->isValid() ) {
        vertexPos = GlobalPoint(vtx->x(),vtx->y(),vtx->z());
        deltaZ = useFixedZ ? theHalfZ : vtx->zError() * theNsigmaDz;
      }
    }
  }

  // inizialize to the maximum possible value
  double deta = 0.4;
  double dphi = 0.6;

  // evaluate the dynamical region if possible
  deta = theNsigmaEta*(staTrack.etaError());
  dphi = theNsigmaPhi*(staTrack.phiError());

  // Region_Parametrizations to take into account possible L2 error matrix inconsistencies
  double region_dEta = 0;
  double region_dPhi = 0;
  double eta = 0;
  double phi = 0;

  // eta, pt parametrization from MC study (circa 2009?)
  if ( pt <= 10. ) {
     // angular coefficients
     float acoeff_Phi = (thePhiRegionPar2 - thePhiRegionPar1)/5;
     float acoeff_Eta = (theEtaRegionPar2 - theEtaRegionPar1)/5;

     eta = theEtaRegionPar1 + (acoeff_Eta)*(pt-5.);
     phi = thePhiRegionPar1 + (acoeff_Phi)*(pt-5.) ;
  }
  // parametrization 2nd bin in pt from MC study  
  if ( pt > 10. && pt < 100. ) {
     eta = theEtaRegionPar2;
     phi = thePhiRegionPar2;
  }
  // parametrization 3rd bin in pt from MC study
  if ( pt >= 100. ) {
     // angular coefficients
     float acoeff_Phi = (thePhiRegionPar1 - thePhiRegionPar2)/900;
     float acoeff_Eta = (theEtaRegionPar1 - theEtaRegionPar2)/900;

     eta = theEtaRegionPar2 + (acoeff_Eta)*(pt-100.);
     phi = thePhiRegionPar2 + (acoeff_Phi)*(pt-100.);
  }

  double region_dPhi1 = std::min(phi,dphi);
  double region_dEta1 = std::min(eta,deta);

  // decide to use either a parametrization or a dynamical region
  region_dPhi = useFixedPhi ? theDeltaPhi : std::max(thePhiMin,region_dPhi1);
  region_dEta = useFixedEta ? theDeltaEta : std::max(theEtaMin,region_dEta1);

  float deltaR = theDeltaR;
  double minPt = useFixedPt ? thePtMin : std::max(thePtMin,pt*0.6);


  const MeasurementTrackerEvent* measurementTracker = nullptr;
  if (!theMeasurementTrackerToken.isUninitialized()) {
    edm::Handle<MeasurementTrackerEvent> hmte;
    ev.getByToken(theMeasurementTrackerToken, hmte);
    measurementTracker = hmte.product();
  }

  auto region = std::make_unique<RectangularEtaPhiTrackingRegion>(dirVector, vertexPos,
                                               minPt, deltaR,
                                               deltaZ, region_dEta, region_dPhi,
                                               theOnDemand,
                                               thePrecise,
                                               measurementTracker);

  LogDebug("MuonTrackingRegionBuilder")<<"the region parameters are:\n"
				       <<"\n dirVector: "<<dirVector
				       <<"\n vertexPos: "<<vertexPos
				       <<"\n minPt: "<<minPt
				       <<"\n deltaR:"<<deltaR
				       <<"\n deltaZ:"<<deltaZ
				       <<"\n region_dEta:"<<region_dEta
				       <<"\n region_dPhi:"<<region_dPhi
				       <<"\n on demand parameter: "<<static_cast<int>(theOnDemand);
  
  return region;

}

void MuonTrackingRegionBuilder::fillDescriptions(edm::ParameterSetDescription& descriptions) {
  {
    edm::ParameterSetDescription desc;
    desc.add<double>("EtaR_UpperLimit_Par1",0.25);
    desc.add<double>("DeltaR",0.2);
    desc.add<edm::InputTag>("beamSpot",edm::InputTag(""));
    desc.add<int>("OnDemand",-1);
    desc.add<edm::InputTag>("vertexCollection",edm::InputTag(""));
    desc.add<double>("Rescale_phi",3.0);
    desc.add<bool>("Eta_fixed",false);
    desc.add<double>("Rescale_eta",3.0);
    desc.add<double>("PhiR_UpperLimit_Par2",0.2);
    desc.add<double>("Eta_min",0.05);
    desc.add<bool>("Phi_fixed",false);
    desc.add<double>("Phi_min",0.05);
    desc.add<double>("PhiR_UpperLimit_Par1",0.6);
    desc.add<double>("EtaR_UpperLimit_Par2",0.15);
    desc.add<edm::InputTag>("MeasurementTrackerName",edm::InputTag(""));
    desc.add<bool>("UseVertex",false);
    desc.add<double>("Rescale_Dz",3.0);
    desc.add<bool>("Pt_fixed",false);
    desc.add<bool>("Z_fixed",true);
    desc.add<double>("Pt_min",1.5);
    desc.add<double>("DeltaZ",15.9);
    desc.add<double>("DeltaEta",0.2);
    desc.add<double>("DeltaPhi",0.2);
    desc.add<int>("maxRegions",1);
    desc.add<bool>("precise",true);
    desc.add<edm::InputTag>("input",edm::InputTag(""));
    descriptions.add("MuonTrackingRegionBuilder",desc);
  }
  {
    edm::ParameterSetDescription desc;
    desc.add<double>("EtaR_UpperLimit_Par1",0.25);
    desc.add<double>("DeltaR",0.2);
    desc.add<edm::InputTag>("beamSpot",edm::InputTag("hltOnlineBeamSpot"));
    desc.add<int>("OnDemand",-1);
    desc.add<edm::InputTag>("vertexCollection",edm::InputTag("pixelVertices"));
    desc.add<double>("Rescale_phi",3.0);
    desc.add<bool>("Eta_fixed",false);
    desc.add<double>("Rescale_eta",3.0);
    desc.add<double>("PhiR_UpperLimit_Par2",0.2);
    desc.add<double>("Eta_min",0.05);
    desc.add<bool>("Phi_fixed",false);
    desc.add<double>("Phi_min",0.05);
    desc.add<double>("PhiR_UpperLimit_Par1",0.6);
    desc.add<double>("EtaR_UpperLimit_Par2",0.15);
    desc.add<edm::InputTag>("MeasurementTrackerName",edm::InputTag("hltESPMeasurementTracker"));
    desc.add<bool>("UseVertex",false);
    desc.add<double>("Rescale_Dz",3.0);
    desc.add<bool>("Pt_fixed",false);
    desc.add<bool>("Z_fixed",true);
    desc.add<double>("Pt_min",1.5);
    desc.add<double>("DeltaZ",15.9);
    desc.add<double>("DeltaEta",0.2);
    desc.add<double>("DeltaPhi",0.2);
    desc.add<int>("maxRegions",1);
    desc.add<bool>("precise",true);
    desc.add<edm::InputTag>("input",edm::InputTag("hltL2Muons","UpdatedAtVtx"));
    descriptions.add("hltMuonTrackingRegionBuilder",desc);
  }
  descriptions.setComment("Build a TrackingRegion around a standalone muon. Options to define region around beamspot or primary vertex and dynamic regions are included.");
}
