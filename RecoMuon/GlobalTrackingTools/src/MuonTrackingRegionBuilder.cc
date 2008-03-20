/** 
 *  Class: MuonTrackingRegionBuilder
 *
 *  Build a TrackingRegion around a standalone muon 
 *
 *  $Date: 2008/03/05 21:12:55 $
 *  $Revision: 1.7 $
 *
 *  \author A. Everett - Purdue University
 *  \author A. Grelli -  Purdue University, Pavia University
 */

#include "RecoMuon/GlobalTrackingTools/interface/MuonTrackingRegionBuilder.h"

//---------------
// C++ Headers --
//---------------


//-------------------------------
// Collaborating Class Headers --
//-------------------------------

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

using namespace std;

//
// constructor
//
MuonTrackingRegionBuilder::MuonTrackingRegionBuilder(const edm::ParameterSet& par, 
                                                     const MuonServiceProxy* service) :
 theService(service) {

  // vertex Collection and Beam Spot
  theBeamSpotTag = par.getParameter<edm::InputTag>("beamSpot");
  theVertexCollTag = par.getParameter<edm::InputTag>("vertexCollection");

  // parmeters
  theNsigmaEta  = par.getParameter<double>("Rescale_eta");
  theNsigmaPhi  = par.getParameter<double>("Rescale_phi");
  theNsigmaDz   = par.getParameter<double>("Rescale_Dz");
  theTkEscapePt = par.getParameter<double>("EscapePt");

  // upper limits parameters   
  theEtaRegionPar1 = par.getParameter<double>("EtaR_UpperLimit_Par1"); 
  theEtaRegionPar2 = par.getParameter<double>("EtaR_UpperLimit_Par2");
  thePhiRegionPar1 = par.getParameter<double>("PhiR_UpperLimit_Par1");
  thePhiRegionPar2 = par.getParameter<double>("PhiR_UpperLimit_Par2");

  useVertex = par.getParameter<bool>("UseVertex");
  useFixedRegion = par.getParameter<bool>("UseFixedRegion");

  // fixed limits
  thePhiFixed = par.getParameter<double>("Phi_fixed");
  theEtaFixed = par.getParameter<double>("Eta_fixed");

  thePhiMin = par.getParameter<double>("Phi_min");
  theEtaMin = par.getParameter<double>("Eta_min");
  theHalfZ  = par.getParameter<double>("DeltaZ_Region");
  theDeltaR = par.getParameter<double>("DeltaR");

  // perigee reference point
  theVertexPos = GlobalPoint(0.0,0.0,0.0);

}


//
//
//
RectangularEtaPhiTrackingRegion* 
MuonTrackingRegionBuilder::region(const reco::TrackRef& track) const {

  return region(*track);

}


//
//
//
void MuonTrackingRegionBuilder::setEvent(const edm::Event& event) {
  
  theEvent = &event;

}


//
//
//
RectangularEtaPhiTrackingRegion* 
MuonTrackingRegionBuilder::region(const reco::Track& staTrack) const {

  // get the free trajectory state of the muon updated at vertex
  TSCPBuilderNoMaterial tscpBuilder; 
  TrajectoryStateTransform tsTransform;
  FreeTrajectoryState muFTS = tsTransform.initialFreeState(staTrack,&*theService->magneticField());
   
  // get track direction at vertex
  GlobalVector dirVector(muFTS.momentum());

  // get track momentum
  const math::XYZVector& mo = staTrack.innerMomentum();
  GlobalVector mom(mo.x(),mo.y(),mo.z());
  if ( staTrack.p() > 1.5 ) {
    mom = dirVector; 
  }

  // initial vertex position -  in the following it is replaced with beam spot/vertexing
  GlobalPoint vertexPos(0.0,0.0,0.0);
  double deltaZatVTX = 0.0;

  // retrieve beam spot information
  edm::Handle<reco::BeamSpot> bsHandle;
  bool bsHandleFlag = theEvent->getByLabel(theBeamSpotTag, bsHandle);
  // check the validity, otherwise vertexing
  if ( bsHandleFlag && !useVertex ) {
    const reco::BeamSpot& bs = *bsHandle;
    vertexPos = GlobalPoint(bs.x0(), bs.y0(), bs.z0());
  } else {
    // get originZPos from list of reconstructed vertices (first or all)
    edm::Handle<reco::VertexCollection> vertexCollection;
    bool vtxHandleFlag = theEvent->getByLabel(theVertexCollTag,vertexCollection);
    // check if there exists at least one reconstructed vertex
    if ( vtxHandleFlag && !vertexCollection->empty() ) {
      // use the first vertex in the collection and assume it is the primary event vertex 
      reco::VertexCollection::const_iterator vtx = vertexCollection->begin();
      vertexPos = GlobalPoint(0.0,0.0,vtx->z());
      // delta Z from vertex error
      deltaZatVTX = vtx->zError() * theNsigmaDz;
    }
  }

  TrajectoryStateClosestToPoint tscp = tscpBuilder(muFTS,theVertexPos);
  const PerigeeTrajectoryError& covar = tscp.perigeeError();
  const PerigeeTrajectoryParameters& param = tscp.perigeeParameters();

  // calculate deltaEta from deltaTheta
  double deltaTheta = covar.thetaError();
  double theta      = param.theta();
  double sin_theta  = sin(theta);

  // get dEta and dPhi
  double deta = theNsigmaEta*(1/fabs(sin_theta))*deltaTheta;
  double dphi = theNsigmaPhi*(covar.phiError());

  /* Region_Parametrizations to take into account possible 
     L2 error matrix inconsistencies. Detailed Explanation in TWIKI
     page.
  */
  double region_dEta = 0;
  double region_dPhi = 0;
  double eta,phi;

  // eta, pt parametrization from MC study
  float pt = abs(mom.perp());
  if ( pt <= 10. ) {
     // angular coefficients
     float acoeff_Phi = (thePhiRegionPar2 - thePhiRegionPar1)/5;
     float acoeff_Eta = (theEtaRegionPar2 - theEtaRegionPar1)/5;

     eta = theEtaRegionPar1 + (acoeff_Eta)*(mom.perp()-5.);
     phi = thePhiRegionPar1 + (acoeff_Phi)*(mom.perp()-5.) ;
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

     eta = theEtaRegionPar2 + (acoeff_Eta)*(mom.perp()-100.);
     phi = thePhiRegionPar2 + (acoeff_Phi)*(mom.perp()-100.);
   }

  // decide to use either a parametrization or a dynamical region
  double region_dPhi1 = min(phi,dphi);
  double region_dEta1 = min(eta,deta);

  // minimum size
  region_dPhi = max(thePhiMin,region_dPhi1);
  region_dEta = max(theEtaMin,region_dEta1);

  float deltaZ = 0.0;
  // standard 15.9 is useVertex than region from vertexing
  if ( useVertex ) {
    deltaZ = deltaZatVTX;
  } else { 
    deltaZ = theHalfZ;
  }

  float deltaR = theDeltaR;
  double minPt = max(theTkEscapePt,mom.perp()*0.6);

  RectangularEtaPhiTrackingRegion* region = 0;  

  if (useFixedRegion) {
     region_dEta = theEtaFixed;
     region_dPhi = thePhiFixed;
  }

  region = new RectangularEtaPhiTrackingRegion(dirVector, vertexPos,
                                               minPt, deltaR,
                                               deltaZ, region_dEta, region_dPhi);
  
  return region;
  
}
