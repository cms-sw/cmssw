/** 
 *  Class: MuonTrackingRegionBuilder
 *
 *  Build a TrackingRegion around a standalone muon 
 *
 *  $Date: 2011/12/23 08:13:36 $
 *  $Revision: 1.20 $
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
#include "TrackingTools/PatternTools/interface/TSCBLBuilderNoMaterial.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

using namespace std;

//
// constructor
//

void MuonTrackingRegionBuilder::init(const MuonServiceProxy* service) { theService= service;}
MuonTrackingRegionBuilder::MuonTrackingRegionBuilder(const edm::ParameterSet& par)
{
  build(par);
}
void MuonTrackingRegionBuilder::build(const edm::ParameterSet& par){
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

  theOnDemand = par.getParameter<double>("OnDemand");
  theMeasurementTrackerName = par.getParameter<std::string>("MeasurementTrackerName");
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
  
  FreeTrajectoryState muFTS = trajectoryStateTransform::initialFreeState(staTrack,&*theService->magneticField());

  LogDebug("MuonTrackingRegionBuilder")<<"from state: "<<muFTS;

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
  // inizialization of BS

  if ( bsHandleFlag && !useVertex ) {
    const reco::BeamSpot& bs =  *bsHandle;
    vertexPos = GlobalPoint(bs.x0(), bs.y0(), bs.z0());
  } else {
    // get originZPos from list of reconstructed vertices (first or all)
    edm::Handle<reco::VertexCollection> vertexCollection;
    bool vtxHandleFlag = theEvent->getByLabel(theVertexCollTag,vertexCollection);
    // check if there exists at least one reconstructed vertex
    if ( vtxHandleFlag && !vertexCollection->empty() ) {
      // use the first vertex in the collection and assume it is the primary event vertex 
      reco::VertexCollection::const_iterator vtx = vertexCollection->begin();
      vertexPos = GlobalPoint(vtx->x(),vtx->y(),vtx->z());
      // delta Z from vertex error
      deltaZatVTX = vtx->zError() * theNsigmaDz;
    }
  }


 // inizialize to the maximum possible value to avoit 
 // problems with TSCBL

  double deta = 0.4;
  double dphi = 0.6;

  // take into account the correct beanspot rotation
  if ( bsHandleFlag ) {

  const reco::BeamSpot& bs =  *bsHandle;

  TSCBLBuilderNoMaterial tscblBuilder;
  TrajectoryStateClosestToBeamLine tscbl = tscblBuilder(muFTS,bs);

 
    // evaluate the dynamical region if possible
    if(tscbl.isValid()){

      PerigeeTrajectoryError trackPerigeeErrors = PerigeeConversions::ftsToPerigeeError(tscbl.trackStateAtPCA());
      GlobalVector pTrack = tscbl.trackStateAtPCA().momentum();

    // calculate deltaEta from deltaTheta
      double deltaTheta = trackPerigeeErrors.thetaError();
      double theta      = pTrack.theta();
      double sin_theta  = sin(theta);

    // get dEta and dPhi
      deta = theNsigmaEta*(1/fabs(sin_theta))*deltaTheta;
      dphi = theNsigmaPhi*(trackPerigeeErrors.phiError());

    }
 }

  /* Region_Parametrizations to take into account possible 
     L2 error matrix inconsistencies. Detailed Explanation in TWIKI
     page.
  */
  double region_dEta = 0;
  double region_dPhi = 0;
  double eta=0; double phi=0;

  // eta, pt parametrization from MC study
  float pt = fabs(mom.perp());

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
    deltaZ = max(theHalfZ,deltaZatVTX);
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
                                               deltaZ, region_dEta, region_dPhi,
					       theOnDemand,
					       true,/*default in the header*/
					       theMeasurementTrackerName);

  LogDebug("MuonTrackingRegionBuilder")<<"the region parameters are:\n"
				       <<"\n dirVector: "<<dirVector
				       <<"\n vertexPos: "<<vertexPos
				       <<"\n minPt: "<<minPt
				       <<"\n deltaR:"<<deltaR
				       <<"\n deltaZ:"<<deltaZ
				       <<"\n region_dEta:"<<region_dEta
				       <<"\n region_dPhi:"<<region_dPhi
				       <<"\n on demand parameter: "<<theOnDemand;

  
  return region;
  
}
