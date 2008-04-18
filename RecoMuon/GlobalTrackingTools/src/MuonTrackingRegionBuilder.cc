/** \class MuonTrackingRegionBuilder
 *  Base class for the Muon reco TrackingRegion Builder
 *
 *  $Date: 2008/03/05 21:12:55 $
 *  $Revision: 1.7 $
 *  \author A. Everett - Purdue University
    \author A. Grelli -  Purdue University, Pavia University
 */
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoMuon/GlobalTrackingTools/interface/MuonTrackingRegionBuilder.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"
#include "TrackingTools/PatternTools/interface/TSCPBuilderNoMaterial.h"

//Beam Spot option & vertexing
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

using namespace std;

/// constructor
MuonTrackingRegionBuilder::MuonTrackingRegionBuilder(const edm::ParameterSet& par, const MuonServiceProxy* service)
  : theService(service)
  {

  // Vertex Collection and Beam Spot
  theBeamSpotTag = par.getParameter<edm::InputTag>("beamSpot");
  theVertexCollTag = par.getParameter<edm::InputTag>("vertexCollection");
  usePixelVertex = par.getParameter<bool>("UseVertex");

  // Parmeters
  Nsigma_eta     = par.getParameter<double>("Rescale_eta");
  Nsigma_phi     = par.getParameter<double>("Rescale_phi");
  Nsigma_Dz      = par.getParameter<double>("Rescale_Dz");

  TkEscapePt     = par.getParameter<double>("EscapePt");

  //Upper Limits Parameters   
  Eta_Region_parameter1 = par.getParameter<double>("EtaR_UpperLimit_Par1"); 
  Eta_Region_parameter2 = par.getParameter<double>("EtaR_UpperLimit_Par2");
  Phi_Region_parameter1 = par.getParameter<double>("PhiR_UpperLimit_Par1");
  Phi_Region_parameter2 = par.getParameter<double>("PhiR_UpperLimit_Par2");

  //Fixed limits
  theFixedFlag     = par.getParameter<bool>("UseFixedRegion");
  Eta_fixed        = par.getParameter<double>("Etafixed");
  Phi_fixed        = par.getParameter<double>("Phifixed");
 
  Phi_minimum      = par.getParameter<double>("Phi_min");
  Eta_minimum      = par.getParameter<double>("Eta_min");
  HalfZRegion_size = par.getParameter<double>("DeltaZ_Region");
  Delta_R_Region   = par.getParameter<double>("DeltaR");

  //Perigee Reference point
  theVertexPos = GlobalPoint(0.0,0.0,0.0);
}

RectangularEtaPhiTrackingRegion* MuonTrackingRegionBuilder::region(const reco::TrackRef& track) const
{
  return region(*track);
}

void MuonTrackingRegionBuilder::setEvent(const edm::Event& event) {
  
  theEvent = &event;
}


RectangularEtaPhiTrackingRegion* MuonTrackingRegionBuilder::region(const reco::Track& staTrack) const
{
  TSCPBuilderNoMaterial tscpBuilder; 
  //Get muon free state updated at vertex
  TrajectoryStateTransform tsTransform;
  FreeTrajectoryState muFTS = tsTransform.initialFreeState(staTrack,&*theService->magneticField());
   
  //Get track direction at vertex
  GlobalVector dirVector(muFTS.momentum());

  //Get track momentum
  const math::XYZVector& mo = staTrack.innerMomentum();
  GlobalVector mom(mo.x(),mo.y(),mo.z());
  if ( staTrack.p() > 1.5 ) {
    mom = dirVector; 
  }

  // Inizial vertex position (in the following replaced with beam spot/vertexing)
  GlobalPoint vertexPos(0.0,0.0,0.0);

  double TheDeltaZatVTX = 0;

  //retrieve beam spot information
  edm::Handle<reco::BeamSpot> bsHandle;
  bool bsHandleFlag = theEvent->getByLabel(theBeamSpotTag, bsHandle);
  //cechk the validity otherwise vertexing
  if(bsHandleFlag && !usePixelVertex) {
    const reco::BeamSpot & bs = *bsHandle;
    GlobalPoint vertexPosBS(bs.x0(), bs.y0(), bs.z0());
    vertexPos = vertexPosBS;
  }else{
  // Get originZPos from list of vertices (first or all)
    edm::Handle<reco::VertexCollection> vertexCollection;
    bool vtxHandleFlag = theEvent->getByLabel(theVertexCollTag,vertexCollection);
  // ceck if exsist a non empty vertex collection
    if(vtxHandleFlag && vertexCollection->size() > 0) {
     reco::VertexCollection::const_iterator Vtx=vertexCollection->begin();// ! only the first low lomi option
     // const Vtx = vertexCollection->front();
      double TheZPosition = Vtx->z();

      GlobalPoint vertexPosVT(0.0,0.0,TheZPosition);
      vertexPos = vertexPosVT;

      // delta Z from vertex error
      TheDeltaZatVTX = (Vtx->zError())*Nsigma_Dz;

     }
   }


  TrajectoryStateClosestToPoint tscp = tscpBuilder(muFTS,theVertexPos);

  const PerigeeTrajectoryError & covar = tscp.perigeeError();
  const PerigeeTrajectoryParameters & param = tscp.perigeeParameters();

  //calculate deltaEta from deltaTheta
  double deltaTheta = covar.thetaError();
  double theta      = param.theta();
  double sin_theta  = sin(theta);

  //Get dEta and dPhi
  double deta = Nsigma_eta*(1/fabs(sin_theta))*deltaTheta;
  double dphi = Nsigma_phi*(covar.phiError());


  /* Region_Parametrizations to take into account possible 
     L2 matrix inconsistencies. Detailed Explanation in TWIKI
     page.
  */
  double region_dEta = 0;
  double region_dPhi = 0;

  double region_dEta1,region_dPhi1,Par_eta,Par_phi;
  float  acoeff1_Phi,acoeff1_Eta,acoeff3_Phi,acoeff3_Eta;

  // Eta , ptparametrization as in MC study
   if(abs(mom.perp())<=10.){
     // angolar coefficients
     acoeff1_Phi = (Phi_Region_parameter2-Phi_Region_parameter1)/5;
     acoeff1_Eta = (Eta_Region_parameter2-Eta_Region_parameter1)/5;

     Par_eta  = Eta_Region_parameter1 + (acoeff1_Eta)*(mom.perp()-5.);
     Par_phi  = Phi_Region_parameter1 + (acoeff1_Phi)*(mom.perp()-5.) ;
   }

   //Parametrization 2nd bin in pt from MC study  
   if(abs(mom.perp())>10. && abs(mom.perp())<100.){

     Par_eta = Eta_Region_parameter2;
     Par_phi = Phi_Region_parameter2;
   }
   //Parametrization 3rd bin in pt from MC study
   if(abs(mom.perp())>=100.){
     // angolar coefficients
     acoeff3_Phi = (Phi_Region_parameter1-Phi_Region_parameter2)/900;
     acoeff3_Eta = (Eta_Region_parameter1-Eta_Region_parameter2)/900;

     Par_eta = Eta_Region_parameter2 + (acoeff3_Eta)*(mom.perp()-100.);
     Par_phi = Phi_Region_parameter2 + (acoeff3_Phi)*(mom.perp()-100.);
   }

  // here decide to use parametrization or dinamical region
  region_dPhi1 = min(Par_phi,dphi);
  region_dEta1 = min(Par_eta,deta);

  // minimum size
  region_dPhi = max(Phi_minimum,region_dPhi1);
  region_dEta = max(Eta_minimum,region_dEta1);

  float deltaZ=0;
  //standard 15.9 is useVertex than region from vertexing
  if(usePixelVertex){
    deltaZ  = TheDeltaZatVTX;
  } else { deltaZ = HalfZRegion_size; }

  float deltaR  = Delta_R_Region;
  double minPt   = max(TkEscapePt,mom.perp()*0.6);
  
  RectangularEtaPhiTrackingRegion * region = 0;  


  if(theFixedFlag) {
     region_dEta = Eta_fixed;
     region_dPhi = Phi_fixed;
  }

  region = new RectangularEtaPhiTrackingRegion(dirVector, vertexPos,
                                               minPt, deltaR,
                                               deltaZ, region_dEta, region_dPhi);
  
  return region;
  


}

