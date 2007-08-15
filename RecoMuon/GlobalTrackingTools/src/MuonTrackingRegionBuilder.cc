/** \class MuonTrackingRegionBuilder
 *  Base class for the Muon reco TrackingRegion Builder
 *
 *  $Date: 2007/05/09 19:28:21 $
 *  $Revision: 1.1 $
 *  \author A. Everett - Purdue University
 */

#include "RecoMuon/GlobalTrackingTools/interface/MuonTrackingRegionBuilder.h"

#include "RecoMuon/TrackingTools/interface/MuonServiceProxy.h"

//#include "DataFormats/TrackReco/interface/Track.h"
//#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackExtraFwd.h"

#include "RecoTracker/TkTrackingRegions/interface/GlobalTrackingRegion.h"
#include "RecoTracker/TkTrackingRegions/interface/RectangularEtaPhiTrackingRegion.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h"

//#include <cmath>

using namespace std;

/// constructor
MuonTrackingRegionBuilder::MuonTrackingRegionBuilder(const edm::ParameterSet& par, const MuonServiceProxy* service)
  : theService(service)
{
  theMakeTkSeedFlag = par.getParameter<bool>("RegionalSeedFlag");

  theVertexPos = GlobalPoint(0.0,0.0,0.0);
  theVertexErr = GlobalError(0.0001,0.0,0.0001,0.0,0.0,28.09);

}

RectangularEtaPhiTrackingRegion* MuonTrackingRegionBuilder::region(const reco::TrackRef& track) const
{
  return region(*track);
}

RectangularEtaPhiTrackingRegion* MuonTrackingRegionBuilder::region(const reco::Track& staTrack) const
{
 
  //Get muon free state updated at vertex
  TrajectoryStateTransform tsTransform;
  FreeTrajectoryState muFTS = tsTransform.initialFreeState(staTrack,&*theService->magneticField());
  
  //Get track direction at vertex
  GlobalVector dirVector(muFTS.momentum());
  
  //Get region size using momentum uncertainty
  
  //Get track momentum
  const math::XYZVector& mo = staTrack.innerMomentum();
  GlobalVector mom(mo.x(),mo.y(),mo.z());
  if ( staTrack.p() > 1.0 ) {
    mom = dirVector; 
  }
  
  //Get Mu state on inner muon surface
  //TrajectoryStateOnSurface muTSOS = tsTransform.innerStateOnSurface(*staTrack,*theService->trackingGeometry(),&*theService->magneticField());
  
  //Get Mu state on tracker bound
  //StateOnTrackerBound fromInside(&*theService->propagator(stateOnTrackerOutProp));
  //muTSOS = fromInside(muFTS);
  
  //Get error of momentum of the Mu state
  GlobalError  dirErr(muFTS.cartesianError().matrix().Sub<AlgebraicSymMatrix33>(3,3));
  GlobalVector dirVecErr(dirVector.x() + sqrt(dirErr.cxx()),
			 dirVector.y() + sqrt(dirErr.cyy()),
			 dirVector.z() + sqrt(dirErr.czz()));
  
  //Get dEta and dPhi
  float eta1 = dirVector.eta();
  float eta2 = dirVecErr.eta();
  float deta(fabs(eta1- eta2));
  float dphi(fabs(Geom::Phi<float>(dirVector.phi())-Geom::Phi<float>(dirVecErr.phi())));
  
  //Get vertex, Pt constraints  
  GlobalPoint vertexPos = (muFTS.position());
  GlobalError vertexErr = (muFTS.cartesianError().position());
  
  double minPt    = max(1.5,mom.perp()*0.6);
  double deltaZ   = min(15.9,3*sqrt(theVertexErr.czz()));
  
  //Adjust tracking region dEta and dPhi  
  double deltaEta = 0.1;
  double deltaPhi = 0.1;

  if ( deta > 0.05 ) {
    deltaEta += deta/2;
  }
  if ( dphi > 0.07 ) {
    deltaPhi += 0.15;
  }

  deltaPhi = min(double(0.2), deltaPhi);
  if(mom.perp() < 25.) deltaPhi = max(double(dphi),0.3);
  if(mom.perp() < 10.) deltaPhi = max(deltaPhi,0.8);
 
  deltaEta = min(double(0.2), deltaEta);
  if( mom.perp() < 6.0 ) deltaEta = 0.5;
  if( fabs(eta1) > 2.25 ) deltaEta = 0.6;
  if( fabs(eta1) > 3.0 ) deltaEta = 1.0;
  //if( fabs(eta1) > 2. && mom.perp() < 10. ) deltaEta = 1.;
  //if ( fabs(eta1) < 1.25 && fabs(eta1) > 0.8 ) deltaEta= max(0.07,deltaEta);
  if ( fabs(eta1) < 1.3  && fabs(eta1) > 1.0 ) deltaPhi = max(0.3,deltaPhi);

  deltaEta = min(double(1.), 1.25 * deltaEta);
  deltaPhi = 1.2 * deltaPhi;
  
  //Get region size using position uncertainty
  
  //Get innerMu position
  const math::XYZPoint& po = staTrack.innerPosition();
  GlobalPoint pos(po.x(),po.y(),po.z());    
  //pos = muTSOS.globalPosition();
  
  float eta3 = pos.eta();
  float deta2(fabs(eta1- eta3));
  float dphi2(fabs(Geom::Phi<float>(dirVector.phi())-Geom::Phi<float>(pos.phi())));  
     
  //Adjust tracking region dEta dPhi
  double deltaEta2 = 0.05;
  double deltaPhi2 = 0.07;
    
  if ( deta2 > 0.05 ) {
    deltaEta2 += deta2 / 2;
  }
  if ( dphi2 > 0.07 ) {
    deltaPhi2 += 0.15;
    if ( fabs(eta3) < 1.0 && mom.perp() < 6. ) deltaPhi2 = dphi2;
  }
  if ( fabs(eta1) < 1.25 && fabs(eta1) > 0.8 ) deltaEta2=max(0.07,deltaEta2);
  if ( fabs(eta1) < 1.3  && fabs(eta1) > 1.0 ) deltaPhi2=max(0.3,deltaPhi2);
  
  deltaEta2 = 1 * max(double(2.5 * deta2),deltaEta2);
  deltaPhi2 = 1 * max(double(3.5 * dphi2),deltaPhi2);
  
  //Use whichever will give smallest region size
  deltaEta = min(deltaEta,deltaEta2);
  deltaPhi = min(deltaPhi,deltaPhi2);

  if( theMakeTkSeedFlag ) {
    deltaEta = deltaEta2;
    deltaPhi = deltaPhi2;
    vertexPos = theVertexPos;
  }
  
  RectangularEtaPhiTrackingRegion * region = 0;  

  region = new RectangularEtaPhiTrackingRegion(dirVector, vertexPos,
                                             minPt, 0.2,
                                             deltaZ, deltaEta, deltaPhi);
  
  return region;
  


}
