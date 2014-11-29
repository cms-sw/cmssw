#include <iostream>
#include <vector>
#include <memory>
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionHitChecker.h"
// Framework
//
#include "TrackingTools/PatternTools/interface/Trajectory.h"

//


std::pair<uint8_t,Measurement1DFloat> ConversionHitChecker::nHitsBeforeVtx(const Trajectory &traj, const reco::Vertex &vtx, double sigmaTolerance) const {

  const std::vector<TrajectoryMeasurement> &measurements = traj.measurements();

  //protect for empty trajectory
  if (measurements.size()==0) {
    return std::pair<unsigned int,Measurement1DFloat>(0,Measurement1DFloat());
  }

  //check ordering of measurements in trajectory
  bool inout = (traj.direction() == alongMomentum);

  GlobalPoint vtxPos(vtx.x(),vtx.y(),vtx.z());
  
  uint8_t nhits = 0;

  std::vector<TrajectoryMeasurement>::const_iterator closest = measurements.begin();
  double distance = 1e6;
  //iterate inside out, when distance to vertex starts increasing, we are at the closest hit
  for (std::vector<TrajectoryMeasurement>::const_iterator fwdit = measurements.begin(), revit = measurements.end()-1;
          fwdit != measurements.end(); ++fwdit,--revit) {
    
    std::vector<TrajectoryMeasurement>::const_iterator it;
    if (inout) {
      it = fwdit;
    }
    else {
      it = revit;
    }

    //increment hit counter, compute distance and set iterator if this hit is valid
    if (it->recHit() && it->recHit()->isValid()) {
      ++nhits;
      distance = (vtxPos - it->updatedState().globalPosition()).mag();
      closest = it;
    }

    if ( (measurements.end()-fwdit)==1) {
      break; 
    }

    //check if next valid hit is farther away from vertex than existing closest
    std::vector<TrajectoryMeasurement>::const_iterator nextit;
    if (inout) {
      nextit = it+1;
    }
    else {
      nextit = it-1;
    }
    
    
    if ( nextit->recHit() && nextit->recHit()->isValid() ) {
      double nextDistance = (vtxPos - nextit->updatedState().globalPosition()).mag();
      if (nextDistance > distance) {
        break;
      }
    }
    
  }

  //compute signed decaylength significance for closest hit and check if it is before the vertex
  //if not then we need to subtract it from the count of hits before the vertex, since it has been implicitly included

  GlobalVector momDir = closest->updatedState().globalMomentum().unit();
  double decayLengthHitToVtx = (vtxPos - closest->updatedState().globalPosition()).dot(momDir);

  AlgebraicVector3 j;
  j[0] = momDir.x();
  j[1] = momDir.y();
  j[2] = momDir.z();
  AlgebraicVector6 jj;
  jj[0] = momDir.x();
  jj[1] = momDir.y();
  jj[2] = momDir.z();
  jj[3] =0.;
  jj[4] =0.;
  jj[5] =0.;
  
  //TODO: In principle the hit measurement position is correlated with the vertex fit
  //at worst though we inflate the uncertainty by a factor of two
  double trackError2 = ROOT::Math::Similarity(jj,closest->updatedState().cartesianError().matrix());
  double vertexError2 = ROOT::Math::Similarity(j,vtx.covariance());
  double decayLenError = sqrt(trackError2+vertexError2);

  Measurement1DFloat decayLength(decayLengthHitToVtx,decayLenError);

  if (decayLength.significance() < sigmaTolerance) {  //decay length is not (significantly) positive, so hit is consistent with the vertex position or late
                                           //subtract it from wrong hits count
    --nhits;
  }
 
  return std::pair<unsigned int,Measurement1DFloat>(nhits,decayLength);

}

uint8_t ConversionHitChecker::nSharedHits(const reco::Track &trk1, const reco::Track &trk2) const {
 
  uint8_t nShared = 0;

  for (trackingRecHit_iterator iHit1 = trk1.recHitsBegin();  iHit1 != trk1.recHitsEnd(); ++iHit1) { 
    const TrackingRecHit *hit1 = (*iHit1);
    if (hit1->isValid()) {
      for (trackingRecHit_iterator iHit2 = trk2.recHitsBegin();  iHit2 != trk2.recHitsEnd(); ++iHit2) { 
        const TrackingRecHit *hit2 = (*iHit2);
        if (hit2->isValid() && hit1->sharesInput(hit2,TrackingRecHit::some)) {
          ++nShared;
        }
      }    
    }
  }
  
  return nShared;

}


