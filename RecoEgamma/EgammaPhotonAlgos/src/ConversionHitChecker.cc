#include <iostream>
#include <vector>
#include <memory>
#include "RecoEgamma/EgammaPhotonAlgos/interface/ConversionHitChecker.h"
// Framework
//


std::pair<uint8_t,Measurement1DFloat> ConversionHitChecker::nHitsBeforeVtx(const reco::TrackExtra & track, const reco::Vertex &vtx, float sigmaTolerance) const {

  // track hits are always inout

  GlobalPoint vtxPos(vtx.x(),vtx.y(),vtx.z());
  
  uint8_t nhits = 0;

  float distance = 1e6;

  auto const & trajParams = track.trajParams();
  // here till debug end
  if (trajParams.size()!=track.recHitsSize()) 
    std::cout << "traj param and hits diff size " << trajParams.size()
         << ' ' << track.recHitsSize() <<std::endl;
  assert(trajParams.size()==track.recHitsSize());
  auto hb = track.recHitsBegin();
  unsigned int closest=0;
  for(unsigned int h=0;h<track.recHitsSize()-1;h++){
    auto recHit = *(hb+h);
    if(!recHit->isValid()) continue;

    auto globalPosition = recHit->surface()->toGlobal(trajParams[h].position());         

    ++nhits;
    distance = (vtxPos - globalPosition).mag();
    closest = h;
    

    //check if next valid hit is farther away from vertex than existing closest
    auto nextHit = *(hb+h);
    
    
    if ( nextHit->isValid() ) {
      auto globalPosition = nextHit->surface()->toGlobal(trajParams[h+1].position());
      auto nextDistance = (vtxPos - globalPosition).mag();
      if (nextDistance > distance) break;
    }
    
  }

  //compute signed decaylength significance for closest hit and check if it is before the vertex
  //if not then we need to subtract it from the count of hits before the vertex, since it has been implicitly included
  auto recHit = *(hb+closest);
  auto momDir = recHit->surface()->toGlobal(trajParams[closest].direction());
  auto globalPosition = recHit->surface()->toGlobal(trajParams[closest].position()); 
  float decayLengthHitToVtx = (vtxPos - globalPosition).dot(momDir);

  AlgebraicVector3 j;
  j[0] = momDir.x();
  j[1] = momDir.y();
  j[2] = momDir.z();
  float vertexError2 = ROOT::Math::Similarity(j,vtx.covariance());
  auto decayLenError = std::sqrt(vertexError2);

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


