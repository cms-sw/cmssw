#ifndef ConversionHitChecker_H
#define ConversionHitChecker_H

/** \class ConversionHitChecker
 *
 *
 * \author J.Bendavid
 *
 * \version   
 * Check hits along a Trajectory and count how many are before
 * the vertex position. (taking into account the uncertainty in the vertex
 * and hit positions.  Also returns a the signed decay length
 * and uncertainty from the closest hit on the track to the vertex position.
 *
 ************************************************************/

//
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackExtra.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/Measurement1DFloat.h"
//
//
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include <utility>

class Trajectory;
class ConversionHitChecker {

public:

  ConversionHitChecker() {}
  ~ConversionHitChecker() {}


  std::pair<uint8_t,Measurement1DFloat> nHitsBeforeVtx(const Trajectory &traj, const reco::Vertex &vtx,
                   double sigmaTolerance = 3.0) const;
                   
  uint8_t nSharedHits(const reco::Track &trk1, const reco::Track &trk2) const;

 


};

#endif // ConversionHitChecker_H


