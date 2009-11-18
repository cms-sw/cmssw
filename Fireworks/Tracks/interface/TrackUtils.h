#ifndef Fireworks_Tracks_TrackUtils_h
#define Fireworks_Tracks_TrackUtils_h
// -*- C++ -*-
//
// Package:     Core
// Class  :     TrackUtils
// $Id: TrackUtils.h,v 1.1 2009/10/04 12:13:07 dmytro Exp $
//

// system include files
#include "Rtypes.h"

// user include files
#include "TEveVSDStructs.h"

// forward declarations
namespace reco {
   class Track;
   class Candidate;
}
class TEveElement;
class TEveTrackPropagator;
class FWModelId;
class DetId;

namespace fireworks {
   struct State {
      TEveVector position;
      TEveVector momentum;
      bool valid;
      State() : valid(false){
      }
      State(const TEveVector& pos) :
         position(pos),valid(false){
      }
      State(const TEveVector& pos, const TEveVector& mom) :
         position(pos),momentum(mom),valid(true){
      }
   };

   class StateOrdering {
      TEveVector m_direction;
public:
      StateOrdering( const TEveVector& momentum ){
         m_direction = momentum;
         m_direction.Normalize();
      }
      bool operator() (const State& state1,
                       const State& state2 ) const
      {
         double product1 = state1.position.Perp()*(state1.position.fX*m_direction.fX + state1.position.fY*m_direction.fY>0 ? 1 : -1);
         double product2 = state2.position.Perp()*(state2.position.fX*m_direction.fX + state2.position.fY*m_direction.fY>0 ? 1 : -1);
         return product1 < product2;
      }
   };

   TEveTrack* prepareTrack(const reco::Track& track,
                           TEveTrackPropagator* propagator,
                           Color_t color,
                           const std::vector<TEveVector>& extraRefPoints = std::vector<TEveVector>());
   TEveTrack* prepareTrack(const reco::Candidate& candidate,
                           TEveTrackPropagator* propagator,
                           Color_t color);

   void pixelLocalXY(const double mpx, const double mpy, const DetId& id,
                     double& lpx, double& lpy);

   double pixelLocalX(const double mpx, const int m_nrows);
   double pixelLocalY(const double mpy, const int m_ncols);
   // monoPoints include pixels (why?)
   void pushTrackerHits(std::vector<TVector3> &monoPoints, std::vector<TVector3> &stereoPoints,
                        const FWModelId &id, const reco::Track &t);
   void addTrackerHitsEtaPhi(std::vector<TVector3> &points, class TEveElementList *tList,
                             Color_t color, int size);
   void addTrackerHits3D(std::vector<TVector3> &points, class TEveElementList *tList,
                         Color_t color, int size);



}

#endif
