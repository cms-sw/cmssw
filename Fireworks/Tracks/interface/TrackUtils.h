#ifndef Fireworks_Tracks_TrackUtils_h
#define Fireworks_Tracks_TrackUtils_h
// -*- C++ -*-
//
// Package:     Tracks
// Class  :     TrackUtils
// $Id: TrackUtils.h,v 1.25 2010/09/07 15:46:48 yana Exp $
//

// system include files
#include "TEveVSDStructs.h"

// forward declarations
namespace reco 
{
   class Track;
}
class RecSegment;

class FWEventItem;
class TEveElement;
class TEveTrack;
class TEveTrackPropagator;
class DetId;
class FWGeometry;
class TEveStraightLineSet;

class SiPixelCluster;
class SiStripCluster;
class TrackingRecHit;

namespace fireworks {
  
struct State {
   TEveVector position;
   TEveVector momentum;
   bool valid;
   State() : valid(false) {
   }
   State(const TEveVector& pos) :
      position(pos), valid(false) {
   }
   State(const TEveVector& pos, const TEveVector& mom) :
      position(pos), momentum(mom), valid(true) {
   }
};

class StateOrdering {
   TEveVector m_direction;
public:
   StateOrdering( const TEveVector& momentum ) {
      m_direction = momentum;
      m_direction.Normalize();
   }
   bool operator() ( const State& state1,
                     const State& state2 ) const {
      double product1 = state1.position.Perp()*(state1.position.fX*m_direction.fX + state1.position.fY*m_direction.fY>0 ? 1 : -1);
      double product2 = state2.position.Perp()*(state2.position.fX*m_direction.fX + state2.position.fY*m_direction.fY>0 ? 1 : -1);
      return product1 < product2;
   }
};

TEveTrack* prepareTrack( const reco::Track& track,
                         TEveTrackPropagator* propagator,
                         const std::vector<TEveVector>& extraRefPoints = std::vector<TEveVector>() );
 
float pixelLocalX( const double mpx, const int m_nrows );
float pixelLocalY( const double mpy, const int m_ncols );

void localSiStrip( short strip, float* localTop, float* localBottom, const float* pars, unsigned int id );

void pushPixelHits( std::vector<TVector3> &pixelPoints, const FWEventItem &iItem, const reco::Track &t );   
void pushNearbyPixelHits( std::vector<TVector3> &pixelPoints, const FWEventItem &iItem, const reco::Track &t );   
void pushPixelCluster( std::vector<TVector3> &pixelPoints, const FWGeometry &geom, DetId id, const SiPixelCluster &c, const float* pars ); 

void addSiStripClusters( const FWEventItem* iItem, const reco::Track &t, class TEveElement *tList, bool addNearbyClusters, bool master );

// Helpers for data extraction
const SiStripCluster* extractClusterFromTrackingRecHit( const TrackingRecHit* rh );

// Helper functions to get human readable informationa about given DetId
// (copied from TrackingTools/TrackAssociator)
std::string info( const DetId& );
std::string info( const std::set<DetId>& );
std::string info( const std::vector<DetId>& );
}

#endif // Fireworks_Tracks_TrackUtils_h
