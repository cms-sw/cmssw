#ifndef Fireworks_Tracks_TrackUtils_h
#define Fireworks_Tracks_TrackUtils_h
// -*- C++ -*-
//
// Package:     Tracks
// Class  :     TrackUtils
// $Id: TrackUtils.h,v 1.17 2010/06/10 17:16:03 amraktad Exp $
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
class TGeoHMatrix;
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
 
void pixelLocalXY( const double mpx, const double mpy, const DetId& id,
                   double& lpx, double& lpy );

double pixelLocalX( const double mpx, const int m_nrows );
double pixelLocalY( const double mpy, const int m_ncols );
void localSiPixel( TVector3& point, double lx, double ly, DetId id, const FWEventItem* iItem ); 
void localSiStrip( TVector3& point, TVector3& pointA, TVector3& pointB, double bc, DetId id, const FWEventItem* iItem );
// monoPoints include pixels (why?)
void pushTrackerHits( std::vector<TVector3> &monoPoints, std::vector<TVector3> &stereoPoints, const FWEventItem &iItem, const reco::Track &t );
void pushPixelHits( std::vector<TVector3> &pixelPoints, const FWEventItem &iItem, const reco::Track &t );   
void pushNearbyPixelHits( std::vector<TVector3> &pixelPoints, const FWEventItem &iItem, const reco::Track &t );   
void pushPixelCluster( std::vector<TVector3> &pixelPoints, const TGeoHMatrix *m, DetId id, const SiPixelCluster &c ); 


void pushSiStripHits( std::vector<TVector3> &monoPoints, std::vector<TVector3> &stereoPoints, const FWEventItem &iItem, const reco::Track &t );
void addSiStripClusters( const FWEventItem* iItem, const reco::Track &t, class TEveElement *tList, bool addNearbyClusters, bool master);


// DETAIL VIEWS
void addTrackerHits3D( std::vector<TVector3> &points, class TEveElementList *tList,
                       Color_t color, int size );
void
addHits(const reco::Track& track,
        const FWEventItem* iItem,
        TEveElement* trkList,
        bool addNearbyHits);
void
addModules(const reco::Track& track,
           const FWEventItem* iItem,
           TEveElement* trkList,
           bool addLostHits);


// Helpers for data extraction
const SiStripCluster* extractClusterFromTrackingRecHit(const TrackingRecHit* rh);


// Helper functions to get human readable informationa about given DetId
// (copied from TrackingTools/TrackAssociator)
std::string info( const DetId& );
std::string info( const std::set<DetId>& );
std::string info( const std::vector<DetId>& );
}

#endif // Fireworks_Tracks_TrackUtils_h
