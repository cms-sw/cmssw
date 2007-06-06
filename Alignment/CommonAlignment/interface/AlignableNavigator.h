#ifndef Alignment_CommonAlignment_AlignableNavigator_h
#define Alignment_CommonAlignment_AlignableNavigator_h

#include <map>
#include <string>

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

class Alignable;
class AlignableDet;

/// A class to navigate from a DetId to an Alignable
/// A map is created at construction time from all
/// sub-structures of the constructor's argument.

class AlignableNavigator 
{

public:
  
  /// Constructor from Alignbable
  AlignableNavigator( Alignable* alignable );

  /// Constructor from two Alignables
  AlignableNavigator( Alignable* tracker, Alignable* muon );

  /// Constructor from list of Alignbable
  AlignableNavigator( std::vector<Alignable*> alignable );

  
  typedef std::map<DetId, Alignable*> MapType;
  typedef std::pair<DetId, Alignable*> PairType;

  /// Returns pointer to Alignable corresponding to given DetId
  Alignable* alignableFromDetId( const DetId& detid );

  /// Returns pointer to Alignable corresponding to given GeomDet
  Alignable* alignableFromGeomDet( const GeomDet* geomDet );

  /// Returns pointer to AlignableDet corresponding to given GeomDet
  AlignableDet* alignableDetFromGeomDet( const GeomDet* geomDet );

  /// Returns pointer to AlignableDet corresponding to given DetId
  AlignableDet* alignableDetFromDetId( const DetId& detid );

  /// Returns vector of AlignableDet* for given vector of Hits
  std::vector<AlignableDet*> alignableDetsFromHits(const std::vector<const TransientTrackingRecHit*>& hitvec);
  /// Returns vector of AlignableDet* for given vector of Hits
  std::vector<AlignableDet*> alignableDetsFromHits
    (const TransientTrackingRecHit::ConstRecHitContainer &hitVec);

  /// Returns number of elements in map
  int size( void ) { return theMap.size(); }

private:

  void recursiveGetId( Alignable* alignable );

  MapType theMap;


};

#endif
