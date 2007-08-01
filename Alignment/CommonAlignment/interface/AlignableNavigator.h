#ifndef Alignment_CommonAlignment_AlignableNavigator_h
#define Alignment_CommonAlignment_AlignableNavigator_h

#include <map>
#include <vector>

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"

class Alignable;
class AlignableDet;
class GeomDet;


/// A class to navigate from a DetId to the corresponding AlignableDetOrUnitPtr.
/// A map is created at construction time from all
/// sub-structures of the constructor's argument(s).

class AlignableNavigator 
{

public:
  
  /// Constructor from Alignable
  explicit AlignableNavigator( Alignable* alignable );

  /// Constructor from two Alignables
  AlignableNavigator( Alignable* tracker, Alignable* muon );

  /// Constructor from list of Alignbable
  explicit AlignableNavigator( std::vector<Alignable*> alignables );

  typedef std::map<DetId, AlignableDetOrUnitPtr> MapType;
  typedef std::pair<DetId, AlignableDetOrUnitPtr> PairType;

  /// Returns AlignableDetOrUnitPtr corresponding to given DetId
  AlignableDetOrUnitPtr alignableFromDetId( const DetId& detid );

  /// Returns AlignableDetOrUnitPtr corresponding to given GeomDet
  AlignableDetOrUnitPtr alignableFromGeomDet( const GeomDet* geomDet );

  /// Deprecated method for backward compatibility:
  /// If geomDet is a GeomDetUnit, giving an error and returning the mother 
  /// of the corresponding AlignableDetUnit which is an AlignableDet.
  /// This could lead to inconsistencies. Use alignableFromGeomDet instead.
  AlignableDet* alignableDetFromGeomDet( const GeomDet* geomDet );

  /// Deprecated method for backward compatibility:
  /// If DetId belongs to a GeomDetUnit/AlignableDetUnit, giving an error
  /// and returning the mother of the AlignableDetUnit which is an
  /// AlignableDet.
  /// This could lead to inconsistencies. Use alignableFromGeomDet instead.
  AlignableDet* alignableDetFromDetId( const DetId& detid );

  /// Returns vector AlignableDetOrUnitPtr for given vector of Hits.
  std::vector<AlignableDetOrUnitPtr> 
    alignablesFromHits(const std::vector<const TransientTrackingRecHit*>& hitvec);

  /// Returns vector of AlignableDetOrUnitPtr for given vector of Hits.
  std::vector<AlignableDetOrUnitPtr> alignablesFromHits
    (const TransientTrackingRecHit::ConstRecHitContainer &hitVec);
  /// For backward compatibility, use alignablesFromHits (cf. alignableDetFromDetId).
  std::vector<AlignableDet*> alignableDetsFromHits
    (const TransientTrackingRecHit::ConstRecHitContainer &hitVec);
  /// For backward compatibility, use alignablesFromHits (cf. alignableDetFromDetId).
  std::vector<AlignableDet*> 
    alignableDetsFromHits(const std::vector<const TransientTrackingRecHit*>& hitvec);

  /// Returns number of elements in map
  int size( void ) { return theMap.size(); }

  /// Given a DetId, returns true if DetIds with this detector and subdetector id are in the map (not necessarily the exact DetId)
  bool detAndSubdetInMap( const DetId& detid ) const;

private:

  void recursiveGetId( Alignable* alignable );

  MapType                          theMap;
  std::vector<std::pair<int,int> > theDetAndSubdet;
};

#endif
