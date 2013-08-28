//   $Revision: 1.15 $
//   $Date: 2009/02/28 21:06:53 $
//   (last update by $Author: flucke $)

#ifndef Alignment_CommonAlignment_AlignableNavigator_h
#define Alignment_CommonAlignment_AlignableNavigator_h

#include <map>
#include <vector>

#include "TrackingTools/TransientTrackingRecHit/interface/TransientTrackingRecHit.h"

#include "Alignment/CommonAlignment/interface/AlignableDetOrUnitPtr.h"

class Alignable;
class AlignableDet;
class AlignableExtras;
class GeomDet;


/// A class to navigate from a DetId to the corresponding AlignableDetOrUnitPtr.
/// A map is created at construction time from all
/// sub-structures of the constructor's argument(s) that are AlignableDet or AlignableDetUnit.

class AlignableNavigator 
{

public:
  
  /// Constructor from one or two Alignables
  explicit AlignableNavigator(Alignable* tracker, Alignable* muon = 0);
  
  /// Constructor from one or two Alignables
  explicit AlignableNavigator(AlignableExtras* extras, Alignable* tracker, Alignable* muon = 0);

  /// Constructor from list of Alignbable
  explicit AlignableNavigator( const std::vector<Alignable*>& alignables );

  typedef std::map<DetId, AlignableDetOrUnitPtr> MapType;
  typedef MapType::value_type PairType;

  /// Returns AlignableDetOrUnitPtr corresponding to given DetId
  AlignableDetOrUnitPtr alignableFromDetId( const DetId& detid );

  /// Returns AlignableDetOrUnitPtr corresponding to given GeomDet
  AlignableDetOrUnitPtr alignableFromGeomDet( const GeomDet* geomDet );

  /// Returns vector AlignableDetOrUnitPtr for given vector of Hits.
  std::vector<AlignableDetOrUnitPtr> 
    alignablesFromHits(const std::vector<const TransientTrackingRecHit*>& hitvec);

  /// Returns vector of AlignableDetOrUnitPtr for given vector of Hits.
  std::vector<AlignableDetOrUnitPtr> alignablesFromHits
    (const TransientTrackingRecHit::ConstRecHitContainer &hitVec);

  /// return all AlignableDetOrUnitPtrs
  std::vector<AlignableDetOrUnitPtr> alignableDetOrUnits();

  /// Returns number of elements in map
  int size( void ) { return theMap.size(); }

  /// Given a DetId, returns true if DetIds with this detector and subdetector id are in the map (not necessarily the exact DetId)
  bool detAndSubdetInMap( const DetId& detid ) const;

private:
  /// Add recursively DetId-AlignableDetOrUnitPtr pairs to map.
  /// Returns number of Alignables with DetId!=0 which are (illegaly) neither AlignableDet
  /// nor AlignableDetUnit and are thus not added to the map.
  unsigned int recursiveGetId(Alignable* alignable);

  MapType                          theMap;
  std::vector<std::pair<int,int> > theDetAndSubdet;
};

#endif
