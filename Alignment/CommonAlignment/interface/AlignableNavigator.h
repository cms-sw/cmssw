#ifndef Alignment_CommonAlignment_AlignableNavigator_h
#define Alignment_CommonAlignment_AlignableNavigator_h

#include <map>
#include <string>

#include "Geometry/CommonDetUnit/interface/GeomDet.h"

class Alignable;

/// A class to navigate from a DetId to an Alignable
/// A map is created at construction time from all
/// sub-structures of the constructor's argument.

class AlignableNavigator 
{

public:
  
  /// Constructor from Alignbable
  AlignableNavigator( Alignable* alignable );

  /// Constructor from list of Alignbable
  AlignableNavigator( std::vector<Alignable*> alignable );

  
  typedef std::map<DetId, Alignable*> MapType;
  typedef std::pair<DetId, Alignable*> PairType;

  /// Returns pointer to Alignable corresponding to given DetId
  Alignable* alignableFromDetId( const DetId& detid );

  /// Returns pointer to Alignable corresponding to given GeomDet
  Alignable* alignableFromGeomDet( const GeomDet* geomDet );

  /// Returns number of elements in map
  int size( void ) { return theMap.size(); }

private:

  void recursiveGetId( Alignable* alignable );

  MapType theMap;


};

#endif
