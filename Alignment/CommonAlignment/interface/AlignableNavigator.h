#ifndef Alignment_CommonAlignment_AlignableNavigator_h
#define Alignment_CommonAlignment_AlignableNavigator_h

#include <map>
#include <string>

class Alignable;

/// A class to navigate from a DetId to an Alignable
/// A map is created at construction time from all
/// sub-structures of the constructor's argument.

class AlignableNavigator 
{

public:
  
  /// Constructor from Alignbable
  AlignableNavigator( Alignable* alignable );

  
  typedef std::map<DetId, Alignable*> MapType;
  typedef std::pair<DetId, Alignable*> PairType;

  /// Returns pointer to Alignable corresponding to given DetId
  Alignable* alignableFromDetId( const DetId& detid );

private:

  void recursiveGetId( Alignable* alignable );

  MapType theMap;


};

#endif
