#ifndef Alignment_CommonAlignment_TrackerAlignableId_H
#define Alignment_CommonAlignment_TrackerAlignableId_H

#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"

/// Helper class to provide unique numerical ID's for Alignables. 
/// The unique ID is formed from:
///  - the AlignableObjectId (DetUnit, Det, Rod, Layer, etc.)
///  - the geographical ID of the first GeomDet in the composite.

class GeomDet;
class Alignable;

class TrackerAlignableId
{

public:

  /// public access to the unique instance 
  static TrackerAlignableId* instance( );

  /// Destructor
  ~TrackerAlignableId() {};

  /// Return geographical ID of first GeomDet
  unsigned int alignableId( Alignable* alignable );

  /// Return Type ID (Det, Rod etc.) of Alignable
  int alignableTypeId( Alignable* alignable ); 

  /// Return type and layer of Alignable
  std::pair<int,int> typeAndLayerFromAlignable( Alignable* alignable );

  /// Return type and layer of GeomDet
  std::pair<int,int> typeAndLayerFromGeomDet( const GeomDet& geomDet );

private:

  /// Constructor
  TrackerAlignableId() {};

  /// Get first AlignableDet of an Alignable
  AlignableDet* firstDet( Alignable* alignable );
 
  /// Get unique identifyer of first AlignableDet of alignable
  unsigned int firstDetId( Alignable* alignable );

};

#endif
