#ifndef Alignment_CommonAlignment_TrackerAlignableId_H
#define Alignment_CommonAlignment_TrackerAlignableId_H

/// \class TrackerAlignableId
///
/// Helper class to provide unique numerical ID's for Alignables. 
/// The unique ID is formed from:
///  - the AlignableObjectId (DetUnit, Det, Rod, Layer, etc.)
///  - the geographical ID of the first GeomDet in the composite.
/// A mapping between the AlignableObjectId and the string name
/// is also provided.
///
///  $Revision: 1.9 $
///  $Date: 2007/05/12 00:27:42 $
///  (last update by $Author: cklae $)

#include <string>
#include <utility>
#include <boost/cstdint.hpp>

class Alignable;
class DetId;
class GeomDet;

class TrackerAlignableId
{

public:
  
  TrackerAlignableId() {}
  ~TrackerAlignableId() {}

  typedef std::pair<uint32_t,int> UniqueId;

  /// Return geographical ID of first GeomDet
  uint32_t alignableId( const Alignable* alignable ) const;

  /// Return Type ID (Det, Rod etc.) of Alignable
  int alignableTypeId( const Alignable* alignable ) const; 

  /// Return uniqueID of alignable, consisting of the geographical ID of the
  /// first GeomDet and the type ID (i.e. Rod, Layer, etc.) 
  UniqueId alignableUniqueId( const Alignable* alignable ) const;
  /// Return type and layer of Alignable
  std::pair<int,int> typeAndLayerFromAlignable( const Alignable* alignable ) const;

  /// Return type and layer of GeomDet
  std::pair<int,int> typeAndLayerFromGeomDet( const GeomDet& geomDet ) const;

  /// Return type and layer of DetId
  std::pair<int,int> typeAndLayerFromDetId( const DetId& detId ) const;

  /// Return string corresponding to given Alignable
  const std::string& alignableTypeName( const Alignable* alignable ) const;

  /// Return string corresponding to given alignable object ID
  const std::string& alignableTypeIdToName( int id ) const;
  

private:

  /// Get first sensor of an Alignable
  const Alignable& firstDet( const Alignable& alignable ) const;
 
  /// Get unique identifyer of first AlignableDet of alignable
  uint32_t firstDetId( const Alignable& alignable ) const;

};

#endif
