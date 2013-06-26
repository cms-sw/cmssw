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
///  $Revision: 1.12 $
///  $Date: 2013/01/07 19:44:30 $
///  (last update by $Author: wmtan $)

#include <utility>

class DetId;
class TrackerTopology;

class TrackerAlignableId
{

public:
  
  TrackerAlignableId() {}

  /// Return type and layer of DetId
  /// Keep this for now.
  /// Concept of a "layer" in Alignment is obsolete.
  /// Will be replaced by a more generic function.
  std::pair<int,int> typeAndLayerFromDetId( const DetId& detId, const TrackerTopology* tTopo ) const;

};

#endif
