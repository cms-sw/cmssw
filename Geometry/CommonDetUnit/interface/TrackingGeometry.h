#ifndef TrackingGeometry_h
#define TrackingGeometry_h

/** \class TrackingGeometry
 *
 *  Base class for the geometry of tracking detectors.
 *
 *  $Date: $
 *  $Revision: $
 *  \author ...
 */

#include "DataFormats/DetId/interface/DetId.h"
#include <vector>
#include <map>

class GeomDetType;
class GeomDetUnit;

class TrackingGeometry
{
public:
  typedef std::vector<GeomDetType*>          DetTypeContainer;
  typedef std::vector<GeomDetUnit*>          DetContainer;
  typedef std::vector<DetId>                 DetIdContainer;
  typedef std::map<DetId,GeomDetUnit*>       mapIdToDet;

  /// Default constructor
  //  virtual TrackingGeometry() {}

  /// Destructor
  virtual ~TrackingGeometry() {}

  /// Return a vector of all det types
  virtual const DetTypeContainer&  detTypes()     const = 0;

  /// Returm a vector of all GeomDetUnit
  virtual const DetContainer&      dets()         const = 0;

  /// Returm a vector of all DetIds
  virtual const DetIdContainer&    detIds()       const = 0;

  /// Return the pointer to the GeomDetUnit corresponding to a given DetId
  virtual const GeomDetUnit*       idToDet(DetId) const = 0;

};

#endif
