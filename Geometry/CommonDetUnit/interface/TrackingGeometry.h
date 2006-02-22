#ifndef TrackingGeometry_h
#define TrackingGeometry_h

/** \class TrackingGeometry
 *
 *  Base class for the geometry of tracking detectors.
 *
 *  $Date: 2005/11/04 18:09:44 $
 *  $Revision: 1.1 $
 *  \author ...
 */

#include "DataFormats/DetId/interface/DetId.h"
#include <vector>
#include <map>

class GeomDetType;
class GeomDetUnit;
class GeomDet;

class TrackingGeometry
{
public:
  typedef std::vector<GeomDetType*>          DetTypeContainer;
  typedef std::vector<GeomDet*>              DetContainer;
  typedef std::vector<GeomDetUnit*>          DetUnitContainer;
  typedef std::vector<DetId>                 DetIdContainer;
  typedef std::map<DetId,GeomDetUnit*>       mapIdToDetUnit;
  typedef std::map<DetId,GeomDet*>           mapIdToDet;

  /// Default constructor
  //  virtual TrackingGeometry() {}
  
  /// Destructor
  virtual ~TrackingGeometry() {}
  
  /// Return a vector of all det types
  virtual const DetTypeContainer&  detTypes()     const = 0;

  /// Returm a vector of all GeomDetUnit
  virtual const DetUnitContainer&      detUnits()         const = 0;

  /// Returm a vector of all GeomDet
  virtual const DetContainer&      dets()         const = 0;

  /// Returm a vector of all GeomDetUnit DetIds
  virtual const DetIdContainer&    detUnitIds()       const = 0;

  /// Returm a vector of all GeomDet DetIds
  virtual const DetIdContainer&    detIds()       const = 0;

  /// Return the pointer to the GeomDetUnit corresponding to a given DetId
  virtual const GeomDetUnit*       idToDetUnit(DetId) const = 0;

  /// Return the pointer to the GeomDet corresponding to a given DetId
  virtual const GeomDet*       idToDet(DetId) const = 0; 

};

#endif
