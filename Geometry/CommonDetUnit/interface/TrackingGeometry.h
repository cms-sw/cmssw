#ifndef TrackingGeometry_h
#define TrackingGeometry_h

/** \class TrackingGeometry
 *
 *  Base class for the geometry of tracking detectors.
 *  A geometry contains both the GeomDetUnit s and bigger composite 
 *  structures, which are concrete GeomDet s.
 *
 *  There are therefore 2 kind of methods in the interface: 
 *   - the generic ones, i.e.
 *  dets(), detIds(), idToDet(), do not distinguish between 
 *  GeomDet and GeomDetUnit, and can be used blindly for the typical use
 *  of accessing the reference frame transformation of the det. 
 *   - Those specific to GeomDetUnit s, i.e. detUnits(),  detUnitIds(), 
 *  idToDetUnit(), are useful when it is necessary to deal with the 
 *  extended interface of GeomDetUnit. 
 *
 *  $Date: 2007/06/05 08:38:46 $
 *  $Revision: 1.4 $
 */

#include "DataFormats/DetId/interface/DetId.h"
#include <vector>
// #include <map>
#include <ext/hash_map>

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
  //  typedef std::map<DetId,GeomDetUnit*>       mapIdToDetUnit;
  // typedef std::map<DetId,GeomDet*>           mapIdToDet;
  typedef  __gnu_cxx::hash_map< unsigned int, GeomDetUnit*> mapIdToDetUnit;
  typedef  __gnu_cxx::hash_map< unsigned int, GeomDet*>     mapIdToDet;

  // Default constructor
  //  virtual TrackingGeometry() {}
  
  /// Destructor.
  virtual ~TrackingGeometry() {}
  
  /// Return a vector of all det types.
  virtual const DetTypeContainer&  detTypes()         const = 0;

  /// Returm a vector of all GeomDetUnit
  virtual const DetUnitContainer&  detUnits()         const = 0;

  /// Returm a vector of all GeomDet (including all GeomDetUnits)
  virtual const DetContainer&      dets()             const = 0;

  /// Returm a vector of all GeomDetUnit DetIds
  virtual const DetIdContainer&    detUnitIds()       const = 0;

  /// Returm a vector of all GeomDet DetIds (including those of GeomDetUnits)
  virtual const DetIdContainer&    detIds()           const = 0;

  /// Return the pointer to the GeomDetUnit corresponding to a given DetId
  virtual const GeomDetUnit*       idToDetUnit(DetId) const = 0;

  /// Return the pointer to the GeomDet corresponding to a given DetId
  /// (valid also for GeomDetUnits)
  virtual const GeomDet*           idToDet(DetId)     const = 0; 

};

#endif
