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
 */

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CommonTopologies/interface/GeomDet.h"
#include <vector>
#include <unordered_map>

class TrackingGeometry {
public:
  using DetTypeContainer = std::vector<const GeomDetType*>;
  using DetContainer = std::vector<const GeomDet*>;
  using DetIdContainer = std::vector<DetId>;
  using mapIdToDetUnit = std::unordered_map<unsigned int, const GeomDet*>;
  using mapIdToDet = std::unordered_map<unsigned int, const GeomDet*>;

  /// Destructor.
  virtual ~TrackingGeometry() {}

  /// Return a vector of all det types.
  virtual const DetTypeContainer& detTypes() const = 0;

  /// Returm a vector of all GeomDet
  virtual const DetContainer& detUnits() const = 0;

  /// Returm a vector of all GeomDet (including all GeomDetUnits)
  virtual const DetContainer& dets() const = 0;

  /// Returm a vector of all GeomDetUnit DetIds
  virtual const DetIdContainer& detUnitIds() const = 0;

  /// Returm a vector of all GeomDet DetIds (including those of GeomDetUnits)
  virtual const DetIdContainer& detIds() const = 0;

  /// Return the pointer to the GeomDetUnit corresponding to a given DetId
  virtual const GeomDet* idToDetUnit(DetId) const = 0;

  /// Return the pointer to the GeomDet corresponding to a given DetId
  /// (valid also for GeomDetUnits)
  virtual const GeomDet* idToDet(DetId) const = 0;
};

#endif
