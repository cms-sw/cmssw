#ifndef GlobalTrackingGeometry_h
#define GlobalTrackingGeometry_h

/** \class GlobalTrackingGeometry
 *
 *  Single entry point to the tracker and muon geometries.
 *  The main purpose is to provide the methods idToDetUnit(DetId) and idToDet(DetId)
 *  that allow to get an element of the geometry given its DetId, regardless of wich subdetector it belongs.
 * 
 *  The slave geometries (TrackerGeometry, MTDGeometry, DTGeometry, CSCGeometry, RPCGeometry, GEMGeometry, ME0Geometry) 
 *  are accessible with the method slaveGeometry(DetId).
 *
 *  \author M. Sani
 */

#include "Geometry/CommonTopologies/interface/TrackingGeometry.h"
#include <vector>
#include <atomic>

class GlobalTrackingGeometry : public TrackingGeometry {
public:
  GlobalTrackingGeometry(std::vector<const TrackingGeometry*>& geos);

  ~GlobalTrackingGeometry() override;

  // Return a vector of all det types.
  const DetTypeContainer& detTypes() const override;

  // Returm a vector of all GeomDetUnit
  const DetContainer& detUnits() const override;

  // Returm a vector of all GeomDet (including all GeomDetUnits)
  const DetContainer& dets() const override;

  // Returm a vector of all GeomDetUnit DetIds
  const DetIdContainer& detUnitIds() const override;

  // Returm a vector of all GeomDet DetIds (including those of GeomDetUnits)
  const DetIdContainer& detIds() const override;

  // Return the pointer to the GeomDetUnit corresponding to a given DetId
  const GeomDet* idToDetUnit(DetId) const override;

  // Return the pointer to the GeomDet corresponding to a given DetId
  // (valid also for GeomDetUnits)
  const GeomDet* idToDet(DetId) const override;

  /// Return the pointer to the actual geometry for a given DetId
  const TrackingGeometry* slaveGeometry(DetId id) const;

private:
  std::vector<const TrackingGeometry*> theGeometries;

  // The const methods claim to simply return these vectors,
  // but actually, they'll fill them up the first time they
  // are called, which is rare (or never).
  mutable std::atomic<DetTypeContainer*> theDetTypes;
  mutable std::atomic<DetContainer*> theDetUnits;
  mutable std::atomic<DetContainer*> theDets;
  mutable std::atomic<DetIdContainer*> theDetUnitIds;
  mutable std::atomic<DetIdContainer*> theDetIds;
};

#endif
