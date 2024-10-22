#ifndef GEMGeometry_GEMGeometry_h
#define GEMGeometry_GEMGeometry_h

/** \class GEMGeometry
 *
 *  The model of the geometry of GEM.
 *
 *  \author M. Maggi - INFN Bari
 */

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/GEMGeometry/interface/GEMEtaPartition.h"
#include "Geometry/GEMGeometry/interface/GEMChamber.h"
#include "Geometry/GEMGeometry/interface/GEMSuperChamber.h"
#include "Geometry/GEMGeometry/interface/GEMRing.h"
#include "Geometry/GEMGeometry/interface/GEMStation.h"
#include "Geometry/GEMGeometry/interface/GEMRegion.h"
#include <vector>
#include <map>

class GeomDetType;

class GEMGeometry : public TrackingGeometry {
public:
  /// Default constructor
  GEMGeometry();

  /// Destructor
  ~GEMGeometry() override;

  friend class GEMGeometryBuilder;
  friend class GeometryAligner;

  // Return a vector of all det types
  const DetTypeContainer& detTypes() const override;

  // Return a vector of all GeomDet
  const DetContainer& detUnits() const override;

  // Return a vector of all GeomDet
  const DetContainer& dets() const override;

  // Return a vector of all GeomDet DetIds
  const DetIdContainer& detUnitIds() const override;

  // Return a vector of all GeomDet DetIds
  const DetIdContainer& detIds() const override;

  // Return the pointer to the GeomDet corresponding to a given DetId
  const GeomDet* idToDetUnit(DetId) const override;

  // Return the pointer to the GeomDet corresponding to a given DetId
  const GeomDet* idToDet(DetId) const override;

  //---- Extension of the interface

  /// Return a vector of all GEM regions
  const std::vector<const GEMRegion*>& regions() const;

  /// Return a vector of all GEM stations
  const std::vector<const GEMStation*>& stations() const;

  /// Return a vector of all GEM rings
  const std::vector<const GEMRing*>& rings() const;

  /// Return a vector of all GEM super chambers
  const std::vector<const GEMSuperChamber*>& superChambers() const;

  /// Return a vector of all GEM chambers
  const std::vector<const GEMChamber*>& chambers() const;

  /// Return a vector of all GEM eta partitions
  const std::vector<const GEMEtaPartition*>& etaPartitions() const;

  // Return a GEMRegion
  const GEMRegion* region(int region) const;

  // Return a GEMStation
  const GEMStation* station(int region, int station) const;

  /// Return a GEMRing
  const GEMRing* ring(int region, int station, int ring) const;

  // Return a GEMSuperChamber given its id
  const GEMSuperChamber* superChamber(GEMDetId id) const;

  // Return a GEMChamber given its id
  const GEMChamber* chamber(GEMDetId id) const;

  /// Return a GEMEtaPartition given its id
  const GEMEtaPartition* etaPartition(GEMDetId id) const;

  /// Add a GEMRegion to the Geometry
  void add(const GEMRegion* region);

  /// Add a GEMStation to the Geometry
  void add(const GEMStation* station);

  /// Add a GEMRing to the Geometry
  void add(const GEMRing* ring);

  /// Add a GEMSuperChamber to the Geometry
  void add(const GEMSuperChamber* sch);

  /// Add a GEMChamber to the Geometry
  void add(const GEMChamber* ch);

  /// Add a GEMEtaPartition  to the Geometry
  void add(const GEMEtaPartition* etaPartition);

  bool hasME0() const;
  bool hasGE11() const;
  bool hasGE21() const;

private:
  DetContainer theEtaPartitions;
  DetContainer theDets;
  DetTypeContainer theEtaPartitionTypes;
  DetIdContainer theEtaPartitionIds;
  DetIdContainer theDetIds;

  // Map for efficient lookup by DetId
  mapIdToDet theMap;

  std::vector<const GEMEtaPartition*> allEtaPartitions;  // Are not owned by this class; are owned by their chamber.
  std::vector<const GEMChamber*> allChambers;  // Are not owned by this class; are owned by their superchamber.
  std::vector<const GEMSuperChamber*> allSuperChambers;  // Are owned by this class.
  std::vector<const GEMRing*> allRings;                  // Are owned by this class.
  std::vector<const GEMStation*> allStations;            // Are owned by this class.
  std::vector<const GEMRegion*> allRegions;              // Are owned by this class.
};

#endif
