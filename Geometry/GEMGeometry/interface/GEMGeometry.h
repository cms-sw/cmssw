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
//#include "Geometry/GEMGeometry/interface/GEMChamber.h"
#include <vector>
#include <map>

class GeomDetType;


class GEMGeometry : public TrackingGeometry {

 public:
  /// Default constructor
  GEMGeometry();

  /// Destructor
  virtual ~GEMGeometry();

  // Return a vector of all det types
  virtual const DetTypeContainer&  detTypes() const override;

  // Return a vector of all GeomDetUnit
  virtual const DetUnitContainer& detUnits() const override;

  // Return a vector of all GeomDet
  virtual const DetContainer& dets() const override;
  
  // Return a vector of all GeomDetUnit DetIds
  virtual const DetIdContainer& detUnitIds() const override;

  // Return a vector of all GeomDet DetIds
  virtual const DetIdContainer& detIds() const override;

  // Return the pointer to the GeomDetUnit corresponding to a given DetId
  virtual const GeomDetUnit* idToDetUnit(DetId) const override;

  // Return the pointer to the GeomDet corresponding to a given DetId
  virtual const GeomDet* idToDet(DetId) const override;


  //---- Extension of the interface

  /// Return a vector of all GEM chambers
  //  const std::vector<GEMChamber*>& chambers() const;

  /// Return a vector of all GEM eta partitions
  const std::vector<const GEMEtaPartition*>& etaPartitions() const;

  // Return a GEMChamber given its id
  //  const GEMChamber* chamber(GEMDetId id) const;

  /// Return a etaPartition given its id
  const GEMEtaPartition* etaPartition(GEMDetId id) const;

  /// Add a GEM etaPartition  to the Geometry
  void add(GEMEtaPartition* etaPartition);

  /// Add a GEM Chamber to the Geometry
  //  void add(GEMChamber* ch);

 private:
  DetUnitContainer theEtaPartitions;
  DetContainer theDets;
  DetTypeContainer theEtaPartitionTypes;
  DetIdContainer theEtaPartitionIds;
  DetIdContainer theDetIds;
  
  // Map for efficient lookup by DetId 
  mapIdToDet theMap;

  std::vector<const GEMEtaPartition*> allEtaPartitions; // Are not owned by this class; are owned by their chamber.
  //  std::vector<GEMChamber*> allChambers; // Are owned by this class.

};

#endif
