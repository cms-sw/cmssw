#ifndef Geometry_GEMGeometry_ME0Geometry_h
#define Geometry_GEMGeometry_ME0Geometry_h

#include "DataFormats/DetId/interface/DetId.h"
#include "Geometry/CommonDetUnit/interface/TrackingGeometry.h"
#include "Geometry/GEMGeometry/interface/ME0EtaPartition.h"
#include "Geometry/GEMGeometry/interface/ME0Layer.h"
#include "Geometry/GEMGeometry/interface/ME0Chamber.h"
#include <vector>
#include <map>

class ME0Geometry : public TrackingGeometry {

 public:
  /// Default constructor
  ME0Geometry();

  /// Destructor
  ~ME0Geometry() override;

  // Return a vector of all det types
  const DetTypeContainer&  detTypes() const override;

  // Return a vector of all GeomDetUnit
  const DetContainer& detUnits() const override;

  // Return a vector of all GeomDet
  const DetContainer& dets() const override;
  
  // Return a vector of all GeomDetUnit DetIds
  const DetIdContainer& detUnitIds() const override;

  // Return a vector of all GeomDet DetIds
  const DetIdContainer& detIds() const override;

  // Return the pointer to the GeomDetUnit corresponding to a given DetId
  const GeomDet* idToDetUnit(DetId) const override;

  // Return the pointer to the GeomDet corresponding to a given DetId
  const GeomDet* idToDet(DetId) const override;


  //---- Extension of the interface

  /// Return a etaPartition given its id
  const ME0EtaPartition* etaPartition(ME0DetId id) const;

  /// Return a layer given its id
  const ME0Layer* layer(ME0DetId id) const;

  /// Return a chamber given its id
  const ME0Chamber* chamber(ME0DetId id) const;


  /// Return a vector of all ME0 eta partitions
  const std::vector<ME0EtaPartition const*>& etaPartitions() const;

  /// Return a vector of all ME0 layers
  const std::vector<const ME0Layer*>& layers() const;

  /// Return a vector of all ME0 chambers
  const std::vector<const ME0Chamber*>& chambers() const;

  /// Add a ME0 etaPartition  to the Geometry
  void add(ME0EtaPartition* etaPartition);

  /// Add a ME0 layer  to the Geometry
  void add(ME0Layer* layer);

  /// Add a ME0 Chamber  to the Geometry
  void add(ME0Chamber* chamber);

 private:
  DetContainer theEtaPartitions;
  DetTypeContainer theEtaPartitionTypes;
  DetIdContainer   theEtaPartitionIds;
  DetIdContainer   theDetIds;
  DetContainer     theDets;
  
  // Map for efficient lookup by DetId 
  mapIdToDet theMap;

  std::vector<ME0EtaPartition const*> allEtaPartitions; // Are not owned by this class; are owned by their layer.
  std::vector<ME0Layer const*> allLayers;               // Are not owned by this class; are owned by their chamber.
  std::vector<ME0Chamber const*> allChambers;           // Are owned by this class.

};

#endif
