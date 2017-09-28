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

  // Return a vector of all det types
  const DetTypeContainer&  detTypes() const override;

  // Return a vector of all GeomDet
  const DetContainer& detUnits() const override;

  // Return a vector of all GeomDet
  const DetContainer& dets() const override;
  
  // Return a vector of all GeomDet DetIds
  const DetIdContainer& detUnitIds() const override;

  // Return a vector of all GeomDet DetIds
  const DetIdContainer& detIds() const override;

  // Return the pointer to the GeomDet corresponding to a given DetId
  const std::shared_ptr< GeomDet > idToDetUnit( DetId ) const override;

  // Return the pointer to the GeomDet corresponding to a given DetId
  const std::shared_ptr< GeomDet > idToDet( DetId ) const override;


  //---- Extension of the interface

  /// Return a vector of all GEM regions
  const std::vector< std::shared_ptr< GEMRegion >>& regions() const;

  /// Return a vector of all GEM stations
  const std::vector< std::shared_ptr< GEMStation >>& stations() const;

  /// Return a vector of all GEM rings
  const std::vector< std::shared_ptr< GEMRing >>& rings() const;

  /// Return a vector of all GEM super chambers
  const std::vector< std::shared_ptr< GEMSuperChamber >>& superChambers() const;

  /// Return a vector of all GEM chambers
  const std::vector< std::shared_ptr< GEMChamber >>& chambers() const;

  /// Return a vector of all GEM eta partitions
  const std::vector< std::shared_ptr< GEMEtaPartition >>& etaPartitions() const;

  // Return a GEMRegion 
  const std::shared_ptr< GEMRegion > region(int region) const;

  // Return a GEMStation
  const std::shared_ptr< GEMStation > station( int region, int station ) const;

  /// Return a GEMRing
  const std::shared_ptr< GEMRing > ring( int region, int station, int ring ) const;

  // Return a GEMSuperChamber given its id
  const std::shared_ptr< GEMSuperChamber > superChamber( GEMDetId id ) const;

  // Return a GEMChamber given its id
  const std::shared_ptr< GEMChamber > chamber( GEMDetId id ) const;

  /// Return a GEMEtaPartition given its id
  const std::shared_ptr< GEMEtaPartition > etaPartition( GEMDetId id ) const;

  /// Add a GEMRegion to the Geometry
  void add( std::shared_ptr< GEMRegion > region);

  /// Add a GEMStation to the Geometry
  void add( std::shared_ptr< GEMStation > station);

  /// Add a GEMRing to the Geometry
  void add( std::shared_ptr< GEMRing > ring);

  /// Add a GEMSuperChamber to the Geometry
  void add( std::shared_ptr< GEMSuperChamber > sch);
 
  /// Add a GEMChamber to the Geometry
  void add( std::shared_ptr< GEMChamber > ch);

  /// Add a GEMEtaPartition  to the Geometry
  void add( std::shared_ptr< GEMEtaPartition > etaPartition);

 private:
  DetContainer theEtaPartitions;
  DetContainer theDets;
  DetTypeContainer theEtaPartitionTypes;
  DetIdContainer theEtaPartitionIds;
  DetIdContainer theDetIds;
  
  // Map for efficient lookup by DetId 
  mapIdToDet theMap;

  std::vector< std::shared_ptr< GEMEtaPartition >> allEtaPartitions;
  std::vector< std::shared_ptr< GEMChamber >> allChambers;
  std::vector< std::shared_ptr< GEMSuperChamber >> allSuperChambers;
  std::vector< std::shared_ptr< GEMRing >> allRings;
  std::vector< std::shared_ptr< GEMStation >> allStations;
  std::vector< std::shared_ptr< GEMRegion >> allRegions;
};

#endif
