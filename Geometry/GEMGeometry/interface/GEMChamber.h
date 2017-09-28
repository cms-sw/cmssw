#ifndef GEMGeometry_GEMChamber_h
#define GEMGeometry_GEMChamber_h

/** \class GEMChamber
 *
 *  Model of a GEM chamber.
 *   
 *  A chamber is a GeomDet.
 *  The chamber is composed by 6,8 or 10 eta partitions (GeomDetUnit).
 *
 *  \author S. Dildick
 */

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

class GEMEtaPartition;

class GEMChamber : public GeomDet {
public:
  /// Constructor
  GEMChamber(GEMDetId id, const ReferenceCountingPointer<BoundPlane>& plane);

  /// Destructor
  ~GEMChamber() override;

  /// Return the GEMDetId of this chamber
  GEMDetId id() const;

  // Which subdetector
  SubDetector subDetector() const override {return GeomDetEnumerators::GEM;}

  /// equal if the id is the same
  bool operator==(const GEMChamber& ch) const;

  /// Add EtaPartition to the chamber
  void add( std::shared_ptr< GEMEtaPartition > roll );

  /// Return the rolls in the chamber
  std::vector< std::shared_ptr< GeomDet >> components() const override;

  /// Return the sub-component (roll) with a given id in this chamber
  const std::shared_ptr< GeomDet > component( DetId id ) const override;

  /// Return the eta partition corresponding to the given id 
  const std::shared_ptr< GEMEtaPartition > etaPartition( GEMDetId id ) const;

  const std::shared_ptr< GEMEtaPartition > etaPartition( int isl ) const;
  
  /// Return the eta partitions
  const std::vector< std::shared_ptr< GEMEtaPartition >>& etaPartitions() const;

  /// Retunr numbers of eta partitions
  int nEtaPartitions() const;

private:

  GEMDetId detId_;

  // vector of eta partitions for a chamber
  std::vector< std::shared_ptr< GEMEtaPartition >> etaPartitions_;

};
#endif
