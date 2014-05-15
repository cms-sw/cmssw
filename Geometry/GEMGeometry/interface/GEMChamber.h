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
  virtual ~GEMChamber();

  /// Return the GEMDetId of this chamber
  GEMDetId id() const;

  // Which subdetector
  virtual SubDetector subDetector() const {return GeomDetEnumerators::GEM;}

  /// equal if the id is the same
  bool operator==(const GEMChamber& ch) const;

  /// Add EtaPartition to the chamber which takes ownership
  void add(GEMEtaPartition* roll);

  /// Return the rolls in the chamber
  virtual std::vector<const GeomDet*> components() const;

  /// Return the sub-component (roll) with a given id in this chamber
  virtual const GeomDet* component(DetId id) const;

  /// Return the eta partition corresponding to the given id 
  const GEMEtaPartition* etaPartition(GEMDetId id) const;

  const GEMEtaPartition* etaPartition(int isl) const;
  
  /// Return the eta partitions
  const std::vector<const GEMEtaPartition*>& etaPartitions() const;

  /// Retunr numbers of eta partitions
  int nEtaPartitions() const;

private:

  GEMDetId detId_;

  // vector of eta partitions for a chamber
  std::vector<const GEMEtaPartition*> etaPartitions_;

};
#endif
