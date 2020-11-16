#ifndef Geometry_GEMGeometry_GEMSuperChamber_h
#define Geometry_GEMGeometry_GEMSuperChamber_h

/** \class GEMSuperChamber
 *
 *  Model of a GEM super chamber.
 *   
 *  The super chamber is composed of 2 chambers.
 *  It's detId is layer 0, chambers are layer 1 and 2
 *
 *  \author S. Dildick
 */

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

class GEMChamber;

class GEMSuperChamber : public GeomDet {
public:
  /// constructor
  GEMSuperChamber(GEMDetId id, const ReferenceCountingPointer<BoundPlane>& plane);

  /// destructor
  ~GEMSuperChamber() override;

  /// Return the GEMDetId of this super chamber
  GEMDetId id() const;

  /// Return the vector of GEMDetIds in this super chamber
  const std::vector<GEMDetId>& ids() const;

  // Which subdetector
  SubDetector subDetector() const override { return GeomDetEnumerators::GEM; }

  /// equal if the id is the same
  bool operator==(const GEMSuperChamber& sch) const;

  /// Add chamber to the super chamber which takes ownership
  void add(const GEMChamber* ch);

  /// Return the chambers in the super chamber
  std::vector<const GeomDet*> components() const override;

  /// Return the sub-component (chamber) with a given id in this super chamber
  const GeomDet* component(DetId id) const override;

  /// Return the chamber corresponding to the given id
  const GEMChamber* chamber(GEMDetId id) const;

  const GEMChamber* chamber(int layer) const;

  /// Return the chambers in the super chamber
  const std::vector<const GEMChamber*>& chambers() const;

  /// Return numbers of chambers
  int nChambers() const;

  float computeDeltaPhi(const LocalPoint& position, const LocalVector& direction) const;

private:
  GEMDetId detId_;

  // vector of chambers for a super chamber
  std::vector<const GEMChamber*> chambers_;
};
#endif
