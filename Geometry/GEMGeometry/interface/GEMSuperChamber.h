#ifndef GEMGeometry_GEMSuperChamber_h
#define GEMGeometry_GEMSuperChamber_h

/** \class GEMSuperChamber
 *
 *  Model of a GEM super chamber.
 *   
 *  The super chamber is composed of 2 chambers.
 *  It's detId is the same as ch1 detId
 *
 *  \author S. Dildick
 */

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

class GEMChamber;

class GEMSuperChamber : public GeomDet 
{
 public:
  /// constructor
  GEMSuperChamber(GEMDetId id, const ReferenceCountingPointer<BoundPlane>& plane);

  /// destructor
  ~GEMSuperChamber();

  /// Return the GEMDetId of this super chamber
  GEMDetId id() const;
  
  /// Return the vector of GEMDetIds in this super chamber
  const std::vector<GEMDetId>& ids() const;

  // Which subdetector
  virtual SubDetector subDetector() const {return GeomDetEnumerators::GEM;}

  /// equal if the id is the same
  bool operator==(const GEMSuperChamber& sch) const;

  /// Add chamber to the super chamber which takes ownership
  void add(GEMChamber* ch);
  
  /// Return the chambers in the super chamber
  virtual std::vector<const GeomDet*> components() const;

  /// Return the sub-component (chamber) with a given id in this super chamber
  virtual const GeomDet* component(DetId id) const;

  /// Return the chamber corresponding to the given id 
  const GEMChamber* chamber(GEMDetId id) const;

  const GEMChamber* chamber(int layer) const;
  
  /// Return the chambers in the super chamber
  const std::vector<const GEMChamber*>& chambers() const;

  /// Return numbers of chambers
  int nChambers() const;

 private:

  GEMDetId detId_;
  std::vector<GEMDetId> detIds_;

  // vector of chambers for a super chamber
  std::vector<const GEMChamber*> chambers_;

};
#endif
