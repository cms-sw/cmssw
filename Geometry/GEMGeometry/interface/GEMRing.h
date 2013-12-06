#ifndef GEMGeometry_GEMRing_h
#define GEMGeometry_GEMRing_h

/** \class GEMRing
 *
 *  Model of a GEM Ring
 *   
 *  A ring is composed of 36 super chambers
 *
 *  \author S. Dildick
 */

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

class GEMSuperChamber;

class GEMRing
{
 public:
  /// constructor
  GEMRing(int region, int station, int ring);

  /// destructor
  ~GEMRing();

  /// Return the vector of GEMDetIds in this ring
  std::vector<GEMDetId> ids() const;

  /// equal if the id is the same
  bool operator==(const GEMRing& sch) const;

  /// Add super chamber to the ring which takes ownership
  void add(GEMSuperChamber* ch);
  
  /// Return the super chambers in the ring
  virtual std::vector<const GeomDet*> components() const;

  /// Return the sub-component (super chamber) with a given id in this ring
  virtual const GeomDet* component(DetId id) const;

  /// Return the chamber corresponding to the given id 
  const GEMSuperChamber* superChamber(GEMDetId id) const;

  // Return a super chamber
  const GEMSuperChamber* superChamber(int sch) const;
  
  /// Return the chambers in the ring
  std::vector<const GEMSuperChamber*> superChambers() const;

  /// Return numbers of chambers
  int nSuperChambers() const;

  /// Return the region number
  int region() const;
  
  /// Return the station number
  int station() const;
  
  /// Return the ring number
  int ring() const;

 private:

  int region_;
  int station_;
  int ring_;

  std::vector<GEMDetId> detIds_;

  // vector of chambers for a super chamber
  std::vector<const GEMSuperChamber*> superChambers_;

};
#endif
