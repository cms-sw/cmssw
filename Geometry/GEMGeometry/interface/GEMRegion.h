#ifndef GEMGeometry_GEMRegion_h
#define GEMGeometry_GEMRegion_h

/** \class GEMRegion
 *
 *  Model of a GEM Region
 *   
 *  A region has maximal 5 stations
 *
 *  \author S. Dildick
 */

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"

class GEMStation;
class GEMSuperChamber;

class GEMRegion
{
 public:
  /// constructor
  GEMRegion(int region);

  /// destructor
  ~GEMRegion();

  /// Return the vector of GEMDetIds in this ring
  std::vector<GEMDetId> ids() const;

  /// equal if the id is the same
  bool operator==(const GEMRegion& reg) const;

  /// Add station to the region which takes ownership
  void add(GEMStation* st);
  
  /// Return the super chambers in the region
  virtual std::vector<const GeomDet*> components() const;

  /// Return the sub-component (super chamber) with a given id in this region
  virtual const GeomDet* component(DetId id) const;

  /// Return the super chamber corresponding to the given id 
  const GEMSuperChamber* superChamber(GEMDetId id) const;
  
  /// Return the super chambers in the region
  std::vector<const GEMSuperChamber*> superChambers() const;

  /// Return a station 
  const GEMStation* station(int st) const;

  /// Return all stations
  const std::vector<const GEMStation*>& stations() const;

  /// Return numbers of stations
  int nStations() const;

  /// Return the region
  int region() const;

 private:

  int region_;
  // vector of stations for a region
  std::vector<const GEMStation*> stations_;

};
#endif
