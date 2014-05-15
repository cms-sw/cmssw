#ifndef GEMGeometry_GEMStation_h
#define GEMGeometry_GEMStation_h

/** \class GEMStation
 *
 *  Model of a GEM Station
 *   
 *  A station is composed of maximal 3 rings
 *
 *  \author S. Dildick
 */

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "Geometry/GEMGeometry/interface/GEMRing.h"

class GEMSuperChamber;

class GEMStation
{
 public:
  /// constructor
  GEMStation(int region, int station);

  /// destructor
  ~GEMStation();

  /// Return the vector of GEMDetIds in this station
  std::vector<GEMDetId> ids() const;

  /// equal if the id is the same
  bool operator==(const GEMStation& st) const;

  /// Add ring to the station which takes ownership
  void add(GEMRing* ring);
  
  /// Return the super chambers in the station
  virtual std::vector<const GeomDet*> components() const;

  /// Return the sub-component (super chamber) with a given id in this station
  virtual const GeomDet* component(DetId id) const;

  /// Return the chamber corresponding to the given id 
  const GEMSuperChamber* superChamber(GEMDetId id) const;

  /// Return the super chambers in the region
  std::vector<const GEMSuperChamber*> superChambers() const;

  /// Get a ring
  const GEMRing* ring(int ring) const;
  
  /// Return the rings in the station
  const std::vector<const GEMRing*>& rings() const;

  /// Return numbers of rings for this station
  int nRings() const;

  /// Set the station name
  void setName(std::string name); 

  /// Set the station name
  const std::string getName() const; 

  /// Get the region 
  int region() const;
  
  /// Get the station
  int station() const;

 private:

  int region_;
  int station_;

  // vector of rings for a station
  std::vector<const GEMRing*> rings_;
  std::string name_;

};
#endif
