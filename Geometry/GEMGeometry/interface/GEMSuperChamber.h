#ifndef GEMGeometry_GEMSuperChamber_h
#define GEMGeometry_GEMSuperChamber_h

/** \class GEMSuperChamber
 *
 *  Model of a GEM super chamber.
 *   
 *  The super chamber is composed of 2 chambers,
 *  but does not inherit from GeomDet. It's id is
 *  the same as the ch1 detid
 *
 *  \author S. Dildick
 */

#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include <vector>

class GEMChamber;

class GEMSuperChamber
{
 public:
  /// constructors
  GEMSuperChamber();
  GEMSuperChamber(GEMDetId ch1, GEMDetId ch2);

  /// destructor
  ~GEMSuperChamber();

  /// Return the GEMDetId of this super chamber
  const std::vector<GEMDetId>& ids() const;
  
  /// equal if its ids are the same
  bool operator==(const GEMSuperChamber& sch) const;

  /// Add chamber to the super chamber which takes ownership
  void add(GEMChamber* ch);
  
  /// Return the chamber corresponding to the given id 
  const GEMChamber* chamber(GEMDetId id) const;

  const GEMChamber* chamber(int layer) const;
  
  /// Return the chambers in the super chamber
  const std::vector<const GEMChamber*>& chambers() const;

  /// Return numbers of chambers
  int nChambers() const;

 private:

  std::vector<GEMDetId> detIds_;

  // vector of chambers for a super chamber
  std::vector<const GEMChamber*> chambers_;

};
#endif
