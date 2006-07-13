#ifndef Alignment_MuonAlignment_AlignableCSCChamber_H
#define Alignment_MuonAlignment_AlignableCSCChamber_H


#include <iosfwd> 
#include <iostream>
#include <vector>

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"

#include "Alignment/CommonAlignment/interface/AlignableDet.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"

/// A muon CSC Chamber( composite of AlignableDets )


class AlignableCSCChamber: public AlignableComposite 
{

 public:

  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;

  friend std::ostream& operator << ( std::ostream &, const AlignableCSCChamber & ); 
  

  /// Constructor from geomdets of the CSCChamber components
  AlignableCSCChamber( GeomDet* geomDet );
  
  ~AlignableCSCChamber();
  
  virtual std::vector<Alignable*> components() const ;

  AlignableDet &det(int i);

  virtual float length() const;

  /// Alignable object identifier
  virtual int alignableObjectId () const { return AlignableObjectId::AlignableCSCChamber; }

 private:
  // gets the global position as the average over all Dets in the Rod
  PositionType computePosition(); 
  // get the global orientation
  RotationType computeOrientation(); //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface();

  std::vector<AlignableDet*> theDets;


};


#endif  // ALIGNABLE_CSC_CHAMBER_H


