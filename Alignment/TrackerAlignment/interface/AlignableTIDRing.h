#ifndef Alignment_TrackerAlignment_AlignableTIDRing_H
#define Alignment_TrackerAlignment_AlignableTIDRing_H


#include <iomanip>
#include <vector>

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"


/// A composite of AlignableDets for one TID ring
class AlignableTIDRing: public AlignableComposite 
{
public:

  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;

  friend std::ostream& operator << ( std::ostream &, const AlignableTIDRing & ); 
  

  AlignableTIDRing( std::vector<GeomDet*>& geomDets );
  
  ~AlignableTIDRing();
  
  virtual std::vector<Alignable*> components() const ;

  AlignableDet &det(int i);

  /// Alignable object identifier
  virtual int alignableObjectId () const 
  {
    return AlignableObjectId::AlignableTIDRing;
  }

private:

  // gets the global position as the average over all Dets in the Ring
  PositionType computePosition(); 
  // get the global orientation
  RotationType computeOrientation(); //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface();

  std::vector<AlignableDet*> theDets ;

};


#endif  // ALIGNABLE_RING_H


