#ifndef Alignment_TrackerAlignment_AlignableTrackerPetal_H
#define Alignment_TrackerAlignment_AlignableTrackerPetal_H


#include <iomanip> 
#include <vector>

#include "Geometry/CommonDetUnit/interface/GeomDetUnit.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"


/// A composite of AlignableDets for a phi segment of an endcap layer
class AlignableTrackerPetal: public AlignableComposite 
{
public:

  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;

  friend std::ostream& operator << ( std::ostream &, const AlignableTrackerPetal & ); 
  

  AlignableTrackerPetal( std::vector<GeomDet*>& geomDets );

  ~AlignableTrackerPetal();
  
  virtual std::vector<Alignable*> components() const ;

  AlignableDet &det(int i);

  /// Alignable object identifier
  virtual int alignableObjectId () const {
    return AlignableObjectId::AlignablePetal;
  }

 private:

  // gets the global position as the average over all Dets in the Petal
  PositionType computePosition(); 
  // get the global orientation
  RotationType computeOrientation(); //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface();

  std::vector<AlignableDet*> theDets;

};


#endif  // ALIGNABLE_PETAL_H


