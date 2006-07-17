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
  typedef std::vector<const GeomDet*> GeomDetContainer;

  /// Constructor from GeomDets
  AlignableTIDRing( GeomDetContainer& geomDets );
  
  /// Destructor
  ~AlignableTIDRing();
  
  /// Return all components
  virtual std::vector<Alignable*> components() const ;

  /// Return AlignableDet at given index
  AlignableDet &det(int i);

  /// Return alignable object identifier
  virtual int alignableObjectId() const { return AlignableObjectId::AlignableTIDRing; }

  /// Printout of DetUnits in the ring
  friend std::ostream& operator << ( std::ostream &, const AlignableTIDRing & ); 
  
private:

  // Get the global position as the average over all Dets in the Ring
  PositionType computePosition(); 

  // Get the global orientation
  RotationType computeOrientation(); //see explanation for "theOrientation"

  // Get the Surface
  AlignableSurface computeSurface();

  std::vector<AlignableDet*> theDets ;

};


#endif  // ALIGNABLE_RING_H


