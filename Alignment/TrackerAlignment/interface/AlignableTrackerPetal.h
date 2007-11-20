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
  typedef std::vector<const GeomDet*> GeomDetContainer;

  /// Printout all DetUnits in the Petal
  friend std::ostream& operator << ( std::ostream &, const AlignableTrackerPetal & ); 
  
  /// Constructor from GeomDets
  AlignableTrackerPetal( GeomDetContainer& geomDets );

  /// Destructor
  ~AlignableTrackerPetal();
  
  /// Return all components of the Petal
  virtual std::vector<Alignable*> components() const ;

  /// Return AlignableDet at given index
  AlignableDet &det(int i);

  /// Alignable object identifier
  virtual int alignableObjectId() const { return AlignableObjectId::AlignablePetal; }

 private:

  /// Gets the global position as the average over all Dets in the Petal
  PositionType computePosition(); 
  /// Get the global orientation
  RotationType computeOrientation();
  /// Get the Surface
  AlignableSurface computeSurface();

  std::vector<AlignableDet*> theDets;

};


#endif  // ALIGNABLE_PETAL_H


