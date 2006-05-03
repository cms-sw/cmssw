#ifndef Alignment_TrackerAlignment_AlignableTIDLayer_H
#define Alignment_TrackerAlignment_AlignableTIDLayer_H

#include <iomanip>
#include <vector>

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/TrackerAlignment/interface/AlignableTIDRing.h"

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"


/// An AlignableTIDLayer is composed of all the Rings in a TID layer.

class AlignableTIDLayer: public AlignableComposite
{
  
public:

  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;


  /// Constructor from rings
  AlignableTIDLayer( const std::vector<AlignableTIDRing*> rings );
  
  /// Destructor
  ~AlignableTIDLayer();
  
  /// Return list of all components
  virtual std::vector<Alignable*> components() const;

  /// Return ring at given index
  AlignableTIDRing &ring (int i);

  /// Return alignable object identifier
  virtual int alignableObjectId() const { return AlignableObjectId::AlignableTIDLayer; }

  /// Printout layer information (not recursive)  
  friend std::ostream& operator << ( std::ostream &, const AlignableTIDLayer & ); 

  /// Recursive printout of layer structure
  void dump();

private:

  /// Get the layer position (on the beam Axis and at the average z of the rings)
  PositionType computePosition();

  /// Get the layer orientation (no rotation by default)
  RotationType computeOrientation();

  /// Get the Surface
  AlignableSurface computeSurface();

  std::vector<AlignableTIDRing*> theRings;

};

#endif //AlignableTIDLayer_H




