#ifndef Alignment_TrackerAlignment_AlignablePixelHalfBarrelLayer_H
#define Alignment_TrackerAlignment_AlignablePixelHalfBarrelLayer_H

#include <vector>
#include <iomanip>

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/TrackerAlignment/interface/AlignableTrackerRod.h"

/// The AlignablePixelHalfBarrelLayer is made of all the Rods in a Layer

class AlignablePixelHalfBarrelLayer: public AlignableComposite 
{

public:

  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;

  /// Constructor from rods
  AlignablePixelHalfBarrelLayer( const std::vector<AlignableTrackerRod*> rods );
  
  /// Destructor
  ~AlignablePixelHalfBarrelLayer();
  
  /// Return all components
  virtual std::vector<Alignable*> components() const;

  /// Return rod at given index
  AlignableTrackerRod& rod (int i);

  /// Return ladder (== rod) at given index (for backward compatibility)
  AlignableTrackerRod& ladder (int i) { return rod(i); }

  /// Twist layer by given angle (in radians)
  virtual void twist( float radians );

  /// Alignable object identifier
  virtual int alignableObjectId() const { return AlignableObjectId::AlignablePixelHalfBarrelLayer; }

  /// Printout layer information (not recursive)
  friend std::ostream& operator << (std::ostream &, const AlignablePixelHalfBarrelLayer & ); 

  /// Recursive printout of the layer structure
  void dump( void );

private:
  
  /// Get the layer position (on the beam Axis and at the average x of the components)
  PositionType computePosition();

  /// Get the layer orientation (no rotation by default)
  RotationType computeOrientation();

  // Get the Surface
  AlignableSurface computeSurface();

  std::vector<AlignableTrackerRod*> theRods;

};

#endif //AlignablePixelHalfBarrelLayer_H




