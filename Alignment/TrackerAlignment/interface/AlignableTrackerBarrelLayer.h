#ifndef Alignment_TrackerAlignment_AlignableTrackerBarrelLayer_H
#define Alignment_TrackerAlignment_AlignableTrackerBarrelLayer_H

#include <vector>
#include <iomanip>

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/TrackerAlignment/interface/AlignableTrackerRod.h"


/// The AlignableTrackerBarrelLayer composite consists of all the Rods in a Layer

class AlignableTrackerBarrelLayer: public AlignableComposite 
{

public:

  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;

  /// Constructor from rods
  AlignableTrackerBarrelLayer( const std::vector<AlignableTrackerRod*> rods );
  
  /// Destructor
  ~AlignableTrackerBarrelLayer();
  
  /// Return all components
  virtual std::vector<Alignable*> components() const;

  /// Return rod at given index
  AlignableTrackerRod &rod ( int i );

  /// Twist layer by given angle
  virtual void twist( float radians );

  /// Alignable object identifier
  virtual int alignableObjectId() const { return AlignableObjectId::AlignableBarrelLayer; }

  /// Printout layer information (not recursive)
  friend std::ostream& operator << ( std::ostream &, const AlignableTrackerBarrelLayer & ); 

  /// Recursive printout of layer structure
  void dump( void );

private:
  /// Get layer position  (on the beam Axis and at the average z of the rods)
  PositionType computePosition();

  /// Get layer orientation (no rotation by default)
  RotationType computeOrientation();

  // Get the Surface
  AlignableSurface computeSurface();

  std::vector<AlignableTrackerRod*> theRods;

};

#endif //AlignableTrackerBarrelLayer_H




