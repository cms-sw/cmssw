#ifndef Alignment_TrackerAlignment_AlignablePixelHalfBarrel_H
#define Alignment_TrackerAlignment_AlignablePixelHalfBarrel_H

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/TrackerAlignment/interface/AlignablePixelHalfBarrelLayer.h"

// Geometry interface
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"



/// AlignablePixelHalfBarrel corresponds to half of the pixel barrel, either x>0 or x<0 . 
/// The cut plane is a vertical plane along the beam (z)

class AlignablePixelHalfBarrel: public AlignableComposite 
{

public:
  
  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;

  /// Constructor from pixel barrel layers
  AlignablePixelHalfBarrel( const std::vector<AlignablePixelHalfBarrelLayer*> barrelLayers );

  /// Destructor
  ~AlignablePixelHalfBarrel();

  /// Return all components
  virtual std::vector<Alignable*> components() const;

  /// Return layer at given index
  AlignablePixelHalfBarrelLayer &layer( int i );

  /// Twist all components by given angle
  virtual void twist( float radians );

  /// Alignable object identifier
  virtual int alignableObjectId() const { return AlignableObjectId::AlignablePixelHalfBarrel; }

  /// Printout Half Barrel information (not recursive)
  friend std::ostream& operator << ( std::ostream&, const AlignablePixelHalfBarrel& ); 

  /// Recursive printout of half barrel structure
  void dump( void );

 private:

  /// Get position of barrel (average of y and z over all DetUnits)
  PositionType computePosition(); 

  // Get global orientation (no rotation by default)
  RotationType computeOrientation();

  // Get the Surface
  AlignableSurface computeSurface();

  std::vector<AlignablePixelHalfBarrelLayer*> thePixelHalfBarrelLayers;

};

#endif //AlignablePixelHalfBarrel_H








