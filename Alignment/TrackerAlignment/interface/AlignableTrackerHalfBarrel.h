#ifndef Alignment_TrackerAlignment_AlignableTrackerHalfBarrel_H
#define Alignment_TrackerAlignment_AlignableTrackerHalfBarrel_H

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/TrackerAlignment/interface/AlignableTrackerBarrelLayer.h"

// Geometry interface
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"


/// An AlignableTrackerHalfBarrel is made of all Rods in the forward or backward
/// half of the Barrel

class AlignableTrackerHalfBarrel: public AlignableComposite 
{
  
public:
  
  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;
  
  /// Constructor from barrel layers
  AlignableTrackerHalfBarrel( const std::vector<AlignableTrackerBarrelLayer*> barrelLayers );
  
  /// Desctructor
  ~AlignableTrackerHalfBarrel();

  /// Return all components of half barrel
  virtual std::vector<Alignable*> components() const;
  
  /// Return layer at given index
  AlignableTrackerBarrelLayer &layer( int i );
  
  /// Twist all components by given angle
  virtual void twist( float radians );
  
  /// Return alignable object identifier
  virtual int alignableObjectId() const { return AlignableObjectId::AlignableHalfBarrel; }

  /// Printout half barrel information (not recursive)
  friend std::ostream& operator << ( std::ostream&, const AlignableTrackerHalfBarrel& ); 

  /// Recursive printout of the half barrel structure
  void dump( void );
  
  
private:

  /// Get the global position as the average over all components
  PositionType computePosition(); 
  /// Get the global orientation
  RotationType computeOrientation();
  /// Get the Surface
  AlignableSurface computeSurface();
  
  std::vector<AlignableTrackerBarrelLayer*> theBarrelLayers;

};

#endif //AlignableTrackerHalfBarrel_H








