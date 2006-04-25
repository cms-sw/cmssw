#ifndef Alignment_TrackerAlignment_AlignablePxHalfBarrelLayer_H
#define Alignment_TrackerAlignment_AlignablePxHalfBarrelLayer_H

#include <vector>
#include <iomanip>

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/TrackerAlignment/interface/AlignableTrackerRod.h"

/// The AlignablePxHalfBarrelLayer is made of all the Rods in a Layer

class AlignablePxHalfBarrelLayer: public AlignableComposite 
{

 public:
  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;
  
  friend std::ostream& operator << (std::ostream &, const AlignablePxHalfBarrelLayer & ); 
  void dump( void );

  AlignablePxHalfBarrelLayer( const std::vector<AlignableTrackerRod*> rods );
  
  ~AlignablePxHalfBarrelLayer();
  
  virtual std::vector<Alignable*> components() const 
  {
	std::vector<Alignable*> result; 
	result.insert( result.end(), theRods.begin(), theRods.end());
    return result;
  }

  AlignableTrackerRod& rod (int i);

  /// For backward compatibility
  AlignableTrackerRod& ladder (int i) { return rod(i); }

  /// a PxHalfBarrelLayer ist twisted by rotating each AlignableTrackerRod
  /// around the original center (before any "mis-alignment".. e.g. the nominal 
  /// position...here you have to watch out!  once the nomnal position might include 
  /// already some "aligned" detector) and with the orientation of +/- its original 
  /// local z-axis. Furthermore the rotation angle is calculated from the rod 
  /// length....which currently is simply calculated from the detunits on 
  /// the rod... and NOT from the distance between the two supporting barrel
  /// disks....which would be more correct...
  virtual void twist( float );

  /// Alignable object identifier
  virtual int alignableObjectId () const 
  {
    return AlignableObjectId::AlignablePxHalfBarrelLayer;
  }

 private:
  // put the layer in on the beam Axis and at the average x of the Rods
  PositionType computePosition();
  
  // actually this is set to defaut... NO rotation, hence just the original
  // orientation of the CMS frame...
  RotationType computeOrientation();

  // get the Surface
  AlignableSurface computeSurface();

  std::vector<AlignableTrackerRod*> theRods;

};

#endif //AlignablePxHalfBarrelLayer_H




