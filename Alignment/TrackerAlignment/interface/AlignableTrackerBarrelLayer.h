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
  
  friend std::ostream& operator << ( std::ostream &, const AlignableTrackerBarrelLayer & ); 
  void dump( void );

  AlignableTrackerBarrelLayer( const std::vector<AlignableTrackerRod*> rods );
  
  ~AlignableTrackerBarrelLayer();
  
  virtual std::vector<Alignable*> components() const
  {
    std::vector<Alignable*> result; 
	result.insert( result.end(), theRods.begin(), theRods.end());
    return result;
  }

  AlignableTrackerRod &rod (int i);

  /// a BarrelLayer ist twisted by rotating each Rod around the original center
  /// (before any "mis-alignment"... e.g. the nominal position...
  /// here you have to watch out! once the nominal position might include 
  /// already some "aligned" detector)
  /// and with the orientation of +/- its original local z-axis. Furthermore
  /// the rotation angle is calculated from the rod length... which currently 
  /// is simply calculated from the GeomDetUnits on the rod... and NOT from 
  /// the distance between the two supporting barrel disks... which would be 
  /// more correct...
  virtual void twist(float);

  /// Alignable object identifier
  virtual int alignableObjectId () const 
  { 
    return AlignableObjectId::AlignableBarrelLayer; 
  }

 private:
  // put the layer in on the beam Axis and at the average z of the Rods
  PositionType computePosition();
  
  // actually this is set to defaut... NO rotation, hence just the original
  // orientation of the CMS frame...
  RotationType computeOrientation();

  // get the Surface
  AlignableSurface computeSurface();

  std::vector<AlignableTrackerRod*> theRods;

};

#endif //AlignableTrackerBarrelLayer_H




