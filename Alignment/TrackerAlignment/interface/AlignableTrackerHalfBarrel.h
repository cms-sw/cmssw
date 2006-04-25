#ifndef Alignment_TrackerAlignment_AlignableTrackerHalfBarrel_H
#define Alignment_TrackerAlignment_AlignableTrackerHalfBarrel_H

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/TrackerAlignment/interface/AlignableTrackerBarrelLayer.h"

// Geometry interface
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"


/// An AlignableTrackerHalfBarrel is made of all Rods in the forward or backward
/// half of the Barrel
///

class AlignableTrackerHalfBarrel: public AlignableComposite 
{
  
public:
  
  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;
  

  AlignableTrackerHalfBarrel( const std::vector<AlignableTrackerBarrelLayer*> barrelLayers );
  
  ~AlignableTrackerHalfBarrel();
  
  virtual std::vector<Alignable*> components() const 
  {
	
	std::vector<Alignable*> result; 
	result.insert( result.end(), theBarrelLayers.begin(), theBarrelLayers.end() );
	return result;
	
  }
  
  AlignableTrackerBarrelLayer &layer(int i);
  
  virtual void twist(float);
  
  /// Alignable object identifier
  virtual int alignableObjectId () const 
  {
	return AlignableObjectId::AlignableHalfBarrel;
  }


  /// Print out Half Barrel information (not recursive)
  friend std::ostream& operator << ( std::ostream&, const AlignableTrackerHalfBarrel& ); 

  void dump( void ); /// Dump the whole Half Barrel structure
  
  
private:

  // gets the global position as the average over all DetUnits in the Rod
  PositionType computePosition(); 
  // get the global orientation
  RotationType computeOrientation(); //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface();
  
  
  std::vector<AlignableTrackerBarrelLayer*> theBarrelLayers;

};

#endif //AlignableTrackerHalfBarrel_H








