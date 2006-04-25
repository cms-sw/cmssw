#ifndef Alignment_TrackerAlignment_AlignablePxHalfBarrel_H
#define Alignment_TrackerAlignment_AlignablePxHalfBarrel_H

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/TrackerAlignment/interface/AlignablePxHalfBarrelLayer.h"

// Geometry interface
#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"



/// AlignablePxHalfBarrel corresponds to half of the pixel barrel, either x>0 or x<0 . 
/// The cut plane is a vertical plane along the beam (z)
/// It consists of all (half)layers in this half of the barrel.
///
/// At the moment (Nov 2004) the half layers are not
/// implemented in OSCAR/ORCA, so the whole layers
/// are divided first according to x, and if x is 
/// sufficiently near 0, then according to y: layer with
/// positive y goes to the halfbarrel containing positive
/// x layers
class AlignablePxHalfBarrel: public AlignableComposite 
{

 public:
  
  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;


  AlignablePxHalfBarrel( const std::vector<AlignablePxHalfBarrelLayer*> barrelLayers );

  ~AlignablePxHalfBarrel();

  virtual std::vector<Alignable*> components() const 
  {

	std::vector<Alignable*> result; 
	result.insert( result.end(), thePxHalfBarrelLayers.begin(), thePxHalfBarrelLayers.end() );
	return result;

  }

  AlignablePxHalfBarrelLayer &layer(int i);

  virtual void twist(float);

  /// Alignable object identifier
  virtual int alignableObjectId () const 
  {
	return AlignableObjectId::AlignablePxHalfBarrel;
  }

  /// Print out Half Barrel information (not recursive)
  friend std::ostream& operator << ( std::ostream&, const AlignablePxHalfBarrel& ); 

  void dump( void ); /// Dump the whole Half Barrel structure

 private:
  // gets the global position as centre of the cutting plane (about vertex)
  // average of y and z over all DetUnits, but not x
  PositionType computePosition(); 
  // get the global orientation
  RotationType computeOrientation(); //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface();


  std::vector<AlignablePxHalfBarrelLayer*> thePxHalfBarrelLayers;

};

#endif //AlignablePxHalfBarrel_H








