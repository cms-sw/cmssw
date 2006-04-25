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

  /// Print out layer information (not recursive)  
  friend std::ostream& operator << ( std::ostream &, const AlignableTIDLayer & ); 

  void dump(); /// Dump whole layer structure

  AlignableTIDLayer( const std::vector<AlignableTIDRing*> rings );
  
  ~AlignableTIDLayer();
  
  virtual std::vector<Alignable*> components() const 
  {
	std::vector<Alignable*> result; 
	result.insert( result.end(), theRings.begin(), theRings.end() );
	return result;
  }

  AlignableTIDRing &ring (int i);

  /// Alignable object identifier
  virtual int alignableObjectId () const {return AlignableObjectId::AlignableTIDLayer;}

 private:
  // put the layer in on the beam Axis and at the average z of the Rings
  PositionType computePosition();
  
  // actually this is set to defaut... NO rotation, hence just the original
  // orientation of the CMS frame...
  RotationType computeOrientation();

  // get the Surface
  AlignableSurface computeSurface();

  std::vector<AlignableTIDRing*> theRings;

};

#endif //AlignableTIDLayer_H




