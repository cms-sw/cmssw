#ifndef Alignment_TrackerAlignment_AlignableTID_H
#define Alignment_TrackerAlignment_AlignableTID_H
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/TrackerAlignment/interface/AlignableTIDLayer.h"

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/// The AlignableTID is composed of AlignableTIDLayers, which are geometrically discs.
/// They are separated in forward and backward (positive and negative z).
/// TID disks are further divided into three rings.

class AlignableTID: public AlignableComposite 
{

 public:
  
  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;

  AlignableTID( const std::vector<AlignableTIDLayer*> tidLayers  );
  
  ~AlignableTID();

  virtual std::vector<Alignable*> components() const 
  {
	std::vector<Alignable*> result; 
	result.insert( result.end(), theLayers.begin(), theLayers.end() );
	return result;
  } 

  AlignableTIDLayer &layer(int i);

  /// Alignable object identifier
  virtual int alignableObjectId () const {return AlignableObjectId::AlignableTID;}

  /// Print out TID information (not recursive)
  friend std::ostream& operator << ( std::ostream&, const AlignableTID& ); 

  void dump( void ); /// Dump the whole TID structure

 private:
  // gets the global position as the average over all positions of the layers
  // as the layers know there position, orientation etc.
  PositionType computePosition(); 

  // get the global orientation 
  RotationType computeOrientation(); //see explanation for "theOrientation"

  // get the Surface
  AlignableSurface computeSurface();

  std::vector<AlignableTIDLayer*> theLayers;

};

#endif //AlignableTID_H








