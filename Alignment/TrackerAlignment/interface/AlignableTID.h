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

  /// Constructor from TID Layers
  AlignableTID( const std::vector<AlignableTIDLayer*> tidLayers  );

  /// Destructor
  ~AlignableTID();

  /// Return components of TID
  virtual std::vector<Alignable*> components() const;

  /// Return layer at given index
  AlignableTIDLayer &layer(int i);

  /// Alignable object identifier
  virtual int alignableObjectId() const { return AlignableObjectId::AlignableTID; }

  /// Printout TID information (not recursive)
  friend std::ostream& operator << ( std::ostream&, const AlignableTID& ); 
  
  /// Recursive printout of TID structure
  void dump( void ); 

private:
  
  /// Get position from average position of components
  PositionType computePosition(); 

  // Get the global orientation (no rotation by default)
  RotationType computeOrientation();

  // Get the Surface
  AlignableSurface computeSurface();

  std::vector<AlignableTIDLayer*> theLayers;

};

#endif //AlignableTID_H








