#ifndef Alignment_TrackerAlignment_AlignableTrackerEndcap_H
#define Alignment_TrackerAlignment_AlignableTrackerEndcap_H

#include <vector>
#include <iomanip>

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/TrackerAlignment/interface/AlignableTrackerEndcapLayer.h"

#include "Geometry/TrackerNumberingBuilder/interface/GeometricDet.h"

/// The AlignableTrackerEndcap is made of AlignableTrackerEndcapLayers,
/// which are geometrically discs (or wheels...)
/// They are separated in forward and backward (positive and negative z).

class AlignableTrackerEndcap: public AlignableComposite 
{

public:
  
  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;

  /// Constructor from list of layers
  AlignableTrackerEndcap( const std::vector<AlignableTrackerEndcapLayer*> endcapLayers );

  /// Destructor
  ~AlignableTrackerEndcap();


  /// Return all components
  virtual std::vector<Alignable*> components() const;

  /// Return layer at given index
  AlignableTrackerEndcapLayer &layer(int i);

  /// Return alignable object identifier
  virtual int alignableObjectId() const { return AlignableObjectId::AlignableEndcap; }

  /// Print out Endcap information (not recursive)
  friend std::ostream& operator << ( std::ostream&, const AlignableTrackerEndcap& ); 

  /// Recursive printout of the Endcap structure
  void dump( void );

 private:

  /// Get the global position as the average over all positions of the layers
  PositionType computePosition(); 
  /// Get the global orientation 
  RotationType computeOrientation(); //see explanation for "theOrientation"
  /// Get the Surface
  AlignableSurface computeSurface();

  std::vector<AlignableTrackerEndcapLayer*> theEndcapLayers;

};

#endif //AlignableTrackerEndcap_H








