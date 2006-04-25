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


  AlignableTrackerEndcap( const std::vector<AlignableTrackerEndcapLayer*> endcapLayers );

  ~AlignableTrackerEndcap();

  virtual std::vector<Alignable*> components() const 
  {

	std::vector<Alignable*> result; 
	result.insert( result.end(), theEndcapLayers.begin(), theEndcapLayers.end() );
	return result;

  } 

  AlignableTrackerEndcapLayer &layer(int i);

  /// Alignable object identifier
  virtual int alignableObjectId () const 
  {
    return AlignableObjectId::AlignableEndcap;
  }

  /// Print out Endcap information (not recursive)
  friend std::ostream& operator << ( std::ostream&, const AlignableTrackerEndcap& ); 

  void dump( void ); /// Dump the whole Endcap structure

 private:
  // gets the global position as the average over all positions of the layers
  PositionType computePosition(); 
  // get the global orientation 
  RotationType computeOrientation(); //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface();


  std::vector<AlignableTrackerEndcapLayer*> theEndcapLayers;

};

#endif //AlignableTrackerEndcap_H








