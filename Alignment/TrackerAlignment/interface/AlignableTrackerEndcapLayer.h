#ifndef Alignment_TrackerAlignment_AlignableTrackerEndcapLayer_H
#define Alignment_TrackerAlignment_AlignableTrackerEndcapLayer_H

#include <vector>
#include <iomanip>

#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/TrackerAlignment/interface/AlignableTrackerPetal.h"


/// The AlignableTrackerEndcapLayer is composed of all the Petals in a disk (or wheel)

class AlignableTrackerEndcapLayer: public AlignableComposite 
{

public:
  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;
  
  friend std::ostream& operator << ( std::ostream &, const AlignableTrackerEndcapLayer & ); 
  void dump( void );

  AlignableTrackerEndcapLayer( const std::vector<AlignableTrackerPetal*> petals );
  
  ~AlignableTrackerEndcapLayer();
  
  virtual std::vector<Alignable*> components() const 
  {
    std::vector<Alignable*> result; 
	result.insert( result.end(), thePetals.begin(), thePetals.end());
    return result;
  }

  AlignableTrackerPetal &petal (int i);

  /// Alignable object identifier
  virtual int alignableObjectId () const 
  {
	return AlignableObjectId::AlignableEndcapLayer;
  }

 private:
  // put the layer in on the beam Axis and at the average z of the Petals
  PositionType computePosition();
  
  // actually this is set to defaut... NO rotation, hence just the original
  // orientation of the CMS frame...
  RotationType computeOrientation();

  // get the Surface
  AlignableSurface computeSurface();

  std::vector<AlignableTrackerPetal*> thePetals;  //collection of Petals...

};

#endif //AlignableTrackerEndcapLayer_H




