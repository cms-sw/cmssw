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

  /// Constructor
  AlignableTrackerEndcapLayer( const std::vector<AlignableTrackerPetal*> petals );
  
  /// Destructor
  ~AlignableTrackerEndcapLayer();

  /// Return all components
  virtual std::vector<Alignable*> components() const;

  /// Return petal at given index
  AlignableTrackerPetal &petal (int i);

  /// Return alignable object identifier
  virtual int alignableObjectId() const { return AlignableObjectId::AlignableEndcapLayer; }

  /// Printout of the layer information (not recursive)
  friend std::ostream& operator << ( std::ostream &, const AlignableTrackerEndcapLayer & ); 

  /// Recursive printout of the layer structure
  void dump( void );

 private:
  /// Get layer position, on the beam Axis and at the average z of the Petals
  PositionType computePosition();
  
  /// Get orientation (zero by default)
  RotationType computeOrientation();

  /// Get the Surface
  AlignableSurface computeSurface();

  std::vector<AlignableTrackerPetal*> thePetals;

};

#endif //AlignableTrackerEndcapLayer_H




