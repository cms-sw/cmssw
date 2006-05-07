#ifndef Alignment_MuonAlignment_AlignableMuBarrel_H
#define Alignment_MuonAlignment_AlignableMuBarrel_H

#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MuonAlignment/interface/AlignableDTWheel.h"

#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "Geometry/Vector/interface/Basic3DVector.h"

#include <vector>

class GeomDet;

/// Concrete class for muon DT Barrel alignable.
///
/// Misalignment can be de-/reactivated (forwarded to components).
///

class AlignableMuBarrel : public AlignableComposite 
{

 public:

  AlignableMuBarrel( const std::vector<AlignableDTWheel*> dtWheels );

  ~AlignableMuBarrel();
  
  virtual std::vector<Alignable*> components() const 
  {

        std::vector<Alignable*> result;
        result.insert( result.end(), theDTWheels.begin(), theDTWheels.end() );
        return result;

  }
  
  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;

  // gets the global position as the average over all positions of the layers
  PositionType computePosition() ;
  // get the global orientation
  RotationType computeOrientation() ; //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface() ;

  AlignableDTWheel &wheel(int i);  
  
  virtual void twist(float);

  /// Printout muon Barrel information (not recursive)
  friend std::ostream& operator << ( std::ostream&, const AlignableMuBarrel& );

  /// Recursive printout of the muon Barrel structure
  void dump( void );




private:

  std::vector<AlignableDTWheel*> theDTWheels;


};

#endif




