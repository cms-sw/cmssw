#ifndef Alignment_MuonAlignment_AlignableCSCStation_H
#define Alignment_MuonAlignment_AlignableCSCStation_H

/** \class AlignableCSCStation 
 *  The alignable muon CSC station.
 *
 *  $Date: 2006/8/4 10:00:01 $
 *  $Revision: 1.0 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */


#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MuonAlignment/interface/AlignableCSCChamber.h"

#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "Geometry/Vector/interface/Basic3DVector.h"

#include <vector>

class GeomDet;
class AlignableCSCChamber;

/// Concrete class for muon CSC Station alignable.
///
/// Misalignment can be de-/reactivated (forwarded to components).
///

class AlignableCSCStation : public AlignableComposite 
{

 public:

  AlignableCSCStation( const std::vector<AlignableCSCChamber*> cscChambers );

  ~AlignableCSCStation();
  
  virtual std::vector<Alignable*> components() const 
  {

        std::vector<Alignable*> result;
        result.insert( result.end(), theCSCChambers.begin(), theCSCChambers.end() );
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

  AlignableCSCChamber &chamber(int i);  
  
  //virtual void twist(float);

  /// Return alignable object identifier
  virtual int alignableObjectId() const { return AlignableObjectId::AlignableCSCStation; }

  /// Printout muon CSC Station information (not recursive)
  friend std::ostream& operator << ( std::ostream&, const AlignableCSCStation& ); 

  /// Recursive printout of the muon CSC Station structure
  void dump( void );



private:

  std::vector<AlignableCSCChamber*> theCSCChambers;


};

#endif




