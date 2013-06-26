#ifndef Alignment_MuonAlignment_AlignableCSCRing_H
#define Alignment_MuonAlignment_AlignableCSCRing_H

/** \class AlignableCSCRing 
 *  The alignable muon CSC ring.
 *
 *  $Date: 2011/09/15 10:07:07 $
 *  $Revision: 1.4 $
 *  \author Jim Pivarski - Texas A&M University
 */


#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MuonAlignment/interface/AlignableCSCChamber.h"


#include <vector>

class GeomDet;
class AlignableCSCChamber;

/// Concrete class for muon CSC Ring alignable.
///
/// Misalignment can be de-/reactivated (forwarded to components).
///

class AlignableCSCRing : public AlignableComposite 
{

 public:

  AlignableCSCRing( const std::vector<AlignableCSCChamber*> cscChambers );

  ~AlignableCSCRing();
  
  virtual std::vector<Alignable*> components() const 
  {

        std::vector<Alignable*> result;
        result.insert( result.end(), theCSCChambers.begin(), theCSCChambers.end() );
        return result;

  }
  
  // gets the global position as the average over all positions of the layers
  PositionType computePosition() ;
  // get the global orientation
  RotationType computeOrientation() ; //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface() ;

  AlignableCSCChamber &chamber(int i);  
  
  //virtual void twist(float);

  /// Printout muon CSC Ring information (not recursive)
  friend std::ostream& operator << ( std::ostream&, const AlignableCSCRing& ); 

  /// Recursive printout of the muon CSC Ring structure
  void dump( void ) const;



private:

  std::vector<AlignableCSCChamber*> theCSCChambers;


};

#endif




