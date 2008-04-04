#ifndef Alignment_MuonAlignment_AlignableCSCRing_H
#define Alignment_MuonAlignment_AlignableCSCRing_H

/** \class AlignableCSCRing 
 *  The alignable muon CSC ring.
 *
 *  $Date: Tue Feb 12 04:44:18 CST 2008 $
 *  $Revision: 1.0.0.0 $
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
  virtual StructureType alignableObjectId() const { return align::AlignableCSCRing; }

  /// Printout muon CSC Ring information (not recursive)
  friend std::ostream& operator << ( std::ostream&, const AlignableCSCRing& ); 

  /// Recursive printout of the muon CSC Ring structure
  void dump( void );



private:

  std::vector<AlignableCSCChamber*> theCSCChambers;


};

#endif




