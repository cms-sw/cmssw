#ifndef Alignment_MuonAlignment_AlignableCSCStation_H
#define Alignment_MuonAlignment_AlignableCSCStation_H

/** \class AlignableCSCStation 
 *  The alignable muon CSC station.
 *
 *  $Date: 2007/10/08 14:12:01 $
 *  $Revision: 1.7 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */


#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MuonAlignment/interface/AlignableCSCChamber.h"


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
  virtual StructureType alignableObjectId() const { return align::AlignableCSCStation; }

  /// Printout muon CSC Station information (not recursive)
  friend std::ostream& operator << ( std::ostream&, const AlignableCSCStation& ); 

  /// Recursive printout of the muon CSC Station structure
  void dump( void );



private:

  std::vector<AlignableCSCChamber*> theCSCChambers;


};

#endif




