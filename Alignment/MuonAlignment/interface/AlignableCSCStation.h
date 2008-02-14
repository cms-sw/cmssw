#ifndef Alignment_MuonAlignment_AlignableCSCStation_H
#define Alignment_MuonAlignment_AlignableCSCStation_H

/** \class AlignableCSCStation 
 *  The alignable muon CSC station.
 *
 *  $Date: 2007/12/06 01:30:52 $
 *  $Revision: 1.7.4.1 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */


#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MuonAlignment/interface/AlignableCSCRing.h"


#include <vector>

class GeomDet;
class AlignableCSCRing;

/// Concrete class for muon CSC Station alignable.
///
/// Misalignment can be de-/reactivated (forwarded to components).
///

class AlignableCSCStation : public AlignableComposite 
{

 public:

  AlignableCSCStation( const std::vector<AlignableCSCRing*> cscRings );

  ~AlignableCSCStation();
  
  virtual std::vector<Alignable*> components() const 
  {

        std::vector<Alignable*> result;
        result.insert( result.end(), theCSCRings.begin(), theCSCRings.end() );
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

  AlignableCSCRing &ring(int i);  
  
  //virtual void twist(float);

  /// Return alignable object identifier
  virtual StructureType alignableObjectId() const { return align::AlignableCSCStation; }

  /// Printout muon CSC Station information (not recursive)
  friend std::ostream& operator << ( std::ostream&, const AlignableCSCStation& ); 

  /// Recursive printout of the muon CSC Station structure
  void dump( void );



private:

  std::vector<AlignableCSCRing*> theCSCRings;


};

#endif




