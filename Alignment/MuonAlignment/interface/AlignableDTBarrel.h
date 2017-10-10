#ifndef Alignment_MuonAlignment_AlignableDTBarrel_H
#define Alignment_MuonAlignment_AlignableDTBarrel_H

/** \class AlignableDTBarrel
 *  The alignable muon DT barrel.
 *
 *  $Date: 2008/04/15 16:05:53 $
 *  $Revision: 1.10 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */


#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MuonAlignment/interface/AlignableDTWheel.h"



#include <vector>

class GeomDet;

/// Concrete class for muon DT Barrel alignable.
///
/// Misalignment can be de-/reactivated (forwarded to components).
///

class AlignableDTBarrel : public AlignableComposite 
{

 public:

  AlignableDTBarrel( const std::vector<AlignableDTWheel*>& dtWheels );

  ~AlignableDTBarrel() override;
  
  std::vector<Alignable*> components() const override 
  {

        std::vector<Alignable*> result;
        result.insert( result.end(), theDTWheels.begin(), theDTWheels.end() );
        return result;

  }
  
  // gets the global position as the average over all positions of the layers
  PositionType computePosition() ;
  // get the global orientation
  RotationType computeOrientation() ; //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface() ;

  AlignableDTWheel &wheel(int i);  
  
  /// Printout muon Barrel information (not recursive)
  friend std::ostream& operator << ( std::ostream&, const AlignableDTBarrel& );

  /// Recursive printout of the muon Barrel structure
  void dump( void ) const override;


  // Get alignments sorted by DetId
  Alignments* alignments() const override;

  // Get alignment errors sorted by DetId
  AlignmentErrorsExtended* alignmentErrors() const override;



private:

  std::vector<AlignableDTWheel*> theDTWheels;


};

#endif




