#ifndef Alignment_MuonAlignment_AlignableCSCEndcap_H
#define Alignment_MuonAlignment_AlignableCSCEndcap_H

/** \class AlignableCSCCEndcap
 *  The alignable muon CSC endcap.
 *
 *  $Date: 2011/09/15 09:40:22 $
 *  $Revision: 1.11 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */


#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MuonAlignment/interface/AlignableCSCStation.h"



#include <vector>

class GeomDet;

/// Concrete class for muon CSC Endcap alignable.
///
/// Misalignment can be de-/reactivated (forwarded to components).
///

class AlignableCSCEndcap : public AlignableComposite 
{

 public:

  AlignableCSCEndcap( const std::vector<AlignableCSCStation*> cscStations );

  ~AlignableCSCEndcap();
  
  virtual std::vector<Alignable*> components() const 
  {

        std::vector<Alignable*> result;
        result.insert( result.end(), theCSCStations.begin(), theCSCStations.end() );
        return result;

  }
  
  // gets the global position as the average over all positions of the layers
  PositionType computePosition() ;
  // get the global orientation
  RotationType computeOrientation() ; //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface() ;

  AlignableCSCStation &station(int i);  
  
  /// Printout muon End Cap information (not recursive)
  friend std::ostream& operator << ( std::ostream&, const AlignableCSCEndcap& );

  /// Recursive printout of the muon End Cap structure
  void dump( void ) const;

  // Get alignments sorted by DetId
  Alignments* alignments() const;

  // Get alignment errors sorted by DetId
  AlignmentErrors* alignmentErrors() const;



private:

  std::vector<AlignableCSCStation*> theCSCStations;


};

#endif




