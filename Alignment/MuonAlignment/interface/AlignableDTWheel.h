#ifndef Alignment_MuonAlignment_AlignableDTWheel_H
#define Alignment_MuonAlignment_AlignableDTWheel_H

/** \class AlignableDTWheel
 *  The alignable muon DT wheel.
 *
 *  $Date: 2011/09/15 10:07:07 $
 *  $Revision: 1.11 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */


#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MuonAlignment/interface/AlignableDTStation.h"


#include <vector>

class GeomDet;

/// Concrete class for muon DT Wheel alignable.
///
/// Misalignment can be de-/reactivated (forwarded to components).
///

class AlignableDTWheel : public AlignableComposite 
{

 public:

  AlignableDTWheel( const std::vector<AlignableDTStation*> dtStations );

  ~AlignableDTWheel();
  
  virtual std::vector<Alignable*> components() const 
  {

        std::vector<Alignable*> result;
        result.insert( result.end(), theDTStations.begin(), theDTStations.end() );
        return result;

  }
  
  // gets the global position as the average over all positions of the layers
  PositionType computePosition() ;
  // get the global orientation
  RotationType computeOrientation() ; //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface() ;

  AlignableDTStation &station(int i);  
  
  /// Printout muon DT wheel information (not recursive)
  friend std::ostream& operator << ( std::ostream&, const AlignableDTWheel& );

  /// Recursive printout of the muon DT wheel structure
  void dump( void ) const;


private:

  std::vector<AlignableDTStation*> theDTStations;


};

#endif




