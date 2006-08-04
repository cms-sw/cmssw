#ifndef Alignment_MuonAlignment_AlignableCSCEndcap_H
#define Alignment_MuonAlignment_AlignableCSCEndcap_H

/** \class AlignableCSCCEndcap
 *  The alignable muon CSC endcap.
 *
 *  $Date: 2006/8/4 10:00:01 $
 *  $Revision: 1.0 $
 *  \author Andre Sznajder - UERJ(Brazil)
 */


#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableComposite.h"
#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

#include "Alignment/MuonAlignment/interface/AlignableCSCStation.h"

#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"
#include "Geometry/Vector/interface/Basic3DVector.h"

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
  
  typedef GlobalPoint           PositionType;
  typedef TkRotation<float>     RotationType;

  // gets the global position as the average over all positions of the layers
  PositionType computePosition() ;
  // get the global orientation
  RotationType computeOrientation() ; //see explanation for "theOrientation"
  // get the Surface
  AlignableSurface computeSurface() ;

  AlignableCSCStation &station(int i);  
  
  /// Return alignable object identifier
  virtual int alignableObjectId() const { return AlignableObjectId::AlignableCSCEndcap; }

  /// Printout muon End Cap information (not recursive)
  friend std::ostream& operator << ( std::ostream&, const AlignableCSCEndcap& );

  /// Recursive printout of the muon End Cap structure
  void dump( void );



private:

  std::vector<AlignableCSCStation*> theCSCStations;


};

#endif




