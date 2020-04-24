#ifndef Alignment_TwoBodyDecay_TwoBodyDecayVirtualMeasurement_h
#define Alignment_TwoBodyDecay_TwoBodyDecayVirtualMeasurement_h

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/Math/interface/Point3D.h"
//#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

/** /class TwoBodyDecayVirtualMeasurement
 *
 *  Container-class for the virtual measurements (beam profile, mass-constraint) included
 *  into the estimation of the properties of two-body decays (see TwoBodyDecayEstimator).
 *
 *  /author Edmund Widl
 */


class TwoBodyDecayVirtualMeasurement
{

public:

  TwoBodyDecayVirtualMeasurement( const double primaryMass,
				  const double primaryWidth,
				  const double secondaryMass,
				  const reco::BeamSpot& beamSpot ) :
    thePrimaryMass( primaryMass ),
    thePrimaryWidth( primaryWidth ),
    theSecondaryMass( secondaryMass ),
    theBeamSpot( beamSpot ) {}

  TwoBodyDecayVirtualMeasurement( const TwoBodyDecayVirtualMeasurement & other ) :
    thePrimaryMass( other.thePrimaryMass ),
    thePrimaryWidth( other.thePrimaryWidth ),
    theSecondaryMass( other.theSecondaryMass ),
    theBeamSpot( other.theBeamSpot ) {}

  inline const double & primaryMass( void ) const { return thePrimaryMass; }
  inline const double & primaryWidth( void ) const { return thePrimaryWidth; }
  inline const double & secondaryMass( void ) const { return theSecondaryMass; }

  inline const reco::BeamSpot & beamSpot( void ) const { return theBeamSpot; }
  inline const AlgebraicVector beamSpotPosition( void ) const { return convertXYZPoint( theBeamSpot.position() ); }
  inline const AlgebraicSymMatrix beamSpotError( void ) const { return extractBeamSpotError(); }

private:

  inline const AlgebraicVector convertXYZPoint( const math::XYZPoint & p ) const
    { AlgebraicVector v(3); v(1)=p.x(); v(2)=p.y(); v(3)=p.z(); return v; }

  inline const AlgebraicSymMatrix extractBeamSpotError() const
    { AlgebraicSymMatrix bse(3,0); bse[0][0] = theBeamSpot.BeamWidthX(); bse[1][1] = theBeamSpot.BeamWidthY(); bse[2][2] = theBeamSpot.sigmaZ(); return bse; }

  const double thePrimaryMass;
  const double thePrimaryWidth;
  const double theSecondaryMass;
  const reco::BeamSpot & theBeamSpot;

};

#endif
