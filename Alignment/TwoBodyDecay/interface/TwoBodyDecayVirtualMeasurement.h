#ifndef Alignment_TwoBodyDecay_TwoBodyDecayVirtualMeasurement_h
#define Alignment_TwoBodyDecay_TwoBodyDecayVirtualMeasurement_h

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "DataFormats/GeometryCommonDetAlgo/interface/GlobalError.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"

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
				  const AlgebraicVector& beamSpot,
				  const AlgebraicSymMatrix& beamSpotError ) :
    thePrimaryMass( primaryMass ),
    thePrimaryWidth( primaryWidth ),
    theSecondaryMass( secondaryMass ),
    theBeamSpot( beamSpot ),
    theBeamSpotError( beamSpotError ) {}

  TwoBodyDecayVirtualMeasurement( const double primaryMass,
				  const double primaryWidth,
				  const double secondaryMass,
				  const GlobalPoint& beamSpot,
				  const GlobalError& beamSpotError ) :
    thePrimaryMass( primaryMass ),
    thePrimaryWidth( primaryWidth ),
    theSecondaryMass( secondaryMass ),
    theBeamSpot( convertGlobalPoint( beamSpot ) ),
    theBeamSpotError( beamSpotError.matrix() ) {}

  TwoBodyDecayVirtualMeasurement( void ) :
    thePrimaryMass( 0. ),
    thePrimaryWidth( 0. ),
    theSecondaryMass( 0. ),
    theBeamSpot( AlgebraicVector() ),
    theBeamSpotError( AlgebraicSymMatrix() ) {}

  inline const double primaryMass( void ) const { return thePrimaryMass; }
  inline const double primaryWidth( void ) const { return thePrimaryWidth; }
  inline const double secondaryMass( void ) const { return theSecondaryMass; }

  inline const AlgebraicVector& beamSpot( void ) const { return theBeamSpot; }
  inline const AlgebraicSymMatrix& beamSpotError( void ) const { return theBeamSpotError; }

private:

  inline const AlgebraicVector convertGlobalPoint( const GlobalPoint & gp ) const
    { AlgebraicVector v(3); v(1)=gp.x(); v(2)=gp.y(); v(3)=gp.z(); return v; }

  double thePrimaryMass;
  double thePrimaryWidth;
  double theSecondaryMass;

  AlgebraicVector theBeamSpot;
  AlgebraicSymMatrix theBeamSpotError;

};

#endif
