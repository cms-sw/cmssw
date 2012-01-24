#ifndef Alignment_TwoBodyDecay_TwoBodyDecay_h
#define Alignment_TwoBodyDecay_TwoBodyDecay_h

#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayParameters.h"
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayVirtualMeasurement.h"

/** /class TwoBodyDecay
 *
 *  Container-class for all information associated with a two-body decay (estimated parameters,
 *  chi2 of the fit, validity-flag).
 *
 *  /author Edmund Widl
 */


class TwoBodyDecay
{

public:

  typedef TwoBodyDecayParameters::ParameterName ParameterName;

  TwoBodyDecay( void ) :
    theDecayParameters(), theChi2( 0. ), theValidityFlag( false ) {}

  TwoBodyDecay( const TwoBodyDecayParameters &param, double chi2, bool valid, 
		const TwoBodyDecayVirtualMeasurement &vm ) :
    theDecayParameters( param ), theChi2( chi2 ), theValidityFlag( valid ),
    thePrimaryMass( vm.primaryMass() ), thePrimaryWidth( vm.primaryWidth() ) {}

  ~TwoBodyDecay( void ) {}

  inline const TwoBodyDecayParameters & decayParameters( void ) const { return theDecayParameters; }

  inline const AlgebraicVector & parameters( void ) const { return theDecayParameters.parameters(); }
  inline const AlgebraicSymMatrix & covariance( void ) const { return theDecayParameters.covariance(); }

  /// Get specified decay parameter.
  inline const double operator[]( ParameterName name ) const { return theDecayParameters[name]; }

  /// Get specified decay parameter.
  inline const double operator()( ParameterName name ) const { return theDecayParameters(name); }

  inline const bool hasError( void ) const { return theDecayParameters.hasError(); }

  inline const double chi2( void ) const { return theChi2; }

  inline const bool isValid( void ) const { return theValidityFlag; }
  inline void setInvalid( void ) { theValidityFlag = false; }

  inline const double primaryMass( void ) const { return thePrimaryMass; }
  inline const double primaryWidth( void ) const { return thePrimaryWidth; }

 private:

  TwoBodyDecayParameters theDecayParameters;
  double theChi2;
  bool theValidityFlag;
  double thePrimaryMass;
  double thePrimaryWidth;

};

#endif
