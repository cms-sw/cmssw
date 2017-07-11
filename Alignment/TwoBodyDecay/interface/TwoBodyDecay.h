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

  TwoBodyDecay( ) :
    theDecayParameters(), theChi2( 0. ), theValidityFlag( false ),
    thePrimaryMass( 0. ), thePrimaryWidth( 0. )  {}

  TwoBodyDecay( const TwoBodyDecayParameters &param, double chi2, bool valid, 
		const TwoBodyDecayVirtualMeasurement &vm ) :
    theDecayParameters( param ), theChi2( chi2 ), theValidityFlag( valid ),
    thePrimaryMass( vm.primaryMass() ), thePrimaryWidth( vm.primaryWidth() ) {}

  ~TwoBodyDecay( ) {}

  inline const TwoBodyDecayParameters & decayParameters( ) const { return theDecayParameters; }

  inline const AlgebraicVector & parameters( ) const { return theDecayParameters.parameters(); }
  inline const AlgebraicSymMatrix & covariance( ) const { return theDecayParameters.covariance(); }

  /// Get specified decay parameter.
  inline double operator[]( ParameterName name ) const { return theDecayParameters[name]; }

  /// Get specified decay parameter.
  inline double operator()( ParameterName name ) const { return theDecayParameters(name); }

  inline bool hasError( ) const { return theDecayParameters.hasError(); }

  inline double chi2( ) const { return theChi2; }

  inline bool isValid( ) const { return theValidityFlag; }
  inline void setInvalid( ) { theValidityFlag = false; }

  inline double primaryMass( ) const { return thePrimaryMass; }
  inline double primaryWidth( ) const { return thePrimaryWidth; }

 private:

  TwoBodyDecayParameters theDecayParameters;
  double theChi2;
  bool theValidityFlag;
  double thePrimaryMass;
  double thePrimaryWidth;

};

#endif
