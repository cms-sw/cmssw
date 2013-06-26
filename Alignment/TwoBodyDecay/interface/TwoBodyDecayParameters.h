#ifndef Alignment_TwoBodyDecay_TwoBodyDecayParameters_h
#define Alignment_TwoBodyDecay_TwoBodyDecayParameters_h

/** /class TwoBodyDecayParameters
 *
 *  This class provides the definition and a container for the parameters
 *  describing a two-body decay.
 *
 *  /author Edmund Widl
 */

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"


class TwoBodyDecayParameters
{

public:

  /// Define order of parameters
  enum ParameterName { x = 0, y = 1, z = 2, px = 3, py = 4, pz = 5, theta = 6, phi = 7, mass = 8 };

  enum { dimension = 9 };

  TwoBodyDecayParameters( void ) :
    theParameters( AlgebraicVector() ), theCovariance( AlgebraicSymMatrix() ) {}

  TwoBodyDecayParameters( const AlgebraicVector & param, const AlgebraicSymMatrix & cov ) :
    theParameters( param ), theCovariance( cov ) {}

  TwoBodyDecayParameters( AlgebraicVector param ) :
    theParameters( param ), theCovariance( AlgebraicSymMatrix() ) {}

  TwoBodyDecayParameters( const TwoBodyDecayParameters & other ) :
    theParameters( other.parameters() ), theCovariance( other.covariance() ) {}

  ~TwoBodyDecayParameters( void ) {}

  /// Get decay parameters.
  inline const AlgebraicVector & parameters( void ) const { return theParameters; }

  /// Get error matrix.
  inline const AlgebraicSymMatrix & covariance( void ) const { return theCovariance; }

  /// Get specified decay parameter.
  inline double operator[]( ParameterName name ) const { return theParameters[name]; }

  /// Get specified decay parameter.
  inline double operator()( ParameterName name ) const { return theParameters[name]; }

  /// Get specified range of decay parameters.
  inline const AlgebraicVector sub( ParameterName first, ParameterName last ) const { return theParameters.sub( first+1, last+1 ); }

  inline bool hasError( void ) const { return ( theCovariance.num_row() != 0 ); }

private:

  AlgebraicVector theParameters;
  AlgebraicSymMatrix theCovariance;

};

#endif
