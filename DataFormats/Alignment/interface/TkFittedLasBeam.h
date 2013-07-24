#ifndef DataFormats_Alignment_TkFittedLasBeam_h
#define DataFormats_Alignment_TkFittedLasBeam_h

/// \class TkFittedLasBeam ($Revision: 1.1 $)
///
/// \author Gero Flucke 
/// \date May 2009
/// last update on $Date: 2009/10/13 12:52:25 $ by $Author: flucke $
///
/// An extension of the 'TkLasBeam' containing information about 
/// a track model fit to the laser hits.
/// Documentation in TkLasTrackBasedInterface TWiki

#include "DataFormats/Alignment/interface/TkLasBeam.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include <vector>

class TkFittedLasBeam : public TkLasBeam {
 public:
  typedef float Scalar; // scalar type used in this class
  
  TkFittedLasBeam(); 
  TkFittedLasBeam(const TkLasBeam &lasBeam);
  virtual ~TkFittedLasBeam() {} // anyway virtual since inherited...
    
  /// the parametrisation type used (0 means undefined...)
  unsigned int parametrisation() const { return parametrisation_;}
  /// parallel to derivatives()
  const std::vector<Scalar>& parameters() const { return parameters_;}
  /// covariance of first n=firstFixedParameter() parameters()
  const AlgebraicSymMatrix& parametersCov() const {return paramCovariance_;}
  /// matrix of local derivatives: columns are parameters, rows are hits
  const AlgebraicMatrix& derivatives() const { return derivatives_;}
  /// index of first parameter and its derivative that was not fixed
  /// in the fit, but might be free in a global fit, e.g. within millepede
  unsigned int firstFixedParameter() const { return firstFixedParameter_;}

  /// set parameterisation (0=undefined), derivatives, chi^2 etc:
  /// - 'params' and 'derivatives' must be parallel,
  /// - 'covariance' contains the free part, i.e. has size 'firstFixedParam'
  /// - obeye firstFixedParam <= params.size()
  /// throws if inconsistent
  void setParameters(unsigned int parametrisation,
		     const std::vector<Scalar> &params,
		     const AlgebraicSymMatrix &paramCovariance,
		     const AlgebraicMatrix &derivatives,
		     unsigned int firstFixedParam, float chi2);

private:
  unsigned int parametrisation_; /// type of parameterisation (0 means undefined)
  std::vector<Scalar> parameters_; /// beam parameters (depend on parameterisation_)
  AlgebraicSymMatrix paramCovariance_; /// cov. matrix of 'free' params. (dim=firstFixedParameter_)
  AlgebraicMatrix derivatives_; /// derivatives with respect to parameters_
  unsigned int firstFixedParameter_; /// first non-free parameter in (local) fit
  float chi2_; /// chi^2 value of fit

};


// To get the typedef for the collection:
#include "DataFormats/Alignment/interface/TkFittedLasBeamCollectionFwd.h"

#endif
