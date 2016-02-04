#ifndef Alignment_CommonAlignment_BowedSurfaceAlignmentParameters_h
#define Alignment_CommonAlignment_BowedSurfaceAlignmentParameters_h

#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "CondFormats/Alignment/interface/Definitions.h"

#include "Alignment/CommonAlignmentParametrization/interface/BowedSurfaceAlignmentDerivatives.h"

/// \class BowedSurfaceAlignmentParameters
///
/// Concrete class for alignment parameters and associated quantities 
/// [derived from AlignmentParameters]. 
/// The number of parameters N_PARAM is 9
///  - 3 translations
///  - 3 rotations/slopes
///  - 3 bows/sagittas (u, v and mixed term)
///
///  $Date: 2010/10/26 20:41:07 $
///  $Revision: 1.1 $
/// (last update by $Author: flucke $)

class Alignable;
class AlignableDetOrUnitPtr;
class TrajectoryStateOnSurface;

class BowedSurfaceAlignmentParameters : public AlignmentParameters 
{
public:
  /// Give parameters a name
  typedef BowedSurfaceAlignmentDerivatives BowedDerivs;
  enum AlignmentParameterName {
    dx = BowedDerivs::dx,
    dy = BowedDerivs::dy,
    dz = BowedDerivs::dz,
    dslopeX = BowedDerivs::dslopeX, // NOTE: slope(u) -> k*tan(beta),
    dslopeY = BowedDerivs::dslopeY, //       slope(v) -> l*tan(alpha)
    drotZ   = BowedDerivs::drotZ,   //       rot(w)   -> m*gamma
    dsagittaX = BowedDerivs::dsagittaX,
    dsagittaXY = BowedDerivs::dsagittaXY,
    dsagittaY = BowedDerivs::dsagittaY,
    N_PARAM = BowedDerivs::N_PARAM
  };

  /// Constructor with empty parameters/covariance
  BowedSurfaceAlignmentParameters(Alignable *alignable);

  /// Constructor for full set of parameters
  BowedSurfaceAlignmentParameters(Alignable *alignable, 
			       const AlgebraicVector &parameters, 
			       const AlgebraicSymMatrix &covMatrix);

  /// Constructor for selection 
  BowedSurfaceAlignmentParameters(Alignable *alignable, const AlgebraicVector &parameters, 
			       const AlgebraicSymMatrix &covMatrix, 
			       const std::vector<bool> &selection);

  /// Destructor 
  virtual ~BowedSurfaceAlignmentParameters() {};
  virtual void apply();
  virtual int type() const;

  /// Clone all parameters (for update of parameters)
  virtual BowedSurfaceAlignmentParameters* clone(const AlgebraicVector &parameters, 
						 const AlgebraicSymMatrix &covMatrix) const;
 
  /// Clone selected parameters (for update of parameters)
  virtual BowedSurfaceAlignmentParameters*
    cloneFromSelected(const AlgebraicVector &parameters,
		      const AlgebraicSymMatrix &covMatrix) const;
  
  /// Get all derivatives 
  virtual AlgebraicMatrix derivatives(const TrajectoryStateOnSurface &tsos,
				      const AlignableDetOrUnitPtr &aliDet) const;

  /// Get translation parameters in double precision
  align::LocalVector translation() const;

  /// Get rotation parameters 
  align::EulerAngles rotation() const;

  /// print parameters to screen 
  void print() const;
};

#endif

