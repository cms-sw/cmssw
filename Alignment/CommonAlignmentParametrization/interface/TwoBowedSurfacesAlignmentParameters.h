#ifndef Alignment_CommonAlignment_TwoBowedSurfacesAlignmentParameters_h
#define Alignment_CommonAlignment_TwoBowedSurfacesAlignmentParameters_h

#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"
#include "CondFormats/Alignment/interface/Definitions.h"

#include "Alignment/CommonAlignmentParametrization/interface/BowedSurfaceAlignmentDerivatives.h"

/// \class TwoBowedSurfacesAlignmentParameters
///
/// Concrete class for alignment parameters and associated quantities 
/// [derived from AlignmentParameters]. 
/// The alignable is assumed to have in fact two surfaces devided at 
/// a specific local ySplit.
/// The number of parameters N_PARAM is 2x9, i.e. one set of
///  - 3 translations
///  - 3 rotations/slopes
///  - 3 bows/sagittas (u, v and mixed term)
/// for the two surfaces (similar as in BowedSurfaceAlignmentParameters).
/// Storage is done differently:
/// The movements and translations are averaged and applied to the 
/// alignable in the 'classical' way.
/// In SurfaceDeformation we store:
/// - the half differences of the movements and rotations, to be taken into account,
///   with positive/negative sign, for corrections for the surface at lower/higher y,
/// - the mean values 'm' and half differences 'hd' of the bows/sagittas of each surface,
///   where these corrections have to be applied as m+hd for the one and m-hd
///   for the other sensor
///
///
///  $Date: 2010/12/14 01:03:27 $
///  $Revision: 1.2 $
/// (last update by $Author: flucke $)

class Alignable;
class AlignableDetOrUnitPtr;
class TrajectoryStateOnSurface;

class TwoBowedSurfacesAlignmentParameters : public AlignmentParameters 
{
 public:
  /// Give parameters a name (do not change order, see derivatives(..)!)
  typedef BowedSurfaceAlignmentDerivatives BowedDerivs;
  enum AlignmentParameterName {
    // 1st surface
    dx1 = BowedDerivs::dx,
    dy1 = BowedDerivs::dy,
    dz1 = BowedDerivs::dz,
    dslopeX1 = BowedDerivs::dslopeX, // NOTE: slope(u) -> halfWidth*tan(beta),
    dslopeY1 = BowedDerivs::dslopeY, //       slope(v) -> halfLength*tan(alpha)
    drotZ1   = BowedDerivs::drotZ,   //       rot(w)   -> g-scale*gamma
    dsagittaX1  = BowedDerivs::dsagittaX,
    dsagittaXY1 = BowedDerivs::dsagittaXY,
    dsagittaY1  = BowedDerivs::dsagittaY,
    // 2nd surface
    dx2 = BowedDerivs::dx + BowedDerivs::N_PARAM,
    dy2 = BowedDerivs::dy + BowedDerivs::N_PARAM,
    dz2 = BowedDerivs::dz + BowedDerivs::N_PARAM,
    dslopeX2 = BowedDerivs::dslopeX + BowedDerivs::N_PARAM, // NOTE: slope(u) -> k*tan(beta),
    dslopeY2 = BowedDerivs::dslopeY + BowedDerivs::N_PARAM, //       slope(v) -> k*tan(alpha)
    drotZ2   = BowedDerivs::drotZ   + BowedDerivs::N_PARAM, //       rot(w)   -> m*gamma
    dsagittaX2  = BowedDerivs::dsagittaX  + BowedDerivs::N_PARAM,
    dsagittaXY2 = BowedDerivs::dsagittaXY + BowedDerivs::N_PARAM,
    dsagittaY2  = BowedDerivs::dsagittaY  + BowedDerivs::N_PARAM,
    // number of parameters
    N_PARAM = BowedDerivs::N_PARAM + BowedDerivs::N_PARAM
  };

  /// Constructor with empty parameters/covariance
  TwoBowedSurfacesAlignmentParameters(Alignable *alignable);

  /// Constructor for full set of parameters
  TwoBowedSurfacesAlignmentParameters(Alignable *alignable, 
			       const AlgebraicVector &parameters, 
			       const AlgebraicSymMatrix &covMatrix);

  /// Constructor for selection 
  TwoBowedSurfacesAlignmentParameters(Alignable *alignable, const AlgebraicVector &parameters, 
			       const AlgebraicSymMatrix &covMatrix, 
			       const std::vector<bool> &selection);

  /// Destructor 
  virtual ~TwoBowedSurfacesAlignmentParameters() {};
  virtual void apply();
  virtual int type() const;

  /// Clone all parameters (for update of parameters)
  virtual TwoBowedSurfacesAlignmentParameters* clone(const AlgebraicVector &parameters, 
						     const AlgebraicSymMatrix &covMatrix) const;
 
  /// Clone selected parameters (for update of parameters)
  virtual TwoBowedSurfacesAlignmentParameters*
    cloneFromSelected(const AlgebraicVector &parameters,
		      const AlgebraicSymMatrix &covMatrix) const;
  
  /// Get all derivatives 
  virtual AlgebraicMatrix derivatives(const TrajectoryStateOnSurface &tsos,
				      const AlignableDetOrUnitPtr &aliDet) const;

  /// print parameters to screen 
  virtual void print() const;

  double ySplit() const { return ySplit_;}

 private:
  double ySplitFromAlignable(const Alignable *ali) const;

  double ySplit_;
};

#endif

