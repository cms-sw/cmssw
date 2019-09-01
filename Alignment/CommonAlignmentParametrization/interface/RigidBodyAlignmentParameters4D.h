#ifndef Alignment_CommonAlignment_RigidBodyAlignmentParameters4D_h
#define Alignment_CommonAlignment_RigidBodyAlignmentParameters4D_h

//#include "Alignment/CommonAlignment/interface/AlignmentParameters.h"
#include "Alignment/CommonAlignmentParametrization/interface/RigidBodyAlignmentParameters.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/// \class RigidBodyAlignmentParameters
///
/// Concrete class for alignment parameters and associated quantities
/// [derived from AlignmentParameters]. The number of parameters
/// N_PARAM is fixed to 6 (3 translations + 3 rotations)
///
///  $Date: 2008/09/02 15:08:12 $
///  $Revision: 1.13 $
/// (last update by $Author: flucke $)

class Alignable;
class AlignableDetOrUnitPtr;
class TrajectoryStateOnSurface;

class RigidBodyAlignmentParameters4D : public RigidBodyAlignmentParameters {
public:
  /// Constructor with empty parameters/covariance (if calcMis = false) or with
  /// parameters (no covariance) created from current (mis-)placement of
  /// alignable (if calcMis = true).
  RigidBodyAlignmentParameters4D(Alignable *alignable, bool calcMis)
      : RigidBodyAlignmentParameters(alignable, calcMis){};

  /// Constructor for full set of parameters
  RigidBodyAlignmentParameters4D(Alignable *alignable,
                                 const AlgebraicVector &parameters,
                                 const AlgebraicSymMatrix &covMatrix)
      : RigidBodyAlignmentParameters(alignable, parameters, covMatrix){};

  /// Constructor for selection
  RigidBodyAlignmentParameters4D(Alignable *alignable,
                                 const AlgebraicVector &parameters,
                                 const AlgebraicSymMatrix &covMatrix,
                                 const std::vector<bool> &selection)
      : RigidBodyAlignmentParameters(alignable, parameters, covMatrix, selection){};

  /// Destructor
  ~RigidBodyAlignmentParameters4D() override{};

  int type() const override;

  /// Get all derivatives
  AlgebraicMatrix derivatives(const TrajectoryStateOnSurface &tsos, const AlignableDetOrUnitPtr &) const override;

  /// Clone all parameters (for update of parameters)
  RigidBodyAlignmentParameters4D *clone(const AlgebraicVector &parameters,
                                        const AlgebraicSymMatrix &covMatrix) const override;

  /// Clone selected parameters (for update of parameters)
  RigidBodyAlignmentParameters4D *cloneFromSelected(const AlgebraicVector &parameters,
                                                    const AlgebraicSymMatrix &covMatrix) const override;
};

#endif
