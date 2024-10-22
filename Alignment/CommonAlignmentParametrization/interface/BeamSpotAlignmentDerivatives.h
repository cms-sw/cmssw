#ifndef Alignment_CommonAlignmentParametrization_BeamSpotAlignmentDerivatives_h
#define Alignment_CommonAlignmentParametrization_BeamSpotAlignmentDerivatives_h

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/// \class BeamSpotAlignmentDerivatives
///
/// Calculates derivatives for the alignable beam spot
///
///  $Date: 2007/03/12 21:28:48 $
///  $Revision: 1.4 $
/// (last update by $Author: cklae $)

class TrajectoryStateOnSurface;

class BeamSpotAlignmentDerivatives {
public:
  /// Returns 4x2 jacobian matrix
  AlgebraicMatrix operator()(const TrajectoryStateOnSurface &tsos) const;
};

#endif
