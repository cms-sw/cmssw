#ifndef Alignment_CommonAlignmentParametrization_BeamSpotAlignmentDerivatives_h
#define Alignment_CommonAlignmentParametrization_BeamSpotAlignmentDerivatives_h

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/// \class BeamSpotAlignmentDerivatives
///
/// Calculates derivatives for the alignable beam spot
///
///  $Date: 2010/09/10 11:18:29 $
///  $Revision: 1.1 $
/// (last update by $Author: mussgill $)

class TrajectoryStateOnSurface;

class BeamSpotAlignmentDerivatives
{
public:
  
  /// Returns 4x2 jacobian matrix
  AlgebraicMatrix operator()(const TrajectoryStateOnSurface &tsos) const;
  
};

#endif

