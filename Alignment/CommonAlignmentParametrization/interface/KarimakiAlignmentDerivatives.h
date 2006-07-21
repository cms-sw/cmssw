#ifndef Alignment_CommonAlignmentParametrization_KarimakiAlignmentDerivatives_h
#define Alignment_CommonAlignmentParametrization_KarimakiAlignmentDerivatives_h


#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"

/// Calculates derivatives à la Karimaki (cf. CR-2003/022)

class KarimakiAlignmentDerivatives 
{
public:
  
  /// Returns 6x2 jacobian matrix
  AlgebraicMatrix operator()(const TrajectoryStateOnSurface tsos) const;
  
};

#endif

