#ifndef Alignment_CommonAlignmentParametrization_KarimakiAlignmentDerivatives_h
#define Alignment_CommonAlignmentParametrization_KarimakiAlignmentDerivatives_h


#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"

/// \class KarimakiAlignmentDerivatives
///
/// Calculates derivatives à la Karimaki (cf. CR-2003/022)
///
///  $Date: 2006/10/17 11:02:42 $
///  $Revision: 1.11 $
/// (last update by $Author: flucke $)

class KarimakiAlignmentDerivatives 
{
public:
  
  /// Returns 6x2 jacobian matrix
  AlgebraicMatrix operator()(const TrajectoryStateOnSurface &tsos) const;
  
};

#endif

