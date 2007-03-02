#ifndef Alignment_CommonAlignmentParametrization_KarimakiAlignmentDerivatives_h
#define Alignment_CommonAlignmentParametrization_KarimakiAlignmentDerivatives_h


#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/// \class KarimakiAlignmentDerivatives
///
/// Calculates derivatives à la Karimaki (cf. CR-2003/022)
///
///  $Date: 2006/10/19 14:20:59 $
///  $Revision: 1.2 $
/// (last update by $Author: flucke $)

class KarimakiAlignmentDerivatives 
{
public:
  
  /// Returns 6x2 jacobian matrix
  AlgebraicMatrix operator()(const TrajectoryStateOnSurface &tsos) const;
  
};

#endif

