#ifndef Alignment_CommonAlignmentParametrization_SegmentAlignmentDerivatives4D_h
#define Alignment_CommonAlignmentParametrization_SegmentAlignmentDerivatives4D_h

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/// \class SegmentAlignmentDerivatives4D
///
/// Calculates derivatives for 4D Segments
///
///  $Date: 2008/12/12 15:58:07 $
///  $Revision: 1.1 $
/// (last update by $Author: pablom $)

class TrajectoryStateOnSurface;

class SegmentAlignmentDerivatives4D 
{
public:
  
  /// Returns 6x4 jacobian matrix
  AlgebraicMatrix operator()(const TrajectoryStateOnSurface &tsos) const;
  
};

#endif

