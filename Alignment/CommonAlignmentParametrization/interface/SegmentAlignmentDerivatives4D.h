#ifndef Alignment_CommonAlignmentParametrization_SegmentAlignmentDerivatives4D_h
#define Alignment_CommonAlignmentParametrization_SegmentAlignmentDerivatives4D_h

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/// \class SegmentAlignmentDerivatives4D
///
/// Calculates derivatives for 4D Segments
///
///  $Date: 2007/03/12 21:28:48 $
///  $Revision: 1.4 $
/// (last update by $Author: P. Martinez $)

class TrajectoryStateOnSurface;

class SegmentAlignmentDerivatives4D {
public:
  /// Returns 6x4 jacobian matrix
  AlgebraicMatrix operator()(const TrajectoryStateOnSurface &tsos) const;
};

#endif
