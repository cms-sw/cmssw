#ifndef Alignment_CommonAlignmentParametrization_KarimakiAlignmentDerivatives_h
#define Alignment_CommonAlignmentParametrization_KarimakiAlignmentDerivatives_h

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/// \class KarimakiAlignmentDerivatives
///
/// Calculates derivatives à la Karimaki (cf. CR-2003/022)
///
///  $Date: 2007/03/02 12:17:09 $
///  $Revision: 1.3 $
/// (last update by $Author: fronga $)

class TrajectoryStateOnSurface;

class KarimakiAlignmentDerivatives 
{
public:
  
  /// Returns 6x2 jacobian matrix of derivatives of residuals in x and y
  /// with respect to rigid body aligment parameters:
  ///
  /// / dr_x/du  dr_y/du |
  /// | dr_x/dv  dr_y/dv |
  /// | dr_x/dw  dr_y/dw |
  /// | dr_x/da  dr_y/da |
  /// | dr_x/db  dr_y/db |
  /// \ dr_x/dg  dr_y/dg /
  AlgebraicMatrix operator()(const TrajectoryStateOnSurface &tsos) const;
  
};

#endif

