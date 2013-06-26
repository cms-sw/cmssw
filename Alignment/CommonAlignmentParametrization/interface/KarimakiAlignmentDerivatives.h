#ifndef Alignment_CommonAlignmentParametrization_KarimakiAlignmentDerivatives_h
#define Alignment_CommonAlignmentParametrization_KarimakiAlignmentDerivatives_h

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/// \class KarimakiAlignmentDerivatives
///
/// Calculates derivatives à la Karimaki (cf. CR-2003/022)
///
///  $Date: 2007/03/12 21:28:48 $
///  $Revision: 1.4 $
/// (last update by $Author: cklae $)

class TrajectoryStateOnSurface;

class KarimakiAlignmentDerivatives 
{
public:
  
  /// Returns 6x2 jacobian matrix
  AlgebraicMatrix operator()(const TrajectoryStateOnSurface &tsos) const;
  
};

#endif

