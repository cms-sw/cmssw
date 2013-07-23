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
  
  /// Returns 6x2 jacobian matrix
  AlgebraicMatrix operator()(const TrajectoryStateOnSurface &tsos) const;
  
};

#endif

