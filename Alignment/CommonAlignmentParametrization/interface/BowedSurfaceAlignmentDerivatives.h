#ifndef Alignment_CommonAlignmentParametrization_BowedSurfaceAlignmentDerivatives_h
#define Alignment_CommonAlignmentParametrization_BowedSurfaceAlignmentDerivatives_h

#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

/// \class BowedSurfaceAlignmentDerivatives
///
/// Calculates alignment derivatives for a bowed surface using Legendre
/// polynomials for the surface structure (as studied by Claus Kleinwort),
/// i.e.
/// - rigid body part (partially from KarimakiAlignmentDerivatives)
/// - bow in local u, v and mixed term.
///
/// If a surface is split into two parts at a given ySplit value,
/// rotation axes are re-centred to that part hit by the track
/// (as predicted by TSOS) and the length of the surface is re-scaled.
///
///  by Gero Flucke, October 2010
///  $Date: 2010/10/26 20:41:07 $
///  $Revision: 1.1 $
/// (last update by $Author: flucke $)

class TrajectoryStateOnSurface;

class BowedSurfaceAlignmentDerivatives 
{
public:
  
  enum AlignmentParameterName {
    dx = 0, dy, dz,
    dslopeX, // NOTE: slope(u) -> k*tan(beta), 
    dslopeY, //       slope(v) -> k*tan(alpha)
    drotZ,   // rotation around w axis, scaled by gammaScale
    dsagittaX, dsagittaXY, dsagittaY,
    N_PARAM
  };

  /// Returns 9x2 jacobian matrix
  AlgebraicMatrix operator()(const TrajectoryStateOnSurface &tsos,
			     double uWidth, double vLength,
			     bool doSplit = false, double ySplit = 0.) const;

  /// scale to apply to convert drotZ to karimaki-gamma,
  /// depending on module width and length (the latter after splitting!)  
  static double gammaScale(double width, double splitLength);
};

#endif
