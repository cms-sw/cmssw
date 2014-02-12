#ifndef Alignment_CommonAlignmentParametrization_FrameToFrameDerivative_h
#define Alignment_CommonAlignmentParametrization_FrameToFrameDerivative_h

#include "CondFormats/Alignment/interface/Definitions.h"
#include "DataFormats/Math/interface/AlgebraicROOTObjects.h"

/// \class FrameToFrameDerivative
///
/// Class for calculating the jacobian d_object/d_composedObject
/// for the rigid body parametrisation of both, i.e. the derivatives
/// expressing the influence of u, v, w, alpha, beta, gamma of the
/// composedObject on u, v, w, alpha, beta, gamma of its component 'object'.
///
///  $Date: 2007/10/08 15:56:00 $
///  $Revision: 1.6 $
/// (last update by $Author: cklae $)

class Alignable;

class FrameToFrameDerivative
{
  public:

  /// Return the derivative DeltaFrame(object)/DeltaFrame(composedObject),
  /// i.e. a 6x6 matrix:
  ///
  /// / du/du_c du/dv_c du/dw_c du/da_c du/db_c du/dg_c |
  /// | dv/du_c dv/dv_c dv/dw_c dv/da_c dv/db_c dv/dg_c |
  /// | dw/du_c dw/dv_c dw/dw_c dw/da_c dw/db_c dw/dg_c |
  /// | da/du_c da/dv_c da/dw_c da/da_c da/db_c da/dg_c |
  /// | db/du_c db/dv_c db/dw_c db/da_c db/db_c db/dg_c |
  /// \ dg/du_c dg/dv_c dg/dw_c dg/da_c dg/db_c dg/dg_c /
  ///
  /// where u, v, w, a, b, g are shifts and rotations of the object
  /// and u_c, v_c, w_c, a_c, b_c, g_c those of the composed object.

  AlgebraicMatrix frameToFrameDerivative(const Alignable* object,
					 const Alignable* composedObject) const;

  /// Calculates derivatives DeltaFrame(object)/DeltaFrame(composedobject) 
  /// using their positions and orientations, see method frameToFrameDerivative(..)
  /// for definition.
  /// As a new method it gets a new interface avoiding CLHEP that should anyway be
  /// replaced by SMatrix at some point...
  AlgebraicMatrix66 getDerivative(const align::RotationType &objectRot,
				  const align::RotationType &composeRot,
				  const align::GlobalPoint &objectPos,
				  const align::GlobalPoint &composePos) const;

  private:
  /// Helper to transform from RotationType to AlgebraicMatrix
  inline static AlgebraicMatrix transform(const align::RotationType&);

  /// Calculates derivatives using the orientation Matrixes and the origin difference vector
  AlgebraicMatrix getDerivative(const align::RotationType &objectRot,
				const align::RotationType &composeRot,
				const align::GlobalVector &posVec) const;
  
  /// Gets linear approximated euler Angles 
  AlgebraicVector linearEulerAngles(const AlgebraicMatrix &rotDelta) const;
 
  /// Calculates the derivative DPos/DPos 
  AlgebraicMatrix derivativePosPos(const AlgebraicMatrix &RotDet,
				   const AlgebraicMatrix &RotRot) const;

  /// Calculates the derivative DPos/DRot 
  AlgebraicMatrix derivativePosRot(const AlgebraicMatrix &RotDet,
				   const AlgebraicMatrix &RotRot,
				   const AlgebraicVector &S) const;

  /// Calculates the derivative DRot/DRot 
  AlgebraicMatrix derivativeRotRot(const AlgebraicMatrix &RotDet,
				   const AlgebraicMatrix &RotRot) const;

};

AlgebraicMatrix FrameToFrameDerivative::transform(const align::RotationType& rot)
{
  AlgebraicMatrix R(3, 3);

  R(1, 1) = rot.xx(); R(1, 2) = rot.xy(); R(1, 3) = rot.xz();
  R(2, 1) = rot.yx(); R(2, 2) = rot.yy(); R(2, 3) = rot.yz();
  R(3, 1) = rot.zx(); R(3, 2) = rot.zy(); R(3, 3) = rot.zz();

  return R;
}

#endif

