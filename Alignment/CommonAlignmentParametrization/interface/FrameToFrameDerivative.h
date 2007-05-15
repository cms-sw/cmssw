#ifndef Alignment_CommonAlignmentParametrization_FrameToFrameDerivative_h
#define Alignment_CommonAlignmentParametrization_FrameToFrameDerivative_h


#include <stdlib.h>
#include <math.h>

#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/Math/interface/Vector3D.h" 
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

#include "Alignment/CommonAlignmentParametrization/interface/AlignmentTransformations.h"

/// \class FrameToFrameDerivative
///
/// class for calculating derivatives between different frames
///
///  $Date: 2007/02/12 16:14:10 $
///  $Revision: 1.3 $
/// (last update by $Author: flucke $)

class FrameToFrameDerivative
{

  public:

  /// Return the derivative DeltaFrame(object)/DeltaFrame(composedobject) 
  AlgebraicMatrix frameToFrameDerivative(const Alignable* object,
					 const Alignable* composedObject) const;

  /// Return the derivative DeltaFrame(object)/DeltaFrame(composedobject) 
  AlgebraicMatrix frameToFrameDerivative(const GeomDet* object,
					 const Alignable* composedObject) const;

  /// Calculates derivatives using the orientation Matrixes and the origin difference vector
  AlgebraicMatrix getDerivative(const AlgebraicMatrix &rotdet, const AlgebraicMatrix &rotrod,
				const AlgebraicVector &deltapos) const;

  /// Gets linear approximated euler Angles 
  AlgebraicVector linearEulerAngles(const AlgebraicMatrix &rotDelta) const;
 
  /// Calculates the derivative DPos/DPos 
  AlgebraicMatrix derivativePosPos(const AlgebraicMatrix &RotDet,
				   const AlgebraicMatrix &RotRot) const;

  /// Calculates the derivative DPos/DRot 
  AlgebraicMatrix derivativePosRot(const AlgebraicMatrix &RotDet, const AlgebraicMatrix &RotRot,
				   const AlgebraicVector &S) const;

  /// Calculates the derivative DRot/DRot 
  AlgebraicMatrix derivativeRotRot(const AlgebraicMatrix &RotDet,
				   const AlgebraicMatrix &RotRot) const;

};

#endif

