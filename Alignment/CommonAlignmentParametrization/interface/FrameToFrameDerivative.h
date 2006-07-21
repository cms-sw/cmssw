#ifndef Alignment_CommonAlignmentParametrization_FrameToFrameDerivative_h
#define Alignment_CommonAlignmentParametrization_FrameToFrameDerivative_h


#include <stdlib.h>
#include <math.h>

#include "Geometry/Surface/interface/Surface.h"
#include "Geometry/CommonDetUnit/interface/GeomDet.h"
#include "DataFormats/Math/interface/Vector3D.h" 
#include "Alignment/CommonAlignment/interface/Alignable.h"
#include "Alignment/CommonAlignment/interface/AlignableDet.h"
#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"

#include "Alignment/CommonAlignmentParametrization/interface/AlignmentTransformations.h"

/// class for calculating derivatives between different frames

class FrameToFrameDerivative
{

  public:

  /// Return the derivative DeltaFrame(object)/DeltaFrame(composedobject) 
  AlgebraicMatrix frameToFrameDerivative( AlignableDet* object,
										  Alignable* composedObject );

  /// Return the derivative DeltaFrame(object)/DeltaFrame(composedobject) 
  AlgebraicMatrix frameToFrameDerivative( const GeomDet* object,
										  Alignable* composedObject );

  /// Calculates derivatives using the orientation Matrixes and the origin difference vector
  AlgebraicMatrix getDerivative( AlgebraicMatrix rotdet, AlgebraicMatrix rotrod,
								 AlgebraicVector deltapos );

  /// Gets linear approximated euler Angles 
  AlgebraicVector linearEulerAngles( AlgebraicMatrix rotDelta );
 
  /// Calculates the derivative DPos/DPos 
  AlgebraicMatrix derivativePosPos( AlgebraicMatrix RotDet,
								    AlgebraicMatrix RotRot );

  /// Calculates the derivative DPos/DRot 
  AlgebraicMatrix derivativePosRot( AlgebraicMatrix RotDet,
									AlgebraicMatrix RotRot, AlgebraicVector S );

  /// Calculates the derivative DRot/DRot 
  AlgebraicMatrix derivativeRotRot( AlgebraicMatrix RotDet,
									AlgebraicMatrix RotRot );

};

#endif

