#ifndef Alignment_CommonAlignmentParametrization_AlignmentTranformations_h
#define Alignment_CommonAlignmentParametrization_AlignmentTranformations_h


#include <stdlib.h>
#include <math.h>

#include "Geometry/Surface/interface/Surface.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "Geometry/CommonDetAlgo/interface/AlgebraicObjects.h"

/// Helper class for Alignment Transformations

class AlignmentTransformations
{

public:

  /// converts Surface::RotationType to AlgebraicMatrix 
  AlgebraicMatrix algebraicMatrix( Surface::RotationType rot ) const;

  /// converts Surface::RotationType to AlgebraicVector 
  AlgebraicVector algebraicVector( Surface::RotationType rot ) const;

  /// converts AlgebraicMatrix to Surface::RotationType 
  Surface::RotationType rotationType( AlgebraicMatrix algM ) const;

  /// converts GlobalVector to AlgebraicVector 
  AlgebraicVector algebraicVector( GlobalVector globalVector ) const;

  /// gets Euler Angles from RotationType 
  AlgebraicVector eulerAngles( Surface::RotationType rot, int flag ) const;

  /// transforms Rotation to local Det Frame 
  Surface::RotationType globalToLocalMatrix( Surface::RotationType rot,
											 Surface::RotationType detrot ) const;
 
  /// transforms Rotation to global Frame 
  Surface::RotationType localToGlobalMatrix( Surface::RotationType aliDetRot,
											 Surface::RotationType detRot ) const;

  /// builds rotation matrix from Euler Angles 
  AlgebraicMatrix rotMatrix3( AlgebraicVector a ) const;

  /// transforms euler angles to local frame 
  AlgebraicVector globalToLocalEulerAngles( AlgebraicVector a, 
											Surface::RotationType rot ) const;

  /// transforms euler angles to global frame 
  AlgebraicVector localToGlobalEulerAngles( AlgebraicVector a, 
											Surface::RotationType rot ) const;

  /// repair rotation matrix for rounding errors 
  Surface::RotationType rectify( Surface::RotationType rot ) const;

};

#endif
