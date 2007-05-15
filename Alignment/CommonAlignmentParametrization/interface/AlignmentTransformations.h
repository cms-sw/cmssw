#ifndef Alignment_CommonAlignmentParametrization_AlignmentTranformations_h
#define Alignment_CommonAlignmentParametrization_AlignmentTranformations_h

/// \class AlignmentTransformations
///
/// Helper class for Alignment Transformations:
///  * between different matrix/vector implementations
///  * Euler angles and rotation matrices
///  * between local and global frame
///
///  $Date: 2007/01/25 09:18:30 $
///  $Revision: 1.2 $
///  $Author: flucke $ did last update.


#include "DataFormats/GeometrySurface/interface/Surface.h"
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"


class AlignmentTransformations
{

public:

  /// converts Surface::RotationType to AlgebraicMatrix 
  AlgebraicMatrix algebraicMatrix(const Surface::RotationType &rot) const;

  /// converts Surface::RotationType to AlgebraicVector 
  AlgebraicVector algebraicVector(const Surface::RotationType &rot) const;

  /// converts AlgebraicMatrix to Surface::RotationType 
  Surface::RotationType rotationType(const AlgebraicMatrix &algM) const;

  /// converts GlobalVector to AlgebraicVector 
  AlgebraicVector algebraicVector(const GlobalVector &globalVector) const;

  /// gets Euler Angles from RotationType 
  AlgebraicVector eulerAngles(const Surface::RotationType &rot, int flag) const;

  /// transforms Rotation to local Det Frame 
  Surface::RotationType globalToLocalMatrix(const Surface::RotationType &rot,
					    const Surface::RotationType &detrot) const;
 
  /// transforms Rotation to global Frame 
  Surface::RotationType localToGlobalMatrix(const Surface::RotationType &aliDetRot,
					    const Surface::RotationType &detRot) const;

  /// builds rotation matrix from Euler Angles 
  AlgebraicMatrix rotMatrix3(const AlgebraicVector &a) const;

  /// transforms euler angles to local frame 
  AlgebraicVector globalToLocalEulerAngles(const AlgebraicVector &a, 
					   const Surface::RotationType &rot) const;

  /// transforms euler angles to global frame 
  AlgebraicVector localToGlobalEulerAngles(const AlgebraicVector &a, 
					   const Surface::RotationType &rot) const;

  /// repair rotation matrix for rounding errors 
  Surface::RotationType rectify(const Surface::RotationType &rot) const;

};

#endif
