#ifndef Alignment_CommonAlignment_Utilities_H
#define Alignment_CommonAlignment_Utilities_H

/** \namespace align
 *
 *  Namespace for common calculations in alignment.
 *
 *  $Date: 2007/04/09 00:40:21 $
 *  $Revision: 1.7 $
 *  \author Chung Khim Lae
 */

#include "CondFormats/Alignment/interface/Definitions.h"
#include "CondFormats/Alignment/interface/Alignments.h"
#include <CLHEP/Vector/ThreeVector.h>
#include <CLHEP/Vector/AxisAngle.h>
#include <CLHEP/Matrix/Vector.h>
#include <CLHEP/Matrix/SymMatrix.h>
#include <CLHEP/Matrix/Matrix.h>
#include <CLHEP/Vector/Rotation.h>

//#include "Alignment/CommonAlignment/interface/Alignable.h"
//#include "Alignment/CommonAlignment/interface/AlignableSurface.h"

//class Alignable;
//class AlignableSurface;

namespace align
{
  typedef std::vector<GlobalPoint>  GlobalPoints;
  typedef std::vector<GlobalVector> GlobalVectors;
  typedef std::vector<LocalPoint>   LocalPoints;
  typedef std::vector<LocalVector>  LocalVectors;

  /// Convert rotation matrix to angles about x-, y-, z-axes (frame rotation).
  EulerAngles toAngles(
		       const RotationType&
		       );

  /// Convert rotation angles about x-, y-, z-axes to matrix.
  RotationType toMatrix(
			const EulerAngles&
			);

  /// Find mother's position from the average of its daughters' positions.
  PositionType motherPosition(
			      const std::vector<const PositionType*>& dauPos
			      ); 

  /// Find matrix to rotate from nominal to current vectors.
  /// Assume both sets of vectors have the same size and order.
  RotationType diffRot(
		const GlobalVectors& current,
		const GlobalVectors& nominal
		);

	GlobalVector diffR(
		const GlobalVectors& current,
		const GlobalVectors& nominal
		);
	/// Fins the CM of a set of points
	GlobalVector centerOfMass(
		const GlobalVectors& theVs
		);
	
	
	/// Correct a rotation matrix for rounding errors.
  void rectify(
		RotationType&
		);
	
}

#endif
