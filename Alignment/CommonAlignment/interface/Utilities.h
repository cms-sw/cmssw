#ifndef Alignment_CommonAlignment_Utilities_H
#define Alignment_CommonAlignment_Utilities_H

/** \namespace align
 *
 *  Namespace for common calculations in alignment.
 *
 *  $Date: 2007/04/07 03:29:04 $
 *  $Revision: 1.6 $
 *  \author Chung Khim Lae
 */

#include "CondFormats/Alignment/interface/Definitions.h"

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

  /// Correct a rotation matrix for rounding errors.
  void rectify(
	       RotationType&
	       );
}

#endif
