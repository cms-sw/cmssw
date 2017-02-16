#ifndef Alignment_CommonAlignment_Utilities_H
#define Alignment_CommonAlignment_Utilities_H

/** \namespace align
 *
 *  Namespace for common calculations in alignment.
 *
 *  $Date: 2007/10/08 15:22:08 $
 *  $Revision: 1.10 $
 *  \author Chung Khim Lae
 */

#include <map>
#include <memory>

#include "CondFormats/Alignment/interface/Definitions.h"
#include "Alignment/CommonAlignment/interface/AlignableObjectId.h"

class Alignable;
class AlignmentParameters;
class AlignmentLevel;

namespace align
{
  typedef std::vector<Scalar>       Scalars;
  typedef std::vector<GlobalPoint>  GlobalPoints;
  typedef std::vector<GlobalVector> GlobalVectors;
  typedef std::vector<LocalPoint>   LocalPoints;
  typedef std::vector<LocalVector>  LocalVectors;
  typedef std::vector<Alignable*>   Alignables;
  typedef std::vector<AlignmentParameters*> Parameters;
  typedef std::vector<std::unique_ptr<AlignmentLevel> > AlignmentLevels;

  typedef std::map<std::pair<Alignable*, Alignable*>, AlgebraicMatrix> Correlations;

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

  /// Find the CM of a set of points
  GlobalVector centerOfMass(
			    const GlobalVectors& theVs
			    );
	

  /// Correct a rotation matrix for rounding errors.
  void rectify(
	       RotationType&
	       );
}

#endif
