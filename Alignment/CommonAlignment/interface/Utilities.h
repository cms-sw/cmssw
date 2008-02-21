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

#include "CondFormats/Alignment/interface/Definitions.h"

class Alignable;
class AlignmentParameters;

namespace align
{
  typedef std::vector<Scalar>       Scalars;
  typedef std::vector<GlobalPoint>  GlobalPoints;
  typedef std::vector<GlobalVector> GlobalVectors;
  typedef std::vector<LocalPoint>   LocalPoints;
  typedef std::vector<LocalVector>  LocalVectors;
  typedef std::vector<LocalVector>  LocalVectors;
  typedef std::vector<Alignable*>   Alignables;
  typedef std::vector<AlignmentParameters*> Parameters;

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

  /// Find the position of an Alignable within its mother. Starts from 1.
  /// 0 if ID or structure name is invalid.
  unsigned int position(
   	                    align::ID,
      					  		  const std::string& // structure name
                        );

  /// Name of tree of an Alignable given its ID and structure name.
  /// Eg: TPB HalfBarrel1 Layer2 Ladder3 Module4.
  std::string treeName(
      								 align::ID,
      								 const std::string&,  // structure name
           	           char delimiter = ' ' // separator between 2 levels
      							   );
}

#endif
