#ifndef Alignment_CommonAlignment_AlignTools_H
#define Alignment_CommonAlignment_AlignTools_H

/** \namespace align
 *
 *  Tools for comparing alignables
 *
 *  $Date: 2007/09/11 21:24:54 $
 *  $Revision: 1.1 $
 *  \author Nhan Tran
 */

#include "Alignment/CommonAlignment/interface/Utilities.h"

namespace align{

	///Finds the TR between two alignables - first alignable is reference
	AlgebraicVector diffAlignables(Alignable* refAli, Alignable* curAli);

	///Finds the TR between 2 sets of alignables
	///For example, if TIB/TID were to move as one unit
	void diffAlignables(Alignables& refAlis, Alignables& curAlis, std::vector<AlgebraicVector>& diffs);
	
	///Moves the alignable by the AlgebraicVector
	void moveAlignable(Alignable* ali, AlgebraicVector diff);

	///Creates the points which are used in diffAlignables
	///A set of points corresponding to lowest daughters
	void createPoints(GlobalVectors* Vs, Alignable* ali);

}

#endif
