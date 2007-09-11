#ifndef Alignment_CommonAlignment_AlignTools_H
#define Alignment_CommonAlignment_AlignTools_H

/** \namespace align
 *
 *  Tools for comparing alignables
 *
 *  $Date: 2007/08/22 00:40:21 $
 *  $Revision: 1.1 $
 *  \author Nhan Tran
 */

#include "CondFormats/Alignment/interface/Definitions.h"
#include "Alignment/CommonAlignment/interface/Utilities.h"
#include "Alignment/CommonAlignment/interface/SurveyDet.h"
#include "Alignment/CommonAlignment/interface/Alignable.h"

namespace align{
	
	typedef std::vector<GlobalPoint>  GlobalPoints;
  typedef std::vector<GlobalVector> GlobalVectors;
  typedef std::vector<LocalPoint>   LocalPoints;
  typedef std::vector<LocalVector>  LocalVectors;

	///Finds the TR between two alignables - first alignable is reference
	AlgebraicVector diffAlignables(Alignable* refAli, Alignable* curAli);

	///Finds the TR between 2 sets of alignables
	///For example, if TIB/TID were to move as one unit
	void diffAlignables(std::vector<Alignable*>& refAlis, std::vector<Alignable*>& curAlis, std::vector<AlgebraicVector>& diffs);
	
	///Moves the alignable by the AlgebraicVector
	void moveAlignable(Alignable* ali, AlgebraicVector diff);

	///Creates the points which are used in diffAlignables
	///A set of points corresponding to lowest daughters
	void createPoints(GlobalVectors* Vs, Alignable* ali);

}

#endif
