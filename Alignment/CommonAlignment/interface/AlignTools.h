#ifndef Alignment_CommonAlignment_AlignTools_H
#define Alignment_CommonAlignment_AlignTools_H

/** \namespace align
 *
 *  Tools for comparing alignables
 *
 *  $Date: 2008/06/17 12:46:54 $
 *  $Revision: 1.5 $
 *  \author Nhan Tran
 */

#include "Alignment/CommonAlignment/interface/Utilities.h"

namespace align{

	///Finds the TR between two alignables - first alignable is
        ///reference. Returns a vector with 12 components. First six are global, second
        ///six are local.
	AlgebraicVector diffAlignables(Alignable* refAli, Alignable* curAli, const std::string &weightBy, bool weightById, const std::vector< unsigned int > &weightByIdVector);
	
	///Finds the TR between 2 sets of alignables
	///For example, if TIB/TID were to move as one unit
	//void diffAlignables(Alignables& refAlis, Alignables& curAlis, std::vector<AlgebraicVector>& diffs);
	
	///Moves the alignable by the AlgebraicVector
	void moveAlignable(Alignable* ali, AlgebraicVector diff);

	///Creates the points which are used in diffAlignables
	///A set of points corresponding to lowest daughters
	void createPoints(GlobalVectors* Vs, Alignable* ali, const std::string &weightBy, bool weightById, const std::vector< unsigned int > &weightByIdVector);
	
	// read module list, return bool
	bool readModuleList( unsigned int, unsigned int, const std::vector< unsigned int > & );

}

#endif
