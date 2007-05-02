#ifndef LaserAlignmentAlignmentAlgorithmBW_H
#define LaserAlignmentAlignmentAlgorithmBW_H

/** \class AlignmentAlgorithmBW
 *  Alignment Algorithm for the Laser Alignment System developed by Bruno Wittmer.
 *
 *  The details of this algorithm are given in his <a href="http://darwin.bth.rwth-aachen.de/opus3/volltexte/2002/348/">PhD Thesis</a>
 *
 *  $Date: 2007/04/12 14:20:32 $
 *  $Revision: 1.4 $
 *  \author Maarten Thomas
 */

#include <vector>
#include "TMatrixT.h"
#include "DataFormats/LaserAlignment/interface/LASAlignmentParameter.h"
#include "Alignment/LaserAlignment/interface/LASvector.h"
#include "Alignment/LaserAlignment/interface/LASvector2D.h"

class AlignmentAlgorithmBW
{
public:
	/// define vector and matrix formats for easier calculation of the alignment corrections
	typedef LASvector<double> LASvec;
	typedef LASvector2D<double> LASvec2D;
	
public:
	/// constructor
	AlignmentAlgorithmBW();
	/// destructor
	virtual ~AlignmentAlgorithmBW();
	/// the actual algorithm
	std::vector<LASAlignmentParameter> run(const std::string theName, LASvec2D & data, LASvec2D & errors, bool useBSframe, int theRing);

private:
	/// create the z positions of the discs
	LASvec makeZPositions(bool useBSframe);
};

#endif /* LaserAlignmentAlignmentAlgorithmBW_H */
