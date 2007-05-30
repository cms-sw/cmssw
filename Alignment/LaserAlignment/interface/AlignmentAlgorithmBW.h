#ifndef LaserAlignmentAlignmentAlgorithmBW_H
#define LaserAlignmentAlignmentAlgorithmBW_H

/** \class AlignmentAlgorithmBW
 *  Alignment Algorithm for the Laser Alignment System developed by Bruno Wittmer.
 *
 *  The details of this algorithm are given in his <a href="http://darwin.bth.rwth-aachen.de/opus3/volltexte/2002/348/">PhD Thesis</a>
 *
 *  $Date: 2007/04/05 08:32:31 $
 *  $Revision: 1.2 $
 *  \author Maarten Thomas
 */

#include <valarray>

class AlignmentAlgorithmBW
{
public:
	/// define vector formats for easier calculation of the alignment corrections
	typedef std::valarray<double> LASvec;
	typedef std::valarray<LASvec> LASvec2D;
	
public:
	/// constructor
	AlignmentAlgorithmBW();
	/// destructor
	virtual ~AlignmentAlgorithmBW();
	/// the actual algorithm
	void run(const std::string theName, LASvec2D & data, LASvec2D & errors, bool useBSframe, int theRing);

private:
	/// return the transposed LASvec2D
	LASvec2D trans(LASvec2D input);
	/// return the sum of all elements of a LASvec2D
	double sum(LASvec2D input);
	/// return sum of elements in the columns of a LASvec2D
	LASvec sumc(LASvec2D input);
	/// return sum of elements in the rows of a LASvec2D
	LASvec sumr(LASvec2D input);
	/// return the result of LASvec2D * LASvec
	LASvec2D multiply(LASvec2D input, LASvec factor);
	/// create the z positions of the discs
	LASvec makeZPositions(bool useBSframe);
	
};

#endif /* LaserAlignmentAlignmentAlgorithmBW_H */
