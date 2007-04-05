#ifndef LaserAlignmentAlignmentAlgorithmBW_H
#define LaserAlignmentAlignmentAlgorithmBW_H

/** \class AlignmentAlgorithmBW
 *  Alignment Algorithm for the Laser Alignment System developed by Bruno Wittmer.
 *
 *  The details of this algorithm are given in his <a href="http://darwin.bth.rwth-aachen.de/opus3/volltexte/2002/348/">PhD Thesis</a>
 *
 *  $Date: Wed Apr  4 16:50:14 CEST 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

class AlignmentAlgorithmBW
{
	/// define vector formats for easier calculation of the alignment corrections
	typedef std::valarray<double> LASvec;
	typedef std::valarray<LASvec> LASvec2D;
	
public:
	/// constructor
	AlignmentAlgorithmBW();
	/// destructor
	virtual ~AlignmentAlgorithmBW();

private:
	/* data */
	
	/// return the transposed LASvec2D
	LASvec2D trans(LASvec2D & input);
	/// return sum of elements in the columns of a LASvec2D
	LASvec sumc(LASvec2D & input);
	/// return sum of elements in the rows of a LASvec2D
	LASvec sumr(LASvec2D & input);
	
};

#endif /* LaserAlignmentAlignmentAlgorithmBW_H */
