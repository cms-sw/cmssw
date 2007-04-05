/** \file AlignmentAlgorithmBW.cc
 *  Implementation of Bruno Wittmer's alignment algorithm for the Laser Alignment System
 *
 *  $Date: 2007/04/05 06:21:22 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/interface/AlignmentAlgorithmBW.h"
#include "FWCore/Utilities/interface/Exception.h"

AlignmentAlgorithmBW::AlignmentAlgorithmBW()
{
	
}

AlignmentAlgorithmBW::~AlignmentAlgorithmBW()
{
	
}

AlignmentAlgorithmBW::run(LASvec2D & data, LASvec2D & errors, bool useBSframe, int theRing)
{
	// run the actual algorithm to calculate the alignment parameters
	
	// transpose the data
	LASvec2D dataTrans = data.trans;
	// create vector with the z positions of the discs
	LASvec zPositions = makeZPositions(useBSframe);
	
	// set the radius and the phi positions of the beams
	double r0;
	LASvec phiPositions;
	
	switch (theRing)
	{
		case 4:
		r0 = 56.4; // radius of Ring 4
		for( unsigned int i = 0; i < 8; ++i )
		{
			// calculate the right phi for the current beam
		  phiPositions[i] = 0.3927 + double(i * double(double(2 * M_PI) / 8));
		}
		break;
		case 6:
		r0 = 84.0; // radius of Ring 6
		for( unsigned int i = 0; i < 8; ++i )
		{
			// calculate the right phi for the current beam
		  phiPositions[i] = 0.3927 + double(i * double(double(2 * M_PI) / 8));
		}
		break;
		default:
		throw cms::Exception("WrongRingNumber","Not a valid ring number") << theRing << " is not a valid ring number for this alignment algorithm!!!\n";
	}

	// lenght of an endcap
	double theTECLength = zPositions[8] - zPositions[0];
	// number of sectors and discs
	int nSectors = data.size();
	int nDiscs = data[0].size();
	
	// calculate first some constant parameters
	/**
	 * sZ = \f$\sum_k z_k\f$
	**/
	double sZ=zPositions.sum();
	/**
	 * sZq = \f$\sum_k z_k^2\f$
	**/
	double sZq=(zPositions*zPositions).sum();
	/**
	 * nenn1 = \f$ n_{beams} \cdot \left(\left(\sum_k z_k\right)^2 - n_{discs} \sum_k z_k^2\right)\f$
	**/
	double nenn1 = nSectors * (pow(sZ,2)-nDiscs*sZq);
	/**
	 * sT = \f$ \sum_i \Theta_i \f$
	**/
	double sT = phiPositions.sum();
	/**
	 * sTq = \f$ \sum_i \Theta_i^2 \f$
	**/
	double sTq = (phiPositions*phiPositions).sum();
	/**
	 * nenn2 = \f$ n_{beams} \cdot \sum_i \Theta_i^2 - \left(\sum_i \Theta_i \right)^2 \f$
	**/
	double nenn2 = nSectors * sTq - pow(sT,2);
	/**
	 * nenn3 = \f$ n_{discs} \cdot \sum_k z_k^2 - \left(\sum_k z_k \right)^2 \f$
	**/
	double nenn3 = nDiscs * sZq - pow(sZ,2);
	
}

LASvec2D AlignmentAlgorithmBW::trans(LASvec2D & input)
{
	LASvec2D result;
	for( unsigned int i = 0; i < input.size(); ++i )
	{
		for( unsigned int j = 0; j < input[i].size(); ++j )
		{
			result[j][i] = input[i][j];
		}
	}
	return result;
}

LASvec AlignmentAlgorithmBW::sumc(LASvec2D & input)
{
	LASvec result;
	LASvec2D temp = trans(input);
	for( unsigned int i = 0; i < temp.size(); ++i )
	{
		result[i] = temp[i].sum();
	}
	return result;
}

LASvec AlignmentAlgorithmBW::sumr(LASvec2D & input)
{
	LASvec result;
	for( unsigned int i = 0; i < input.size(); ++i )
	{
		result[i] = input[i].sum();
	}
	return result;
}

LASvec AlignmentAlgorithmBW::makeZPositions(bool useBSframe)
{
	LASvec zPos(9,0);
	zPos[0] = 14.0;
	zPos[1] = 28.0;
	zPos[2] = 42.0;
	zPos[3] = 56.0;
	zPos[4] = 73.5;
	zPos[5] = 92.5;
	zPos[6] = 113.0;
	zPos[7] = 134.5;
	
	if(useBSframe) zPos -= 70.5; // Beamsplitter at z = 0
	
	return zPos;
}
