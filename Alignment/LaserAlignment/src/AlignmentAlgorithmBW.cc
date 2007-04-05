/** \file AlignmentAlgorithmBW.cc
 *  Implementation of Bruno Wittmer's alignment algorithm for the Laser Alignment System
 *
 *  $Date: Wed Apr  4 17:01:29 CEST 2007 $
 *  $Revision: 1.1 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/interface/AlignmentAlgorithmBW.h"

AlignmentAlgorithmBW::AlignmentAlgorithmBW()
{
	
}

AlignmentAlgorithmBW::~AlignmentAlgorithmBW()
{
	
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