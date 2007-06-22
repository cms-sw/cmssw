/** \file AlignmentAlgorithmBW.cc
 *  Implementation of Bruno Wittmer's alignment algorithm for the Laser Alignment System
 *
 *  $Date: 2007/04/05 08:32:32 $
 *  $Revision: 1.2 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/interface/AlignmentAlgorithmBW.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

AlignmentAlgorithmBW::AlignmentAlgorithmBW()
{
	
}

AlignmentAlgorithmBW::~AlignmentAlgorithmBW()
{
	
}

void AlignmentAlgorithmBW::run(const std::string theName, LASvec2D & data, LASvec2D & errors, bool useBSframe, int theRing)
{
	edm::LogInfo("AlignmentAlgorithmBW") << " running the alignment algorithm for " << theName;
	// run the actual algorithm to calculate the alignment parameters
	
	// transpose the data
	LASvec2D dataTrans = trans(data);
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

	/**
	 * A = sum of all beam positions
	**/
	double A = sum(dataTrans);
	/**
	 * B = sum of data[i][j]*zPositions[i]
	**/
	double B = sum(multiply(data,zPositions));
	/**
	 * Ai = vector containing sum of the columns in dataTrans (i.e. beam positions on disc i in all sectors)
	**/
	LASvec Ai = sumc(dataTrans);
	/**
	 * Ak = vector containing sum of the rows in dataTrans (i.e. beam position in sector j on all discs)
	**/
	LASvec Ak = sumr(dataTrans);
	/**
	 * C = sum of dataTrans[i][j] * sin(phiPositions[i])
	**/
	double C = sum(multiply(dataTrans,sin(phiPositions)));
	/**
	 * Ci = vector containing sum of the columns in dataTrans*sin(phiPositions) (i.e. position of disc i in all sectors * sin(phiPositions))
	**/
	LASvec Ci = sumc(multiply(dataTrans,sin(phiPositions)));
	/**
	 * D = sum of trans(dataTrans * sin(phiPositions)) * zPositions
	**/
	double D = sum(multiply(trans(multiply(dataTrans,sin(phiPositions))),zPositions));
	/**
	 * E = sum of dataTrans[i][j] * cos(phiPositions[i])
	**/
	double E = sum(multiply(dataTrans,cos(phiPositions)));
	/**
	 * Ei = vector containing sum of the columns in dataTrans*cos(phiPositions) (i.e. position of disc i in all sectors * cos(phiPositions))
	**/
	LASvec Ei = sumc(multiply(dataTrans,cos(phiPositions)));
	/**
	 * F = sum of trans(dataTrans * cos(phiPositions)) * zPositions
	**/
	double F = sum(multiply(trans(multiply(dataTrans,cos(phiPositions))),zPositions));
	
	// ********************************************************************** //
	// **               Calculate the alignment parameters                 ** //
	// ********************************************************************** //

	/**
	 * dphi0 = (sZq * A - sZ * B)/nenn1/r0
	**/
	double dphi0 = (sZq * A - sZ * B)/nenn1/r0;
	/**
	 * dphit = theTECLength * (nDiscs * B - sZ * A)/nenn1/r0
	**/
	double dphit = theTECLength * (nDiscs * B - sZ * A)/nenn1/r0;
	/**
	 * dphik = -(zPositions * dphit/theTECLength + dphi0 + Ai/nSectors/r0)  
	**/
	LASvec dphik = -1.0 * (zPositions * dphit/theTECLength + dphi0 + Ai *1.0/nSectors/r0);
	/**
	 * dx0 = 2*(sZ * D - sZq * C)/nenn1
	**/
	double dx0 = 2.0 * (sZ * D - sZq * C)/nenn1;
	/**
	 * dxt = 2*theTECLength*(sZ*C - nDiscs * D)/nenn1
	**/
	double dxt = 2.0 * theTECLength * (sZ * C - nDiscs * D)/nenn1;
	/**
	 * dxk = -zPositions * dxt/theTECLength - dx0 + Ci * 2/nSectors
	**/
	LASvec dxk = -1.0 * zPositions * dxt/theTECLength - dx0 + Ci * 2.0/nSectors;
	/**
	 * dy0 = 2*(sZq * E - sZ * F)/nenn1
	**/
	double dy0 = 2.0 * (sZq * E - sZ * F)/nenn1;
	/**
	 * dyt = 2*theTECLength*(nDiscs*F-sZ*E)/nenn1
	**/
	double dyt = 2.0 * theTECLength * (nDiscs * F - sZ * E)/nenn1;
	/**
	 * dyk = -zPositions * dyt/theTECLength - dy0 - Ei * 2/nSectors
	**/
	LASvec dyk = -1.0 * zPositions * dyt/theTECLength - dy0 - Ei * 2.0/nSectors;
	
	// print the parameters
	LogDebug("AlignmentAlgorithmBW") << " here are the calculated alignment parameters:" << std::endl
		<< "      \t Dphi0 = " << dphi0 << "\t Dx0 = " << dx0 << "\t Dy0 = " << dy0 << std::endl
		<< "      \t Dphit = " << dphit << "\t Dxt = " << dxt << "\t Dyt = " << dyt << std::endl << std::endl
		<< "      \t Dphi[1] = " << dphik[0] << "\t Dx[1] = " << dxk[0] << " \t Dy[1] = " << dyk[0] << std::endl
		<< "      \t Dphi[2] = " << dphik[1] << "\t Dx[2] = " << dxk[1] << " \t Dy[2] = " << dyk[1] << std::endl
		<< "      \t Dphi[3] = " << dphik[2] << "\t Dx[3] = " << dxk[2] << " \t Dy[3] = " << dyk[2] << std::endl
		<< "      \t Dphi[4] = " << dphik[3] << "\t Dx[4] = " << dxk[3] << " \t Dy[4] = " << dyk[3] << std::endl
		<< "      \t Dphi[5] = " << dphik[4] << "\t Dx[5] = " << dxk[4] << " \t Dy[5] = " << dyk[4] << std::endl
		<< "      \t Dphi[6] = " << dphik[5] << "\t Dx[6] = " << dxk[5] << " \t Dy[6] = " << dyk[5] << std::endl
		<< "      \t Dphi[7] = " << dphik[6] << "\t Dx[7] = " << dxk[6] << " \t Dy[7] = " << dyk[6] << std::endl
		<< "      \t Dphi[8] = " << dphik[7] << "\t Dx[8] = " << dxk[7] << " \t Dy[8] = " << dyk[7] << std::endl
		<< "      \t Dphi[9] = " << dphik[8] << "\t Dx[9] = " << dxk[8] << " \t Dy[9] = " << dyk[8] << std::endl;
		
		// we want to store this parameters in a separate DataFormat to study them later in more detail!?
}

AlignmentAlgorithmBW::LASvec2D AlignmentAlgorithmBW::trans(LASvec2D input)
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

double AlignmentAlgorithmBW::sum(LASvec2D input)
{
	double result = 0.0;
	for( unsigned int i = 0; i < input.size(); ++i )
	{
		result += input[i].sum();
	}
	return result;
}

AlignmentAlgorithmBW::LASvec AlignmentAlgorithmBW::sumc(LASvec2D input)
{
	LASvec result;
	LASvec2D temp = trans(input);
	for( unsigned int i = 0; i < temp.size(); ++i )
	{
		result[i] = temp[i].sum();
	}
	return result;
}

AlignmentAlgorithmBW::LASvec AlignmentAlgorithmBW::sumr(LASvec2D input)
{
	LASvec result;
	for( unsigned int i = 0; i < input.size(); ++i )
	{
		result[i] = input[i].sum();
	}
	return result;
}

AlignmentAlgorithmBW::LASvec2D AlignmentAlgorithmBW::multiply(LASvec2D input, LASvec factor)
{
	if (input[0].size() != factor.size())
		throw cms::Exception("AlignmentAlgorithmBW","Size of Matrix and Vector do not match!") << "size of LASvec2D ("
		<< input.size() << "," << input[0].size() << " does not match the size of LASvec " << factor.size();
		
	LASvec2D result;
	for( unsigned int i = 0; i < input.size(); ++i )
	{
		for( unsigned int j = 0; j < input[i].size(); ++j )
		{
			result[i][j] = input[i][j] * factor[j];
		}
	}
	return result;
}

AlignmentAlgorithmBW::LASvec AlignmentAlgorithmBW::makeZPositions(bool useBSframe)
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
