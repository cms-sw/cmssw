/** \file AlignmentAlgorithmBW.cc
 *  Implementation of Bruno Wittmer's alignment algorithm for the Laser Alignment System
 *
 *  $Date: 2008/01/22 19:18:03 $
 *  $Revision: 1.11 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/interface/AlignmentAlgorithmBW.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <math.h>

///
///
///
AlignmentAlgorithmBW::AlignmentAlgorithmBW() {
}





///
///
///
AlignmentAlgorithmBW::~AlignmentAlgorithmBW() {
}





///
///
///
std::vector<LASAlignmentParameter> AlignmentAlgorithmBW::run(const std::string theName, LASvec2D & data, LASvec2D & errors, bool useBSframe, int theRing) {
  
  // run the actual algorithm to calculate the alignment parameters
  edm::LogInfo("AlignmentAlgorithmBW") << " running the alignment algorithm for " << theName;
  
  // create vector to store the alignment parameters
  std::vector<LASAlignmentParameter> theResult;

  // transpose the data
  LASvec2D dataTrans(data[0].size(),data.size());
  dataTrans.transpose(data);

  // create vector with the z positions of the discs
  LASvec zPositions = makeZPositions( useBSframe );

  // set the radius and the phi positions of the beams
  double r0;
  LASvec phiPositions(8);
  
  switch( theRing ) {
    case 4:
      r0 = 56.4; // radius of Ring 4
      for( unsigned int i = 0; i < 8; ++i ) {
	// calculate the right phi for the current beam
	phiPositions[i] = 0.3927 + double(i * double(double(2 * M_PI) / 8));
      }
      break;
  case 6:
    r0 = 84.0; // radius of Ring 6
    for( unsigned int i = 0; i < 8; ++i ) {
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
  // 	double sT = phiPositions.sum();
  /**
   * sTq = \f$ \sum_i \Theta_i^2 \f$
   **/
  // 	double sTq = (phiPositions*phiPositions).sum();
  /**
   * nenn2 = \f$ n_{beams} \cdot \sum_i \Theta_i^2 - \left(\sum_i \Theta_i \right)^2 \f$
   **/
  // 	double nenn2 = nSectors * sTq - pow(sT,2);
  /**
   * nenn3 = \f$ n_{discs} \cdot \sum_k z_k^2 - \left(\sum_k z_k \right)^2 \f$
   **/
  // 	double nenn3 = nDiscs * sZq - pow(sZ,2);
  /**
   * A = sum of all beam positions
   **/
  double A = dataTrans.sum();
  /**
   * B = sum of data[i][j]*zPositions[i]
   **/
  double B = (data * zPositions).sum();
  /**
   * Ai = vector containing sum of the columns in dataTrans (i.e. beam positions on disc i in all sectors)
   **/
  LASvec Ai = dataTrans.sumC();
  /**
   * Ak = vector containing sum of the rows in dataTrans (i.e. beam position in sector j on all discs)
   **/
  LASvec Ak = dataTrans.sumR();
  /**
   * C = sum of dataTrans[i][j] * sin(phiPositions[i])
   **/
  LASvec sinPhiPositions(8);
  sinPhiPositions.sine(phiPositions);
  double C = (dataTrans * sinPhiPositions).sum();
  /**
   * Ci = vector containing sum of the columns in dataTrans*sin(phiPositions) (i.e. position of disc i in all sectors * sin(phiPositions))
   **/
  LASvec Ci = (dataTrans * sinPhiPositions).sumC();
  /**
   * D = sum of trans(dataTrans * sin(phiPositions)) * zPositions
   **/
  LASvec2D dataTransSin(8,9);
  dataTransSin.transpose(dataTrans * sinPhiPositions);
  double D = (dataTransSin * zPositions).sum();
  /**
   * E = sum of dataTrans[i][j] * cos(phiPositions[i])
   **/
  LASvec cosPhiPositions(8);
  cosPhiPositions.cosine(phiPositions);
  double E = (dataTrans * cosPhiPositions).sum();
  /**
   * Ei = vector containing sum of the columns in dataTrans*cos(phiPositions) (i.e. position of disc i in all sectors * cos(phiPositions))
   **/
  LASvec Ei = (dataTrans * cosPhiPositions).sumC();
  /**
   * F = sum of trans(dataTrans * cos(phiPositions)) * zPositions
   **/
  LASvec2D dataTransCos(8,9);
  dataTransCos.transpose(dataTrans * cosPhiPositions);
  double F = (dataTransCos * zPositions).sum();
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
  LASvec dphik = (zPositions * dphit/theTECLength + dphi0 + Ai *1.0/nSectors/r0) * -1.0;
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
  LASvec dxk = zPositions * (-1.0) * dxt/theTECLength - dx0 + Ci * 2.0/nSectors;
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
  LASvec dyk = zPositions *(-1.0)* dyt/theTECLength - dy0 - Ei * 2.0/nSectors;
	
  // print the parameters
  LogDebug("AlignmentAlgorithmBW") << " here are the calculated alignment parameters:" << std::endl
				   << "      Dphi0 = " << dphi0 << "  Dx0 = " << dx0 << "  Dy0 = " << dy0 << std::endl
				   << "      Dphit = " << dphit << "  Dxt = " << dxt << "  Dyt = " << dyt << std::endl << std::endl
				   << "      Dphi[1] = " << dphik[0] << "  Dx[1] = " << dxk[0] << "   Dy[1] = " << dyk[0] << std::endl
				   << "      Dphi[2] = " << dphik[1] << "  Dx[2] = " << dxk[1] << "   Dy[2] = " << dyk[1] << std::endl
				   << "      Dphi[3] = " << dphik[2] << "  Dx[3] = " << dxk[2] << "   Dy[3] = " << dyk[2] << std::endl
				   << "      Dphi[4] = " << dphik[3] << "  Dx[4] = " << dxk[3] << "   Dy[4] = " << dyk[3] << std::endl
				   << "      Dphi[5] = " << dphik[4] << "  Dx[5] = " << dxk[4] << "   Dy[5] = " << dyk[4] << std::endl
				   << "      Dphi[6] = " << dphik[5] << "  Dx[6] = " << dxk[5] << "   Dy[6] = " << dyk[5] << std::endl
				   << "      Dphi[7] = " << dphik[6] << "  Dx[7] = " << dxk[6] << "   Dy[7] = " << dyk[6] << std::endl
				   << "      Dphi[8] = " << dphik[7] << "  Dx[8] = " << dxk[7] << "   Dy[8] = " << dyk[7] << std::endl
				   << "      Dphi[9] = " << dphik[8] << "  Dx[9] = " << dxk[8] << "   Dy[9] = " << dyk[8] << std::endl;
		
  // std::cout << " here are the calculated alignment parameters:" << std::endl
  //  << "      Dphi0 = " << dphi0 << "  Dx0 = " << dx0 << "  Dy0 = " << dy0 << std::endl
  //  << "      Dphit = " << dphit << "  Dxt = " << dxt << "  Dyt = " << dyt << std::endl << std::endl
  //  << "      Dphi[1] = " << dphik[0] << "  Dx[1] = " << dxk[0] << "   Dy[1] = " << dyk[0] << std::endl
  //  << "      Dphi[2] = " << dphik[1] << "  Dx[2] = " << dxk[1] << "   Dy[2] = " << dyk[1] << std::endl
  //  << "      Dphi[3] = " << dphik[2] << "  Dx[3] = " << dxk[2] << "   Dy[3] = " << dyk[2] << std::endl
  //  << "      Dphi[4] = " << dphik[3] << "  Dx[4] = " << dxk[3] << "   Dy[4] = " << dyk[3] << std::endl
  //  << "      Dphi[5] = " << dphik[4] << "  Dx[5] = " << dxk[4] << "   Dy[5] = " << dyk[4] << std::endl
  //  << "      Dphi[6] = " << dphik[5] << "  Dx[6] = " << dxk[5] << "   Dy[6] = " << dyk[5] << std::endl
  //  << "      Dphi[7] = " << dphik[6] << "  Dx[7] = " << dxk[6] << "   Dy[7] = " << dyk[6] << std::endl
  //  << "      Dphi[8] = " << dphik[7] << "  Dx[8] = " << dxk[7] << "   Dy[8] = " << dyk[7] << std::endl
  //  << "      Dphi[9] = " << dphik[8] << "  Dx[9] = " << dxk[8] << "   Dy[9] = " << dyk[8] << std::endl;
  //    
  // //____________________________________________________________________________________________________   
  // // ONLY FOR DEBUGGING
  //   for (int j = 0; j < 8; ++j)
  //     {
  //       // calculate the corrections and the errors
  //       double thePhiCorrected = dphik[0] 
  //      - (sin(phiPositions[0])/r0) * dxk[0]
  //      + (cos(phiPositions[0])/r0) * dyk[0]
  //      - ( dphik[j+1]
  //         - (sin(phiPositions[0])/r0) * dxk[j+1]
  //         + (cos(phiPositions[0])/r0) * dyk[j+1] );
  // 
  //       // for debugging
  //          std::cout << " Fitted relative Correction for " << theName << " in Phi[" << j << "] = " << thePhiCorrected << std::endl;
  //      }
  // 
  //   for (int j = 0; j < 9; ++j)
  //     {
  //       // calculate the correction for each disk (not relative to disk one)
  //       double theAbsPhiCorrected = dphik[j] 
  //         - (sin(phiPositions[0])/r0) * dxk[j]
  //            + (cos(phiPositions[0])/r0) * dyk[j];
  // 
  //       // for debugging
  //          std::cout << " Fitted Correction for " << theName << " in Phi[" << j << "] = " << theAbsPhiCorrected << std::endl;
  //     }
  // // ONLY FOR DEBUGGING    
  // //____________________________________________________________________________________________________
	
  // we want to store this parameters in a separate DataFormat to study them later in more detail
  // unfortunately storing a std::valarray is not working due to reflex dictionary troubles. Therefore
  // store the calculated corrections into a std::vector<double>
  std::vector<double> dphik_, dxk_, dyk_;
  for( unsigned int i = 0; i < 9; ++i ) {
    dphik_.push_back(dphik[i]);
    dxk_.push_back(dxk[i]);
    dyk_.push_back(dyk[i]);
  }
	
  theResult.push_back(LASAlignmentParameter(theName,dphi0,dphit,dphik_,dx0,dxt,dxk_,dy0,dyt,dyk_));
  return theResult;
}


///
///
///
AlignmentAlgorithmBW::LASvec AlignmentAlgorithmBW::makeZPositions( bool useBSframe ) {

  // if the BS frame is used, the BS is positioned at 0.0, disc 1 to 5: z < 0, disc 6 to 9: z > 0.
  LASvec zPos(9);
  zPos[0] = 0.0;
  zPos[1] = 14.0;
  zPos[2] = 28.0;
  zPos[3] = 42.0;
  zPos[4] = 56.0;
  zPos[5] = 73.5;
  zPos[6] = 92.5;
  zPos[7] = 113.0;
  zPos[8] = 134.5;
	
  if(useBSframe) zPos -= 70.5; // Beamsplitter at z = 0
	
  return zPos;
}
