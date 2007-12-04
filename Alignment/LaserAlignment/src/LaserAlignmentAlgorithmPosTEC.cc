/** \file LaserAlignmentAlgorithmPosTEC.cc
 *  
 *
 *  $Date: 2007/10/11 09:19:38 $
 *  $Revision: 1.4 $
 *  \author Maarten Thomas
 */

#include "Alignment/LaserAlignment/interface/LaserAlignmentAlgorithmPosTEC.h"
#include "Alignment/LaserAlignment/interface/Millepede.h" 
#include "FWCore/ParameterSet/interface/ParameterSet.h" 
#include <boost/cstdint.hpp> 

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

LaserAlignmentAlgorithmPosTEC::LaserAlignmentAlgorithmPosTEC(edm::ParameterSet const& theConf, int theAlignmentIteration) 
  : theFirstFixedDiskPosTEC(theConf.getUntrackedParameter<int>("FirstFixedParameterPosTEC",2)), 
    theSecondFixedDiskPosTEC(theConf.getUntrackedParameter<int>("SecondFixedParameterPosTEC",3)),
    theGlobalParametersPosTEC(), theLocalParametersPosTEC()
{

  // initialize Millepede
  initMillepede(theAlignmentIteration);
}

LaserAlignmentAlgorithmPosTEC::~LaserAlignmentAlgorithmPosTEC()
{
}

void LaserAlignmentAlgorithmPosTEC::addLaserBeam(std::vector<double> theMeasurements, int LaserBeam, int LaserRing)
{
  LogDebug("LaserAlignmentAlgorithmPosTEC") << "<LaserAlignmentAlgorithmPosTEC::addLaserBeam()>: adding a new Laser Beam ... ";
  
  float RRing = 0.0;

  if (LaserRing == 4) 
    { RRing = 56.4; }             // radius of Ring 4 
  else if (LaserRing == 6)
    { RRing = 84.0; }             // radius of Ring 6

  float LaserPhi0 = 0.392699082;  // phi position of the first Laserdiode in Ring 4
  float LaserPhi = 0.0;           // phi position of the actual beam; will be calculated
  int nLaserBeams = 8;            // number of beams in one ring

  float PhiMeasured[9] = {0.0};   // the measured phi positions
  float PhiErrors[9] = {0.0};     // errors of the measurements

  for (int i = 0; i < 9; i++)
    {
      PhiMeasured[i] = float(theMeasurements.at(2*i));
      PhiErrors[i] = float(theMeasurements.at((2*i)+1));
      LogDebug("AlignmentAlgorithmPosTEC") << " i = " << i << "\t PhiMeasured[" << i << "] = " << PhiMeasured[i]
					   << "\t PhiError[" << i << "] = " << PhiErrors[i];
    }

  // calculate phi for this beam
  LaserPhi = LaserPhi0 + float(LaserBeam * float(float(2 * M_PI) / nLaserBeams));

  // loop over the discs
  for (int theDisk = 0; theDisk < 9; theDisk++)
    {
      theLocalParametersPosTEC[0] = 1.0;
      theGlobalParametersPosTEC[3*theDisk] = 1.0;                        // displacement by dphi
      theGlobalParametersPosTEC[(3*theDisk)+1] = -sin(LaserPhi)/RRing;  // displacement by dx
      theGlobalParametersPosTEC[(3*theDisk)+2] = cos(LaserPhi)/RRing;   // displacement by dy

      // create the equation for a single measurement
      equloc_(theGlobalParametersPosTEC, theLocalParametersPosTEC, &PhiMeasured[theDisk], &PhiErrors[theDisk]);
    }
  
  // local fit after one laser beam has been added
  fitloc_();
}

void LaserAlignmentAlgorithmPosTEC::doGlobalFit(AlignableTracker * theAlignableTracker)
{
  edm::LogInfo("AlignmentAlgorithmPosTEC") << "<AlignmentAlgorithmPosTEC::doGlobalFit()>: do the global fit ... ";
  
  // number of global parameters
  const int nGlobalParameters = 27;
  // array to store the results of the fit
  float theFittedGlobalParametersPosTEC[nGlobalParameters] = { 0.0 };

  float RRing6 = 84.0;            // radius of Ring 6
  float LaserPhi0 = 0.392699082;  // phi position of the first Laserdiode in Ring 4

  // corrected global Phi + error
  std::vector<float> thePhiCorrected;
  std::vector<float> thePhiCorrectedError;
  std::vector<float> theAbsPhiCorrected;
  std::vector<float> theAbsPhiCorrectedError;

  thePhiCorrected.clear();
  thePhiCorrectedError.clear();
  theAbsPhiCorrected.clear();
  theAbsPhiCorrectedError.clear();

  // do the fit
  fitglo_(theFittedGlobalParametersPosTEC);

  int ep = 1;
  for (int i = 0; i < nGlobalParameters; i++)
    {
      LogDebug("AlignmentAlgorithmPosTEC") << "Global Parameter (TEC+) " << i << " = " << theFittedGlobalParametersPosTEC[i] 
					   << " +/- " << errpar_(&ep);
      ep++;
    }

  int p0 = 0, p1 = 1, p2 = 2, p3 = 3, p4 = 4, p5 = 5;
  int n1 = 1, n2 = 2, n3 = 3, n4 = 4, n5 = 5, n6 =6;

  for (int j = 0; j < 8; ++j)
    {
      // calculate the corrections and the errors
      thePhiCorrected.push_back( theFittedGlobalParametersPosTEC[p0] 
				+ (sin(LaserPhi0)/RRing6) * theFittedGlobalParametersPosTEC[p1]
				- (cos(LaserPhi0)/RRing6) * theFittedGlobalParametersPosTEC[p2]
				- ( theFittedGlobalParametersPosTEC[p3]
				   + (sin(LaserPhi0)/RRing6) * theFittedGlobalParametersPosTEC[p4]
				   - (cos(LaserPhi0)/RRing6) * theFittedGlobalParametersPosTEC[p5] ));
      thePhiCorrectedError.push_back(sqrt(pow(errpar_(&n1),2) 
					  + pow(sin(LaserPhi0)/RRing6,2) * pow(errpar_(&n2),2) 
					  + pow(cos(LaserPhi0)/RRing6,2) * pow(errpar_(&n3),2)
					  + pow(errpar_(&n4),2)
					  + pow(sin(LaserPhi0)/RRing6,2) * pow(errpar_(&n5),2)
					  + pow(cos(LaserPhi0)/RRing6,2) * pow(errpar_(&n6),2) ) );

      // for debugging
      edm::LogInfo("LaserAlignmentAlgorithmPosTEC:Results") << " Fitted relative Correction for TEC+ in Phi[" << j << "] = " << thePhiCorrected.at(j) << " +/- "
						       << thePhiCorrectedError.at(j);


      p3 += 3; p4 += 3; p5 += 3;
      n4 += 3; n5 += 3; n6 += 3;

    }

  for (int j = 0; j < 9; ++j)
    {
      // calculate the correction for each disk (not relative to disk one)
      int e1 = 3*j+1;
      int e2 = 3*j+2;
      int e3 = 3*j+3;
      theAbsPhiCorrected.push_back(-1.0 * theFittedGlobalParametersPosTEC[j*3] 
				   + (sin(LaserPhi0)/RRing6) * theFittedGlobalParametersPosTEC[j*3 + 1]
				   - (cos(LaserPhi0)/RRing6) * theFittedGlobalParametersPosTEC[j*3 + 2]);
      theAbsPhiCorrectedError.push_back(sqrt(pow(errpar_(&e1),2) 
					     + pow(sin(LaserPhi0)/RRing6,2) * pow(errpar_(&e2),2) 
					     + pow(cos(LaserPhi0)/RRing6,2) * pow(errpar_(&e3),2)));
      

      // for debugging
      edm::LogInfo("LaserAlignmentAlgorithmPosTEC:Results") << " Fitted Correction for TEC+ in Phi[" << j << "] = " << theAbsPhiCorrected.at(j) << " +/- "
						       << theAbsPhiCorrectedError.at(j);

    }

  // loop over all discs, access the AlignableTracker to move the discs 
  // according to the calculated alignment corrections
  // AlignableTracker will take care to the propagation of the movements
  // to the lowest level of alignable objects
  const align::Alignables& endcaps = theAlignableTracker->endCaps();
  const Alignable* endcap = endcaps[0];
  if (endcap->globalPosition().z() < 0) endcap = endcaps[1];
  const align::Alignables& disks = endcap->components();

  for (unsigned int i = 0; i < disks.size(); ++i)
    {
      int aPhi = 3*i;
      int aX   = 3*i + 1;
      int aY   = 3*i + 2;
      int ePhi = 3*i + 1;
      int eX = 3*i + 2;
      int eY = 3*i + 3;

      align::GlobalVector translation(-1.0 * theFittedGlobalParametersPosTEC[aX], 
				      -1.0 * theFittedGlobalParametersPosTEC[aY],
				      0.0);
      AlignmentPositionError positionError(errpar_(&eX),errpar_(&eY), 0.0);
      align::RotationType rotationError( Basic3DVector<float>(0.0, 0.0, 1.0), errpar_(&ePhi) );
      Alignable* disk = disks[i];

      disk->move(translation);
      disk->addAlignmentPositionError(positionError);
      disk->rotateAroundGlobalZ(-1.0 * theFittedGlobalParametersPosTEC[aPhi]);
      disk->addAlignmentPositionErrorFromRotation(rotationError);
    }


  // zero initialisation (to avoid problems with the fit of NegTEC!????)
  zerloc_(theGlobalParametersPosTEC, theLocalParametersPosTEC);
  
  edm::LogInfo("LaserAlignmentAlgorithPosTEC") << "<LaserAlignmentAlgorithmPosTEC::doGlobalFit()>: ... done! ";
}

void LaserAlignmentAlgorithmPosTEC::initMillepede(int UnitForIteration)
{
  // number of global and local parameters
  int nGlobalParameters = 27;
  int nLocalParameters = 1;

  // cut parameter for local fit (see page 11 of MILLEPEDE documentation)
  int theCut = 3;
  // verbose output
  int thePrintFlag = 1;
  
  // SIG variable in MILLEPEDE documentation on page 14
  float theUseParameter = 0.0;

  // define 0
  float null = 0.0;

  // the following two parameters are needed for iterations (see page 14)
  int theUnitNumber = 11 + UnitForIteration;        // the unit number for the data file
  float nIterations = 10000.0;   // number of iterations

  // the factors for the constraint of the fit (see page 15)
  float theConstraintFactors[27] = { 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0 };

  // Define dimension parameters etc. (see page 11)
  initgl_(&nGlobalParameters,&nLocalParameters,&theCut,&thePrintFlag);

  // Fix the parameters
  parsig_(&theFirstFixedDiskPosTEC,&theUseParameter);
  parsig_(&theSecondFixedDiskPosTEC,&theUseParameter);

  // initialize the iterations
  initun_(&theUnitNumber,&nIterations);

  // add the constraint
  // in this case, the sum of the rotations around phi is zero
  constf_(theConstraintFactors,&null);

  // zero initialisation
  zerloc_(theGlobalParametersPosTEC, theLocalParametersPosTEC);
}

void LaserAlignmentAlgorithmPosTEC::resetMillepede(int UnitForIteration)
{
  initMillepede(UnitForIteration);
}
