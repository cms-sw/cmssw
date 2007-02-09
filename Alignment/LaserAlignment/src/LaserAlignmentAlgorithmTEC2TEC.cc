/*
 * class to align the Tracker (TEC-TIB-TOB-TEC) with Millepede
 */

#include "Alignment/LaserAlignment/interface/LaserAlignmentAlgorithmTEC2TEC.h"
#include "Alignment/TrackerAlignment/interface/AlignableTrackerEndcap.h"
#include "Alignment/TrackerAlignment/interface/AlignableTrackerHalfBarrel.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/Surface/interface/Surface.h"
#include "DataFormats/TrackingRecHit/interface/AlignmentPositionError.h"

LaserAlignmentAlgorithmTEC2TEC::LaserAlignmentAlgorithmTEC2TEC(edm::ParameterSet const& theConf, int theLaserIteration) 
  : theFirstFixedDiskTEC2TEC(theConf.getUntrackedParameter<int>("FirstFixedParameterTEC2TEC",2)),
    theSecondFixedDiskTEC2TEC(theConf.getUntrackedParameter<int>("SecondFixedParameterTEC2TEC",3)),
    theGlobalParametersTEC2TEC(), theLocalParametersTEC2TEC()
{
  // initialize Millepede
  initMillepede(theLaserIteration);
}

LaserAlignmentAlgorithmTEC2TEC::~LaserAlignmentAlgorithmTEC2TEC()
{
}

void LaserAlignmentAlgorithmTEC2TEC::addLaserBeam(std::vector<double> theMeasurementsPosTEC,
					     std::vector<double> theMeasurementsTOB,
					     std::vector<double> theMeasurementsTIB,
					     std::vector<double> theMeasurementsNegTEC, int LaserBeam, int LaserRing)
{
  LogDebug("LaserAlignmentAlgorithmTEC2TEC") << "<LaserAlignmentAlgorithmTEC2TEC::addLaserBeam()>: adding a new Laser Beam ... ";
  
  // define the "radius" of the Laser ring: for TEC R = 56.4, for TIB = 52.0 and for TOB = 60.8
  // values for TOB are average values!

  float RRing = 0.0;

  if (LaserRing == 1) 
    { RRing = 52.0; }             // radius of TIB 
  else if (LaserRing == 2)
    { RRing = 60.8; }             // radius of TOB
  else if (LaserRing == 4)
    { RRing = 56.4; }             // radius of Ring 4 in TEC

  std::vector<float> LaserPhi;         // phi positions of the Laserdiodes in Ray 4

  // we have 5 (TEC+) + 6 (TIB) + 6 (TOB) + 5 (TEC-) = 22 measurements
  float PhiMeasured[22] = {0.0};   // the measured phi positions
  float PhiErrors[22] = {0.0};     // errors of the measurements

  /* ATTENTION the order of the measurements in the vector<double> theMeasurement is VERY important! */
  // first take the Measurements from TEC+ (first Disc 5, then 4, 3, 2 and finally Disc 1!)
  for (int i = 0; i < 5; i++)
    {
      int n = abs(-4 + i);
      PhiMeasured[i] = float(theMeasurementsPosTEC.at(2*n));
      PhiErrors[i] = float(theMeasurementsPosTEC.at((2*n)+1));
      LogDebug("LaserAlignmentAlgorithmTEC2TEC") << " i = " << i << "\t PhiMeasured[" << i << "] = " << PhiMeasured[i]
					    << "\t PhiError[" << i << "] = " << PhiErrors[i];
    }
  // now take the Measurements from TOB and TIB for the +z part of the Detector
  // the values are sorted by decreasing z position.
  for (int i = 0; i < 3; i++)
    {
      int n = 5 + 2*i;
      PhiMeasured[n] = float(theMeasurementsTOB.at(2*i));
      PhiErrors[n] = float(theMeasurementsTOB.at((2*i)+1));
      PhiMeasured[n+1] = float(theMeasurementsTIB.at(2*i));
      PhiErrors[n+1] = float(theMeasurementsTIB.at((2*i)+1));
      LogDebug("LaserAlignmentAlgorithmTEC2TEC") << " i = " << n << "\t PhiMeasured[" << n << "] = " << PhiMeasured[n]
					    << "\t PhiError[" << n << "] = " << PhiErrors[n] 
					    << "\n i = " << n+1 << "\t PhiMeasured[" << n+1 << "] = " << PhiMeasured[n+1]
					    << "\t PhiError[" << n+1 << "] = " << PhiErrors[n+1];
    }
  // now take the Measurements from TOB and TIB for the -z part of the Detector
  // the values are sorted by decreasing z position. The beam goes now first into TIB!
  for (int i = 3; i < 6; i++)
    {
      int n = 5 + 2*i;
      PhiMeasured[n] = float(theMeasurementsTIB.at(2*i));
      PhiErrors[n] = float(theMeasurementsTIB.at((2*i)+1));
      PhiMeasured[n+1] = float(theMeasurementsTOB.at(2*i));
      PhiErrors[n+1] = float(theMeasurementsTOB.at((2*i)+1));
      LogDebug("LaserAlignmentAlgorithmTEC2TEC") << " i = " << n << "\t PhiMeasured[" << n << "] = " << PhiMeasured[n]
					    << "\t PhiError[" << n << "] = " << PhiErrors[n]
					    << "\n i = " << n+1 << "\t PhiMeasured[" << n+1 << "] = " << PhiMeasured[n+1]
					    << "\t PhiError[" << n+1 << "] = " << PhiErrors[n+1];
    }
  // finally take the Measurements from TEC- (first Disc 1, then 2, 3, 4 and finally Disc 5)
  for (int i = 0; i < 5; i++)
    {
      int n = 17 + i;
      PhiMeasured[n] = float(theMeasurementsNegTEC.at(2*i));
      PhiErrors[n] = float(theMeasurementsNegTEC.at((2*i)+1));
      LogDebug("LaserAlignmentAlgorithmTEC2TEC") << " i = " << n << "\t PhiMeasured[" << n << "] = " << PhiMeasured[n]
					    << "\t PhiError[" << n << "] = " << PhiErrors[n];
    }

  // add the phi positions of the Laserdiodes in Ray 4
  LaserPhi.push_back(0.392699);
  LaserPhi.push_back(1.290297);
  LaserPhi.push_back(1.851296);
  LaserPhi.push_back(2.748894);
  LaserPhi.push_back(3.646491);
  LaserPhi.push_back(4.319690);
  LaserPhi.push_back(5.217288);
  LaserPhi.push_back(5.778286);

  /* ATTENTION here is again the order of the measurements important!!!! 
     How do we want to loop over them? 
     Is the equation - as used for TEC - also usable in this case? (correct RRing???)
  */
  // loop over the discs
  for (int theDisk = 0; theDisk < 22; theDisk++)
    {
      theLocalParametersTEC2TEC[0] = 1.0;
      theGlobalParametersTEC2TEC[3*theDisk] = 1.0;                                     // displacement by dphi
      theGlobalParametersTEC2TEC[(3*theDisk)+1] = -sin(LaserPhi.at(LaserBeam))/RRing;  // displacement by dx
      theGlobalParametersTEC2TEC[(3*theDisk)+2] = cos(LaserPhi.at(LaserBeam))/RRing;   // displacement by dy

      // create the equation for a single measurement
      equloc_(theGlobalParametersTEC2TEC, theLocalParametersTEC2TEC, &PhiMeasured[theDisk], &PhiErrors[theDisk]);
    }
  
  // local fit after one laser beam has been added
  fitloc_();
}

void LaserAlignmentAlgorithmTEC2TEC::doGlobalFit(AlignableTracker * theAlignableTracker)
{
  edm::LogInfo("LaserAlignmentAlgorithmTEC2TEC") << "<LaserAlignmentAlgorithmTEC2TEC::doGlobalFit()>: do the global fit ... ";

  /* ATTENTION here we need the real number of GlobalParameters in the case of TEC-TIB-TOB-TEC Alignment!
     5 * 3 (TEC+) + 6 * 3 (TIB) + 6 * 3 (TOB) + 5 * 3 (TEC-) = 66 */

  // number of global parameters
  const int nGlobalParameters = 66;
  // array to store the results of the fit
  float theFittedGlobalParametersTEC2TEC[nGlobalParameters] = { 0.0 };


  /* ATTENTION which radius do we actual need to calculate the final corrections? */
  float RRing4 = 56.4;            // radius of Ring 4 in TEC

  float LaserPhi0 = 0.392699082;  // phi position of the first Laserdiode in Ring 4

  // we have 5 (TEC+) + 6 (TIB) + 6 (TOB) + 5 (TEC-) = 22 corrections
  // corrected global Phi + error
  std::vector<float> thePhiCorrected;
  std::vector<float> thePhiCorrectedError;

  thePhiCorrected.clear();
  thePhiCorrectedError.clear();

  // do the fit
  fitglo_(theFittedGlobalParametersTEC2TEC);

  int ep = 1;
  for (int i = 0; i < nGlobalParameters; i++)
    {
      LogDebug("LaserAlignmentAlgorithmTEC2TEC") << "Global Parameter (TEC-TIB-TOB-TEC) " << i << " = " << theFittedGlobalParametersTEC2TEC[i] 
					    << " +/- " << errpar_(&ep);
      ep++;
    }

  int p0 = 0, p1 = 1, p2 = 2, p3 = 3, p4 = 4, p5 = 5;
  int n1 = 1, n2 = 2, n3 = 3, n4 = 4, n5 = 5, n6 =6;

  for (int j = 0; j < 21; ++j) /* ATTENTION is this okay in this case? 25.10. I guess so, 22 - 1 corrections to calculate ... */
    {
      thePhiCorrected.push_back( theFittedGlobalParametersTEC2TEC[p0] 
				+ (sin(LaserPhi0)/RRing4) * theFittedGlobalParametersTEC2TEC[p1]
				- (cos(LaserPhi0)/RRing4) * theFittedGlobalParametersTEC2TEC[p2]
				- ( theFittedGlobalParametersTEC2TEC[p3]
				    + (sin(LaserPhi0)/RRing4) * theFittedGlobalParametersTEC2TEC[p4]
				    - (cos(LaserPhi0)/RRing4) * theFittedGlobalParametersTEC2TEC[p5] ) );
      thePhiCorrectedError.push_back(sqrt( pow(errpar_(&n1),2) 
					   + pow(sin(LaserPhi0)/RRing4,2) * pow(errpar_(&n2),2)
					   + pow(cos(LaserPhi0)/RRing4,2) * pow(errpar_(&n3),2)
					   + pow(errpar_(&n4),2)
					   + pow(sin(LaserPhi0)/RRing4,2) * pow(errpar_(&n5),2)
					   + pow(cos(LaserPhi0)/RRing4,2) * pow(errpar_(&n6),2) ) );


      edm::LogInfo("LaserAlignmentAlgorithmTEC2TEC:Results") << " Fitted Correction for TEC-TIB-TOB-TEC in Phi[" << j << "] = " 
							<< thePhiCorrected.at(j) << " +/- " << thePhiCorrectedError.at(j);

      p3 += 3; p4 += 3; p5 += 3;
      n4 += 3; n5 += 3; n6 += 3;

    }
  
  // loop over all discs, access the AlignableTracker to move the discs 
  // according to the calculated alignment corrections
  // AlignableTracker will take care to the propagation of the movements
  // to the lowest level of alignable objects
  // first TEC+ disc 5 -> 1
  for (int i = 0; i < 5; ++i)
    {
      int aPhi = 3*i;
      int aX   = 3*i + 1;
      int aY   = 3*i + 2;
      int ePhi = 3*i + 1;
      int eX = 3*i + 2;
      int eY = 3*i + 3;
      
      // TEC+ discs ... consider right order!!!
      GlobalVector translation(-1.0 * theFittedGlobalParametersTEC2TEC[aX],
		   -1.0 * theFittedGlobalParametersTEC2TEC[aY],
		   0.0);
			AlignmentPositionError positionError(errpar_(&eX), errpar_(&eY), 0.0);
			Surface::RotationType rotationError( Basic3DVector<float>(0.0, 0.0, 1.0), errpar_(&ePhi) );
		
      theAlignableTracker->endCap(0).layer(abs(-4 + i)).move(translation);
	   	theAlignableTracker->endCap(0).layer(abs(-4 + i)).addAlignmentPositionError(positionError);
		  theAlignableTracker->endCap(0).layer(abs(-4 + i)).rotateAroundGlobalZ(-1.0 * theFittedGlobalParametersTEC2TEC[aPhi]);
     	theAlignableTracker->endCap(0).layer(abs(-4 + i)).addAlignmentPositionErrorFromRotation(rotationError);
    }
		
	int ePhiTOB1 = 16, eXTOB1 = 17, eYTOB1 = 18;
	int ePhiTIB1 = 19, eXTIB1 = 20, eYTIB1 = 21;
	int ePhiTIB2 = 31, eXTIB2 = 32, eYTIB2 = 33;
	int ePhiTOB2 = 34, eXTOB2 = 35, eYTOB2 = 36;
	
  GlobalVector translationTOB1(-1.0 * theFittedGlobalParametersTEC2TEC[16],
			       -1.0 * theFittedGlobalParametersTEC2TEC[17],
			       0.0);
	AlignmentPositionError positionErrorTOB1(errpar_(&eXTOB1), errpar_(&eYTOB1), 0.0);
	Surface::RotationType rotationErrorTOB1( Basic3DVector<float>(0.0,0.0,1.0), errpar_(&ePhiTOB1) );

  GlobalVector translationTIB1(-1.0 * theFittedGlobalParametersTEC2TEC[19],
			       -1.0 * theFittedGlobalParametersTEC2TEC[20],
			       0.0);
	AlignmentPositionError positionErrorTIB1(errpar_(&eXTIB1), errpar_(&eYTIB1), 0.0);
	Surface::RotationType rotationErrorTIB1( Basic3DVector<float>(0.0,0.0,1.0), errpar_(&ePhiTIB1) );

  GlobalVector translationTIB2(-1.0 * theFittedGlobalParametersTEC2TEC[31],
			       -1.0 * theFittedGlobalParametersTEC2TEC[32],
			       0.0);
	AlignmentPositionError positionErrorTIB2(errpar_(&eXTIB2), errpar_(&eYTIB2), 0.0);
	Surface::RotationType rotationErrorTIB2( Basic3DVector<float>(0.0,0.0,1.0), errpar_(&ePhiTIB2) );

  GlobalVector translationTOB2(-1.0 * theFittedGlobalParametersTEC2TEC[34],
			       -1.0 * theFittedGlobalParametersTEC2TEC[35],
			       0.0);
	AlignmentPositionError positionErrorTOB2(errpar_(&eXTOB2), errpar_(&eYTOB2), 0.0);
	Surface::RotationType rotationErrorTOB2( Basic3DVector<float>(0.0,0.0,1.0), errpar_(&ePhiTOB2) );

  // TOB and TIB (no loop needed: 3 corrections for TOB+,TIB+,TIB-,TOB-; should
  // be the same value each)
  // TOB+
  theAlignableTracker->outerHalfBarrel(0).move(translationTOB1);
	theAlignableTracker->outerHalfBarrel(0).addAlignmentPositionError(positionErrorTOB1);
  theAlignableTracker->outerHalfBarrel(0).rotateAroundGlobalZ(-1.0 * theFittedGlobalParametersTEC2TEC[15]);
	theAlignableTracker->outerHalfBarrel(0).addAlignmentPositionErrorFromRotation(rotationErrorTOB1);

  // TIB+ 
  theAlignableTracker->innerHalfBarrel(0).move(translationTIB1);
	theAlignableTracker->innerHalfBarrel(0).addAlignmentPositionError(positionErrorTIB1);
  theAlignableTracker->innerHalfBarrel(0).rotateAroundGlobalZ(-1.0 * theFittedGlobalParametersTEC2TEC[18]);
	theAlignableTracker->innerHalfBarrel(0).addAlignmentPositionErrorFromRotation(rotationErrorTIB1);

	// TOB-
  theAlignableTracker->outerHalfBarrel(1).move(translationTOB2);
	theAlignableTracker->outerHalfBarrel(1).addAlignmentPositionError(positionErrorTOB2);
  theAlignableTracker->outerHalfBarrel(1).rotateAroundGlobalZ(-1.0 * theFittedGlobalParametersTEC2TEC[33]);
	theAlignableTracker->outerHalfBarrel(1).addAlignmentPositionErrorFromRotation(rotationErrorTOB2);
	
  // TIB-
  theAlignableTracker->innerHalfBarrel(1).move(translationTIB2);
	theAlignableTracker->innerHalfBarrel(1).addAlignmentPositionError(positionErrorTIB2);
  theAlignableTracker->innerHalfBarrel(1).rotateAroundGlobalZ(-1.0 * theFittedGlobalParametersTEC2TEC[30]);
	theAlignableTracker->innerHalfBarrel(1).addAlignmentPositionErrorFromRotation(rotationErrorTIB2);

  // TEC- disc 1 -> 5
  // loop over all discs, access the AlignableTracker to move the discs 
  // according to the calculated alignment corrections
  // AlignableTracker will take care to the propagation of the movements
  // to the lowest level of alignable objects
  for (int i = 0; i < 5; ++i)
    {
      int aPhi = 3*i + 51;
      int aX   = 3*i + 52;
      int aY   = 3*i + 53;
      int ePhi = 3*i + 52;
      int eX = 3*i + 53;
      int eY = 3*i + 54;
      
      GlobalVector translation(-1.0 * theFittedGlobalParametersTEC2TEC[aX],
			       -1.0 * theFittedGlobalParametersTEC2TEC[aY],
			       0.0);
			AlignmentPositionError positionError(errpar_(&eX), errpar_(&eY), 0.0);
			Surface::RotationType rotationError( Basic3DVector<float>(0.0, 0.0, 1.0), errpar_(&ePhi) );

      theAlignableTracker->endCap(1).layer(i).move(translation);
      theAlignableTracker->endCap(1).layer(i).addAlignmentPositionError(positionError);
      theAlignableTracker->endCap(1).layer(i).rotateAroundGlobalZ(-1.0 * theFittedGlobalParametersTEC2TEC[aPhi]);
			theAlignableTracker->endCap(1).layer(i).addAlignmentPositionErrorFromRotation(rotationError);
    }

  // zero initialisation (to avoid problems with the next Millepede fit!????)
  zerloc_(theGlobalParametersTEC2TEC, theLocalParametersTEC2TEC);

  edm::LogInfo("LaserAlignmentAlgorithmTEC2TEC") << "<LaserAlignmentAlgorithmTEC2TEC::doGlobalFit()>: ... done! ";
}

void LaserAlignmentAlgorithmTEC2TEC::initMillepede(int UnitForIteration)
{
  // number of global and local parameters
  int nGlobalParameters = 66;
  int nLocalParameters = 1; /* ATTENTION is this value still appropriate? */

  // cut parameter for local fit (see page 11 of MILLEPEDE documentation)
  int theCut = 3;
  // verbose output
  int thePrintFlag = 1;
  
  // SIG variable in MILLEPEDE documentation on page 14
  float theUseParameter = 0.0;

  // define 0
  float null = 0.0;

  // the following two parameters are needed for iterations (see page 14)
  int theUnitNumber = 71 + UnitForIteration;        // the unit number for the data file
  float nIterations = 10000.0;   // number of iterations

  /* ATTENTION what constraint can we use for the TEC-TIB-TOB-TEC alignment? */
  // the factors for the constraint of the fit (see page 15)
  float theConstraintFactors[66] = { 1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, 
 				     1.0, 0.0, 0.0, };

  // Define dimension parameters etc. (see page 11)
  initgl_(&nGlobalParameters,&nLocalParameters,&theCut,&thePrintFlag);

  /* ATTENTION how do we choose the fixed reference system for TEC-TIB-TOB-TEC alignment!? */
  // Fix the parameters
  parsig_(&theFirstFixedDiskTEC2TEC,&theUseParameter);
  parsig_(&theSecondFixedDiskTEC2TEC,&theUseParameter);

  // initialize the iterations
  initun_(&theUnitNumber,&nIterations);

  // add the constraint
  // in this case, the sum of the rotations around phi is zero
  constf_(theConstraintFactors,&null);

  // zero initialisation
  zerloc_(theGlobalParametersTEC2TEC, theLocalParametersTEC2TEC);
}

void LaserAlignmentAlgorithmTEC2TEC::resetMillepede(int UnitForIteration)
{
  initMillepede(UnitForIteration);
}
