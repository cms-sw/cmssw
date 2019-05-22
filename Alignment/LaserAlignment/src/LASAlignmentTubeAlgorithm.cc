
#include "Alignment/LaserAlignment/interface/LASAlignmentTubeAlgorithm.h"

///
///
///
LASAlignmentTubeAlgorithm::LASAlignmentTubeAlgorithm() {}

///
///
///
LASBarrelAlignmentParameterSet LASAlignmentTubeAlgorithm::CalculateParameters(
    LASGlobalData<LASCoordinateSet>& measuredCoordinates, LASGlobalData<LASCoordinateSet>& nominalCoordinates) {
  std::cout << " [LASAlignmentTubeAlgorithm::CalculateParameters] -- Starting." << std::endl;

  // for debugging only
  //######################################################################################
  //ReadMisalignmentFromFile( "misalign-var.txt", measuredCoordinates, nominalCoordinates );
  //######################################################################################

  // loop object
  LASGlobalLoop globalLoop;
  int det, beam, disk, pos;

  // phi positions of the AT beams in rad
  const double phiPositions[8] = {0.392699, 1.289799, 1.851794, 2.748894, 3.645995, 4.319690, 5.216791, 5.778784};
  std::vector<double> beamPhiPositions(8, 0.);
  for (beam = 0; beam < 8; ++beam)
    beamPhiPositions.at(beam) = phiPositions[beam];

  // the radii of the alignment tube beams for each halfbarrel.
  // the halfbarrels 1-6 are (see TkLasATModel TWiki): TEC+, TEC-, TIB+, TIB-. TOB+, TOB-
  // in TIB/TOB modules these radii differ from the beam radius..
  // ..due to the radial offsets (after the semitransparent mirrors)
  const double radii[6] = {564., 564., 514., 514., 600., 600.};
  std::vector<double> beamRadii(6, 0.);
  for (int aHalfbarrel = 0; aHalfbarrel < 6; ++aHalfbarrel)
    beamRadii.at(aHalfbarrel) = radii[aHalfbarrel];

  // the z positions of the halfbarrel_end_faces / outer_TEC_disks (in mm);
  // parameters are: det, side(0=+/1=-), z(0=lowerZ/1=higherZ). TECs have no side (use side = 0)
  std::vector<std::vector<std::vector<double> > > endFaceZPositions(
      4, std::vector<std::vector<double> >(2, std::vector<double>(2, 0.)));
  endFaceZPositions.at(0).at(0).at(0) = 1322.5;   // TEC+, *, disk1 ///
  endFaceZPositions.at(0).at(0).at(1) = 2667.5;   // TEC+, *, disk9 /// SIDE INFORMATION
  endFaceZPositions.at(1).at(0).at(0) = -2667.5;  // TEC-, *, disk9 /// MEANINGLESS FOR TEC -> USE .at(0)!
  endFaceZPositions.at(1).at(0).at(1) = -1322.5;  // TEC-, *, disk1 ///
  endFaceZPositions.at(2).at(1).at(0) = -700.;    // TIB,  -, outer
  endFaceZPositions.at(2).at(1).at(1) = -300.;    // TIB,  -, inner
  endFaceZPositions.at(2).at(0).at(0) = 300.;     // TIB,  +, inner
  endFaceZPositions.at(2).at(0).at(1) = 700.;     // TIB,  +, outer
  endFaceZPositions.at(3).at(1).at(0) = -1090.;   // TOB,  -, outer
  endFaceZPositions.at(3).at(1).at(1) = -300.;    // TOB,  -, inner
  endFaceZPositions.at(3).at(0).at(0) = 300.;     // TOB,  +, inner
  endFaceZPositions.at(3).at(0).at(1) = 1090.;    // TOB,  +, outer

  // reduced z positions of the beam spots ( z'_{k,j}, z"_{k,j} )
  double detReducedZ[2] = {0., 0.};
  // reduced beam splitter positions ( zt'_{k,j}, zt"_{k,j} )
  double beamReducedZ[2] = {0., 0.};

  // the z positions of the virtual planes at which the beam parameters are measured
  std::vector<double> disk9EndFaceZPositions(2, 0.);
  disk9EndFaceZPositions.at(0) = -2667.5;  // TEC- disk9
  disk9EndFaceZPositions.at(1) = 2667.5;   // TEC+ disk9

  // define sums over measured values to "simplify" the beam parameter formulas

  // all these have 6 entries, one for each halfbarrel (TEC+,TEC-,TIB+,TIB-,TOB+,TOB-)
  std::vector<double> sumOverPhiZPrime(6, 0.);
  std::vector<double> sumOverPhiZPrimePrime(6, 0.);
  std::vector<double> sumOverPhiZPrimeSinTheta(6, 0.);
  std::vector<double> sumOverPhiZPrimePrimeSinTheta(6, 0.);
  std::vector<double> sumOverPhiZPrimeCosTheta(6, 0.);
  std::vector<double> sumOverPhiZPrimePrimeCosTheta(6, 0.);

  // these have 8 entries, one for each beam
  std::vector<double> sumOverPhiZTPrime(8, 0.);
  std::vector<double> sumOverPhiZTPrimePrime(8, 0.);

  // define sums over nominal values

  // all these have 6 entries, one for each halfbarrel (TEC+,TEC-,TIB+,TIB-,TOB+,TOB-)
  std::vector<double> sumOverZPrimeSquared(6, 0.);
  std::vector<double> sumOverZPrimePrimeSquared(6, 0.);
  std::vector<double> sumOverZPrimeZPrimePrime(6, 0.);
  std::vector<double> sumOverZPrimeZTPrime(6, 0.);
  std::vector<double> sumOverZPrimeZTPrimePrime(6, 0.);
  std::vector<double> sumOverZPrimePrimeZTPrime(6, 0.);
  std::vector<double> sumOverZPrimePrimeZTPrimePrime(6, 0.);

  // all these are scalars
  double sumOverZTPrimeSquared = 0.;
  double sumOverZTPrimePrimeSquared = 0.;
  double sumOverZTPrimeZTPrimePrime = 0.;

  // now calculate them for TIBTOB
  det = 2;
  beam = 0;
  pos = 0;
  do {
    // define the side: 0 for TIB+/TOB+ and 1 for TIB-/TOB-
    const int theSide = pos < 3 ? 0 : 1;

    // define the halfbarrel number from det/side
    const int halfbarrel = det == 2 ? det + theSide : det + 1 + theSide;  // TIB:TOB

    // this is the path the beam has to travel radially after being reflected
    // by the AT mirrors (TIB:50mm, TOB:36mm) -> used for beam parameters
    const double radialOffset = det == 2 ? 50. : 36.;

    // reduced module's z position with respect to the subdetector endfaces (zPrime, zPrimePrime)
    detReducedZ[0] = measuredCoordinates.GetTIBTOBEntry(det, beam, pos).GetZ() -
                     endFaceZPositions.at(det).at(theSide).at(0);  // = zPrime
    detReducedZ[0] /= (endFaceZPositions.at(det).at(theSide).at(1) - endFaceZPositions.at(det).at(theSide).at(0));
    detReducedZ[1] = endFaceZPositions.at(det).at(theSide).at(1) -
                     measuredCoordinates.GetTIBTOBEntry(det, beam, pos).GetZ();  // = zPrimePrime
    detReducedZ[1] /= (endFaceZPositions.at(det).at(theSide).at(1) - endFaceZPositions.at(det).at(theSide).at(0));

    // reduced module's z position with respect to the tec disks +-9 (for the beam parameters)
    beamReducedZ[0] = (measuredCoordinates.GetTIBTOBEntry(det, beam, pos).GetZ() - radialOffset) -
                      disk9EndFaceZPositions.at(0);  // = ZTPrime
    beamReducedZ[0] /= (disk9EndFaceZPositions.at(1) - disk9EndFaceZPositions.at(0));
    beamReducedZ[1] = disk9EndFaceZPositions.at(1) -
                      (measuredCoordinates.GetTIBTOBEntry(det, beam, pos).GetZ() - radialOffset);  // ZTPrimePrime
    beamReducedZ[1] /= (disk9EndFaceZPositions.at(1) - disk9EndFaceZPositions.at(0));

    // residual in phi (in the formulas this corresponds to y_ik/R)
    const double phiResidual = measuredCoordinates.GetTIBTOBEntry(det, beam, pos).GetPhi() -
                               nominalCoordinates.GetTIBTOBEntry(det, beam, pos).GetPhi();

    // sums over measured values
    sumOverPhiZPrime.at(halfbarrel) += phiResidual * detReducedZ[0];
    sumOverPhiZPrimePrime.at(halfbarrel) += phiResidual * detReducedZ[1];
    sumOverPhiZPrimeSinTheta.at(halfbarrel) += phiResidual * detReducedZ[0] * sin(beamPhiPositions.at(beam));
    sumOverPhiZPrimePrimeSinTheta.at(halfbarrel) += phiResidual * detReducedZ[1] * sin(beamPhiPositions.at(beam));
    sumOverPhiZPrimeCosTheta.at(halfbarrel) += phiResidual * detReducedZ[0] * cos(beamPhiPositions.at(beam));
    sumOverPhiZPrimePrimeCosTheta.at(halfbarrel) += phiResidual * detReducedZ[1] * cos(beamPhiPositions.at(beam));

    sumOverPhiZTPrime.at(beam) += phiResidual * beamReducedZ[0];  // note the index change here..
    sumOverPhiZTPrimePrime.at(beam) += phiResidual * beamReducedZ[1];

    // sums over nominal values
    sumOverZPrimeSquared.at(halfbarrel) += pow(detReducedZ[0], 2) / 8.;  // these are defined beam-wise, so: / 8.
    sumOverZPrimePrimeSquared.at(halfbarrel) += pow(detReducedZ[1], 2) / 8.;
    sumOverZPrimeZPrimePrime.at(halfbarrel) += detReducedZ[0] * detReducedZ[1] / 8.;
    sumOverZPrimeZTPrime.at(halfbarrel) += detReducedZ[0] * beamReducedZ[0] / 8.;
    sumOverZPrimeZTPrimePrime.at(halfbarrel) += detReducedZ[0] * beamReducedZ[1] / 8.;
    sumOverZPrimePrimeZTPrime.at(halfbarrel) += detReducedZ[1] * beamReducedZ[0] / 8.;
    sumOverZPrimePrimeZTPrimePrime.at(halfbarrel) += detReducedZ[1] * beamReducedZ[1] / 8.;

    sumOverZTPrimeSquared += pow(beamReducedZ[0], 2) / 8.;
    sumOverZTPrimePrimeSquared += pow(beamReducedZ[1], 2) / 8.;
    sumOverZTPrimeZTPrimePrime += beamReducedZ[0] * beamReducedZ[1] / 8.;

  } while (globalLoop.TIBTOBLoop(det, beam, pos));

  // now for TEC2TEC
  det = 0;
  beam = 0;
  disk = 0;
  do {
    // for the tec, the halfbarrel numbers are equal to the det numbers...
    const int halfbarrel = det;

    // ...so there's no side distinction for the TEC
    const int theSide = 0;

    // also, there's no radial offset for the TEC
    const double radialOffset = 0.;

    // reduced module's z position with respect to the subdetector endfaces (zPrime, zPrimePrime)
    detReducedZ[0] = measuredCoordinates.GetTEC2TECEntry(det, beam, disk).GetZ() -
                     endFaceZPositions.at(det).at(theSide).at(0);  // = zPrime
    detReducedZ[0] /= (endFaceZPositions.at(det).at(theSide).at(1) - endFaceZPositions.at(det).at(theSide).at(0));
    detReducedZ[1] = endFaceZPositions.at(det).at(theSide).at(1) -
                     measuredCoordinates.GetTEC2TECEntry(det, beam, disk).GetZ();  // = zPrimePrime
    detReducedZ[1] /= (endFaceZPositions.at(det).at(theSide).at(1) - endFaceZPositions.at(det).at(theSide).at(0));

    // reduced module's z position with respect to the tec disks +-9 (for the beam parameters)
    beamReducedZ[0] = (measuredCoordinates.GetTEC2TECEntry(det, beam, disk).GetZ() - radialOffset) -
                      disk9EndFaceZPositions.at(0);  // = ZTPrime
    beamReducedZ[0] /= (disk9EndFaceZPositions.at(1) - disk9EndFaceZPositions.at(0));
    beamReducedZ[1] = disk9EndFaceZPositions.at(1) -
                      (measuredCoordinates.GetTEC2TECEntry(det, beam, disk).GetZ() - radialOffset);  // ZTPrimePrime
    beamReducedZ[1] /= (disk9EndFaceZPositions.at(1) - disk9EndFaceZPositions.at(0));

    // residual in phi (in the formulas this corresponds to y_ik/R)
    const double phiResidual = measuredCoordinates.GetTEC2TECEntry(det, beam, disk).GetPhi() -
                               nominalCoordinates.GetTEC2TECEntry(det, beam, disk).GetPhi();

    // sums over measured values
    sumOverPhiZPrime.at(halfbarrel) += phiResidual * detReducedZ[0];
    sumOverPhiZPrimePrime.at(halfbarrel) += phiResidual * detReducedZ[1];
    sumOverPhiZPrimeSinTheta.at(halfbarrel) += phiResidual * detReducedZ[0] * sin(beamPhiPositions.at(beam));
    sumOverPhiZPrimePrimeSinTheta.at(halfbarrel) += phiResidual * detReducedZ[1] * sin(beamPhiPositions.at(beam));
    sumOverPhiZPrimeCosTheta.at(halfbarrel) += phiResidual * detReducedZ[0] * cos(beamPhiPositions.at(beam));
    sumOverPhiZPrimePrimeCosTheta.at(halfbarrel) += phiResidual * detReducedZ[1] * cos(beamPhiPositions.at(beam));

    sumOverPhiZTPrime.at(beam) += phiResidual * beamReducedZ[0];  // note the index change here..
    sumOverPhiZTPrimePrime.at(beam) += phiResidual * beamReducedZ[1];

    // sums over nominal values
    sumOverZPrimeSquared.at(halfbarrel) += pow(detReducedZ[0], 2) / 8.;  // these are defined beam-wise, so: / 8.
    sumOverZPrimePrimeSquared.at(halfbarrel) += pow(detReducedZ[1], 2) / 8.;
    sumOverZPrimeZPrimePrime.at(halfbarrel) += detReducedZ[0] * detReducedZ[1] / 8.;
    sumOverZPrimeZTPrime.at(halfbarrel) += detReducedZ[0] * beamReducedZ[0] / 8.;
    sumOverZPrimeZTPrimePrime.at(halfbarrel) += detReducedZ[0] * beamReducedZ[1] / 8.;
    sumOverZPrimePrimeZTPrime.at(halfbarrel) += detReducedZ[1] * beamReducedZ[0] / 8.;
    sumOverZPrimePrimeZTPrimePrime.at(halfbarrel) += detReducedZ[1] * beamReducedZ[1] / 8.;

    sumOverZTPrimeSquared += pow(beamReducedZ[0], 2) / 8.;
    sumOverZTPrimePrimeSquared += pow(beamReducedZ[1], 2) / 8.;
    sumOverZTPrimeZTPrimePrime += beamReducedZ[0] * beamReducedZ[1] / 8.;

  } while (globalLoop.TEC2TECLoop(det, beam, disk));

  // more "simplification" terms...
  // these here are functions of theta and can be calculated directly
  double sumOverSinTheta = 0.;
  double sumOverCosTheta = 0.;
  double sumOverSinThetaSquared = 0.;
  double sumOverCosThetaSquared = 0.;
  double sumOverCosThetaSinTheta = 0.;
  double mixedTrigonometricTerm = 0.;

  for (beam = 0; beam < 8; ++beam) {
    sumOverSinTheta += sin(beamPhiPositions.at(beam));
    sumOverCosTheta += cos(beamPhiPositions.at(beam));
    sumOverSinThetaSquared += pow(sin(beamPhiPositions.at(beam)), 2);
    sumOverCosThetaSquared += pow(cos(beamPhiPositions.at(beam)), 2);
    sumOverCosThetaSinTheta += cos(beamPhiPositions.at(beam)) * sin(beamPhiPositions.at(beam));
  }

  mixedTrigonometricTerm = 8. * (sumOverCosThetaSquared * sumOverSinThetaSquared - pow(sumOverCosThetaSinTheta, 2)) -
                           pow(sumOverCosTheta, 2) * sumOverSinThetaSquared -
                           pow(sumOverSinTheta, 2) * sumOverCosThetaSquared +
                           2. * sumOverCosTheta * sumOverSinTheta * sumOverCosThetaSinTheta;

  // even more shortcuts before we can calculate the parameters
  double beamDenominator =
      (pow(sumOverZTPrimeZTPrimePrime, 2) - sumOverZTPrimeSquared * sumOverZTPrimePrimeSquared) * beamRadii.at(0);
  std::vector<double> alignmentDenominator(6, 0.);
  std::vector<double> termA(6, 0.), termB(6, 0.), termC(6, 0.), termD(6, 0.);
  for (unsigned int aHalfbarrel = 0; aHalfbarrel < 6; ++aHalfbarrel) {
    alignmentDenominator.at(aHalfbarrel) =
        (pow(sumOverZPrimeZPrimePrime.at(aHalfbarrel), 2) -
         sumOverZPrimeSquared.at(aHalfbarrel) * sumOverZPrimePrimeSquared.at(aHalfbarrel)) *
        mixedTrigonometricTerm;
    termA.at(aHalfbarrel) = sumOverZPrimeZTPrime.at(aHalfbarrel) * sumOverZTPrimeZTPrimePrime -
                            sumOverZPrimeZTPrimePrime.at(aHalfbarrel) * sumOverZTPrimeSquared;
    termB.at(aHalfbarrel) = sumOverZPrimePrimeZTPrime.at(aHalfbarrel) * sumOverZTPrimeZTPrimePrime -
                            sumOverZPrimePrimeZTPrimePrime.at(aHalfbarrel) * sumOverZTPrimeSquared;
    termC.at(aHalfbarrel) = sumOverZPrimeZTPrimePrime.at(aHalfbarrel) * sumOverZTPrimeZTPrimePrime -
                            sumOverZPrimeZTPrime.at(aHalfbarrel) * sumOverZTPrimePrimeSquared;
    termD.at(aHalfbarrel) = sumOverZPrimePrimeZTPrimePrime.at(aHalfbarrel) * sumOverZTPrimeZTPrimePrime -
                            sumOverZPrimePrimeZTPrime.at(aHalfbarrel) * sumOverZTPrimePrimeSquared;
  }

  // have eight alignment tube beams..
  const int numberOfBeams = 8;

  // that's all for preparation, now it gets ugly:
  // calculate the alignment parameters
  LASBarrelAlignmentParameterSet theResult;

  // can do this in one go for all halfbarrels
  for (int aHalfbarrel = 0; aHalfbarrel < 6; ++aHalfbarrel) {
    // no errors yet

    // rotation angles of the lower z endfaces (in rad)
    theResult.GetParameter(aHalfbarrel, 0, 0).first =
        (sumOverPhiZPrime.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosThetaSquared *
             sumOverSinThetaSquared -
         sumOverPhiZPrimePrime.at(aHalfbarrel) * sumOverZPrimeSquared.at(aHalfbarrel) * sumOverCosThetaSquared *
             sumOverSinThetaSquared -
         sumOverPhiZPrimeCosTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosTheta *
             sumOverSinThetaSquared +
         sumOverPhiZPrimePrimeCosTheta.at(aHalfbarrel) * sumOverZPrimeSquared.at(aHalfbarrel) * sumOverCosTheta *
             sumOverSinThetaSquared +
         sumOverPhiZPrimeCosTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosThetaSinTheta *
             sumOverSinTheta -
         sumOverPhiZPrimePrimeCosTheta.at(aHalfbarrel) * sumOverZPrimeSquared.at(aHalfbarrel) *
             sumOverCosThetaSinTheta * sumOverSinTheta -
         sumOverPhiZPrimeSinTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosThetaSquared *
             sumOverSinTheta +
         sumOverPhiZPrimePrimeSinTheta.at(aHalfbarrel) * sumOverZPrimeSquared.at(aHalfbarrel) * sumOverCosThetaSquared *
             sumOverSinTheta -
         sumOverPhiZPrime.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * pow(sumOverCosThetaSinTheta, 2) +
         sumOverPhiZPrimePrime.at(aHalfbarrel) * sumOverZPrimeSquared.at(aHalfbarrel) *
             pow(sumOverCosThetaSinTheta, 2) +
         sumOverPhiZPrimeSinTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosTheta *
             sumOverCosThetaSinTheta -
         sumOverPhiZPrimePrimeSinTheta.at(aHalfbarrel) * sumOverZPrimeSquared.at(aHalfbarrel) * sumOverCosTheta *
             sumOverCosThetaSinTheta) /
        alignmentDenominator.at(aHalfbarrel);

    // rotation angles of the upper z endfaces (in rad)
    theResult.GetParameter(aHalfbarrel, 1, 0).first =
        (sumOverPhiZPrimePrime.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosThetaSquared *
             sumOverSinThetaSquared -
         sumOverPhiZPrime.at(aHalfbarrel) * sumOverZPrimePrimeSquared.at(aHalfbarrel) * sumOverCosThetaSquared *
             sumOverSinThetaSquared -
         sumOverPhiZPrimePrimeCosTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosTheta *
             sumOverSinThetaSquared +
         sumOverPhiZPrimeCosTheta.at(aHalfbarrel) * sumOverZPrimePrimeSquared.at(aHalfbarrel) * sumOverCosTheta *
             sumOverSinThetaSquared +
         sumOverPhiZPrimePrimeCosTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) *
             sumOverCosThetaSinTheta * sumOverSinTheta -
         sumOverPhiZPrimeCosTheta.at(aHalfbarrel) * sumOverZPrimePrimeSquared.at(aHalfbarrel) *
             sumOverCosThetaSinTheta * sumOverSinTheta -
         sumOverPhiZPrimePrimeSinTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) *
             sumOverCosThetaSquared * sumOverSinTheta +
         sumOverPhiZPrimeSinTheta.at(aHalfbarrel) * sumOverZPrimePrimeSquared.at(aHalfbarrel) * sumOverCosThetaSquared *
             sumOverSinTheta -
         sumOverPhiZPrimePrime.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosThetaSinTheta *
             sumOverCosThetaSinTheta +
         sumOverPhiZPrime.at(aHalfbarrel) * sumOverZPrimePrimeSquared.at(aHalfbarrel) * sumOverCosThetaSinTheta *
             sumOverCosThetaSinTheta +
         sumOverPhiZPrimePrimeSinTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosTheta *
             sumOverCosThetaSinTheta -
         sumOverPhiZPrimeSinTheta.at(aHalfbarrel) * sumOverZPrimePrimeSquared.at(aHalfbarrel) * sumOverCosTheta *
             sumOverCosThetaSinTheta) /
        alignmentDenominator.at(aHalfbarrel);

    // x deviations of the lower z endfaces (in mm)
    theResult.GetParameter(aHalfbarrel, 0, 1).first =
        -1. *
        (sumOverPhiZPrime.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosThetaSquared *
             sumOverSinTheta -
         sumOverPhiZPrimePrime.at(aHalfbarrel) * sumOverZPrimeSquared.at(aHalfbarrel) * sumOverCosThetaSquared *
             sumOverSinTheta -
         sumOverPhiZPrimeCosTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosTheta *
             sumOverSinTheta +
         sumOverPhiZPrimePrimeCosTheta.at(aHalfbarrel) * sumOverZPrimeSquared.at(aHalfbarrel) * sumOverCosTheta *
             sumOverSinTheta -
         sumOverPhiZPrime.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosTheta *
             sumOverCosThetaSinTheta +
         sumOverPhiZPrimePrime.at(aHalfbarrel) * sumOverZPrimeSquared.at(aHalfbarrel) * sumOverCosTheta *
             sumOverCosThetaSinTheta +
         sumOverPhiZPrimeCosTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * numberOfBeams *
             sumOverCosThetaSinTheta -
         sumOverPhiZPrimePrimeCosTheta.at(aHalfbarrel) * sumOverZPrimeSquared.at(aHalfbarrel) * numberOfBeams *
             sumOverCosThetaSinTheta -
         sumOverPhiZPrimeSinTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * numberOfBeams *
             sumOverCosThetaSquared +
         sumOverPhiZPrimePrimeSinTheta.at(aHalfbarrel) * sumOverZPrimeSquared.at(aHalfbarrel) * numberOfBeams *
             sumOverCosThetaSquared +
         sumOverPhiZPrimeSinTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosTheta *
             sumOverCosTheta -
         sumOverPhiZPrimePrimeSinTheta.at(aHalfbarrel) * sumOverZPrimeSquared.at(aHalfbarrel) * sumOverCosTheta *
             sumOverCosTheta) /
        alignmentDenominator.at(aHalfbarrel) * beamRadii.at(aHalfbarrel);

    // x deviations of the upper z endfaces (in mm)
    theResult.GetParameter(aHalfbarrel, 1, 1).first =
        -1. *
        (sumOverPhiZPrimePrime.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosThetaSquared *
             sumOverSinTheta -
         sumOverPhiZPrime.at(aHalfbarrel) * sumOverZPrimePrimeSquared.at(aHalfbarrel) * sumOverCosThetaSquared *
             sumOverSinTheta -
         sumOverPhiZPrimePrimeCosTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosTheta *
             sumOverSinTheta +
         sumOverPhiZPrimeCosTheta.at(aHalfbarrel) * sumOverZPrimePrimeSquared.at(aHalfbarrel) * sumOverCosTheta *
             sumOverSinTheta -
         sumOverPhiZPrimePrime.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosTheta *
             sumOverCosThetaSinTheta +
         sumOverPhiZPrime.at(aHalfbarrel) * sumOverZPrimePrimeSquared.at(aHalfbarrel) * sumOverCosTheta *
             sumOverCosThetaSinTheta +
         sumOverPhiZPrimePrimeCosTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * numberOfBeams *
             sumOverCosThetaSinTheta -
         sumOverPhiZPrimeCosTheta.at(aHalfbarrel) * sumOverZPrimePrimeSquared.at(aHalfbarrel) * numberOfBeams *
             sumOverCosThetaSinTheta -
         sumOverPhiZPrimePrimeSinTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * numberOfBeams *
             sumOverCosThetaSquared +
         sumOverPhiZPrimeSinTheta.at(aHalfbarrel) * sumOverZPrimePrimeSquared.at(aHalfbarrel) * numberOfBeams *
             sumOverCosThetaSquared +
         sumOverPhiZPrimePrimeSinTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosTheta *
             sumOverCosTheta -
         sumOverPhiZPrimeSinTheta.at(aHalfbarrel) * sumOverZPrimePrimeSquared.at(aHalfbarrel) * sumOverCosTheta *
             sumOverCosTheta) /
        alignmentDenominator.at(aHalfbarrel) * beamRadii.at(aHalfbarrel);

    // y deviations of the lower z endfaces (in mm)
    theResult.GetParameter(aHalfbarrel, 0, 2).first =
        (sumOverPhiZPrime.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosTheta *
             sumOverSinThetaSquared -
         sumOverPhiZPrimePrime.at(aHalfbarrel) * sumOverZPrimeSquared.at(aHalfbarrel) * sumOverCosTheta *
             sumOverSinThetaSquared -
         sumOverPhiZPrimeCosTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * numberOfBeams *
             sumOverSinThetaSquared +
         sumOverPhiZPrimePrimeCosTheta.at(aHalfbarrel) * sumOverZPrimeSquared.at(aHalfbarrel) * numberOfBeams *
             sumOverSinThetaSquared +
         sumOverPhiZPrimeCosTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverSinTheta *
             sumOverSinTheta -
         sumOverPhiZPrimePrimeCosTheta.at(aHalfbarrel) * sumOverZPrimeSquared.at(aHalfbarrel) * sumOverSinTheta *
             sumOverSinTheta -
         sumOverPhiZPrime.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosThetaSinTheta *
             sumOverSinTheta +
         sumOverPhiZPrimePrime.at(aHalfbarrel) * sumOverZPrimeSquared.at(aHalfbarrel) * sumOverCosThetaSinTheta *
             sumOverSinTheta -
         sumOverPhiZPrimeSinTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosTheta *
             sumOverSinTheta +
         sumOverPhiZPrimePrimeSinTheta.at(aHalfbarrel) * sumOverZPrimeSquared.at(aHalfbarrel) * sumOverCosTheta *
             sumOverSinTheta +
         sumOverPhiZPrimeSinTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * numberOfBeams *
             sumOverCosThetaSinTheta -
         sumOverPhiZPrimePrimeSinTheta.at(aHalfbarrel) * sumOverZPrimeSquared.at(aHalfbarrel) * numberOfBeams *
             sumOverCosThetaSinTheta) /
        alignmentDenominator.at(aHalfbarrel) * beamRadii.at(aHalfbarrel);

    // y deviations of the upper z endfaces (in mm)
    theResult.GetParameter(aHalfbarrel, 1, 2).first =
        (sumOverPhiZPrimePrime.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosTheta *
             sumOverSinThetaSquared -
         sumOverPhiZPrime.at(aHalfbarrel) * sumOverZPrimePrimeSquared.at(aHalfbarrel) * sumOverCosTheta *
             sumOverSinThetaSquared -
         sumOverPhiZPrimePrimeCosTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * numberOfBeams *
             sumOverSinThetaSquared +
         sumOverPhiZPrimeCosTheta.at(aHalfbarrel) * sumOverZPrimePrimeSquared.at(aHalfbarrel) * numberOfBeams *
             sumOverSinThetaSquared +
         sumOverPhiZPrimePrimeCosTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverSinTheta *
             sumOverSinTheta -
         sumOverPhiZPrimeCosTheta.at(aHalfbarrel) * sumOverZPrimePrimeSquared.at(aHalfbarrel) * sumOverSinTheta *
             sumOverSinTheta -
         sumOverPhiZPrimePrime.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosThetaSinTheta *
             sumOverSinTheta +
         sumOverPhiZPrime.at(aHalfbarrel) * sumOverZPrimePrimeSquared.at(aHalfbarrel) * sumOverCosThetaSinTheta *
             sumOverSinTheta -
         sumOverPhiZPrimePrimeSinTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * sumOverCosTheta *
             sumOverSinTheta +
         sumOverPhiZPrimeSinTheta.at(aHalfbarrel) * sumOverZPrimePrimeSquared.at(aHalfbarrel) * sumOverCosTheta *
             sumOverSinTheta +
         sumOverPhiZPrimePrimeSinTheta.at(aHalfbarrel) * sumOverZPrimeZPrimePrime.at(aHalfbarrel) * numberOfBeams *
             sumOverCosThetaSinTheta -
         sumOverPhiZPrimeSinTheta.at(aHalfbarrel) * sumOverZPrimePrimeSquared.at(aHalfbarrel) * numberOfBeams *
             sumOverCosThetaSinTheta) /
        alignmentDenominator.at(aHalfbarrel) * beamRadii.at(aHalfbarrel);
  }

  // another loop is needed here to calculate some terms for the beam parameters
  double vsumA = 0., vsumB = 0., vsumC = 0., vsumD = 0., vsumE = 0., vsumF = 0.;
  for (unsigned int aHalfbarrel = 0; aHalfbarrel < 6; ++aHalfbarrel) {
    vsumA += theResult.GetParameter(aHalfbarrel, 1, 2).first * termA.at(aHalfbarrel) +
             theResult.GetParameter(aHalfbarrel, 0, 2).first * termB.at(aHalfbarrel);
    vsumB += theResult.GetParameter(aHalfbarrel, 1, 1).first * termA.at(aHalfbarrel) +
             theResult.GetParameter(aHalfbarrel, 0, 1).first * termB.at(aHalfbarrel);
    vsumC += beamRadii.at(aHalfbarrel) * (theResult.GetParameter(aHalfbarrel, 1, 0).first * termA.at(aHalfbarrel) +
                                          theResult.GetParameter(aHalfbarrel, 0, 0).first * termB.at(aHalfbarrel));
    vsumD += theResult.GetParameter(aHalfbarrel, 1, 2).first * termC.at(aHalfbarrel) +
             theResult.GetParameter(aHalfbarrel, 0, 2).first * termD.at(aHalfbarrel);
    vsumE += theResult.GetParameter(aHalfbarrel, 1, 1).first * termC.at(aHalfbarrel) +
             theResult.GetParameter(aHalfbarrel, 0, 1).first * termD.at(aHalfbarrel);
    vsumF += beamRadii.at(aHalfbarrel) * (theResult.GetParameter(aHalfbarrel, 1, 0).first * termC.at(aHalfbarrel) +
                                          theResult.GetParameter(aHalfbarrel, 0, 0).first * termD.at(aHalfbarrel));
  }

  // calculate the beam parameters
  for (unsigned int beam = 0; beam < 8; ++beam) {
    // parameter A, defined at lower z
    theResult.GetBeamParameter(beam, 0).first =
        (cos(beamPhiPositions.at(beam)) * vsumA - sin(beamPhiPositions.at(beam)) * vsumB - vsumC +
         sumOverPhiZTPrime.at(beam) * sumOverZTPrimeZTPrimePrime -
         sumOverPhiZTPrimePrime.at(beam) * sumOverZTPrimeSquared) /
        beamDenominator;

    ///////////////////////////////////////////////////////////////////////////////////////////////////
    std::cout << "BBBBBBBB: " << cos(beamPhiPositions.at(beam)) * vsumA << "  "
              << -1. * sin(beamPhiPositions.at(beam)) * vsumB << "  " << -1. * vsumC << "  "
              << sumOverPhiZTPrime.at(beam) * sumOverZTPrimeZTPrimePrime -
                     sumOverPhiZTPrimePrime.at(beam) * sumOverZTPrimeSquared
              << "  " << beamDenominator << std::endl;
    ///////////////////////////////////////////////////////////////////////////////////////////////////

    // parameter B, defined at upper z
    theResult.GetBeamParameter(beam, 1).first =
        (cos(beamPhiPositions.at(beam)) * vsumD - sin(beamPhiPositions.at(beam)) * vsumE - vsumF +
         sumOverPhiZTPrimePrime.at(beam) * sumOverZTPrimeZTPrimePrime -
         sumOverPhiZTPrime.at(beam) * sumOverZTPrimePrimeSquared) /
        beamDenominator;
  }

  return theResult;
}

///
/// get global phi correction from alignment parameters
/// for an alignment tube module in TIB/TOB
///
double LASAlignmentTubeAlgorithm::GetTIBTOBAlignmentParameterCorrection(
    int det,
    int beam,
    int pos,
    LASGlobalData<LASCoordinateSet>& nominalCoordinates,
    LASBarrelAlignmentParameterSet& alignmentParameters) {
  // INITIALIZATION;
  // ALL THIS IS DUPLICATED FOR THE MOMENT, SHOULD FINALLY BE CALCULATED ONLY ONCE
  // AND HARD CODED NUMBERS SHOULD CENTRALLY BE IMPORTED FROM src/LASConstants.h

  // the z positions of the halfbarrel_end_faces / outer_TEC_disks (in mm);
  // parameters are: det, side(0=+/1=-), z(0=lowerZ/1=higherZ). TECs have no side (use side = 0)
  std::vector<std::vector<std::vector<double> > > endFaceZPositions(
      4, std::vector<std::vector<double> >(2, std::vector<double>(2, 0.)));
  endFaceZPositions.at(0).at(0).at(0) = 1322.5;   // TEC+, *, disk1 ///
  endFaceZPositions.at(0).at(0).at(1) = 2667.5;   // TEC+, *, disk9 /// SIDE INFORMATION
  endFaceZPositions.at(1).at(0).at(0) = -2667.5;  // TEC-, *, disk9 /// MEANINGLESS FOR TEC -> USE .at(0)!
  endFaceZPositions.at(1).at(0).at(1) = -1322.5;  // TEC-, *, disk1 ///
  endFaceZPositions.at(2).at(1).at(0) = -700.;    // TIB,  -, outer
  endFaceZPositions.at(2).at(1).at(1) = -300.;    // TIB,  -, inner
  endFaceZPositions.at(2).at(0).at(0) = 300.;     // TIB,  +, inner
  endFaceZPositions.at(2).at(0).at(1) = 700.;     // TIB,  +, outer
  endFaceZPositions.at(3).at(1).at(0) = -1090.;   // TOB,  -, outer
  endFaceZPositions.at(3).at(1).at(1) = -300.;    // TOB,  -, inner
  endFaceZPositions.at(3).at(0).at(0) = 300.;     // TOB,  +, inner
  endFaceZPositions.at(3).at(0).at(1) = 1090.;    // TOB,  +, outer

  // the z positions of the virtual planes at which the beam parameters are measured
  std::vector<double> disk9EndFaceZPositions(2, 0.);
  disk9EndFaceZPositions.at(0) = -2667.5;  // TEC- disk9
  disk9EndFaceZPositions.at(1) = 2667.5;   // TEC+ disk9

  // define the side: 0 for TIB+/TOB+ and 1 for TIB-/TOB-
  const int theSide = pos < 3 ? 0 : 1;

  // define the halfbarrel number from det/side
  const int halfbarrel = det == 2 ? det + theSide : det + 1 + theSide;  // TIB:TOB

  // this is the path the beam has to travel radially after being reflected
  // by the AT mirrors (TIB:50mm, TOB:36mm) -> used for beam parameters
  const double radialOffset = det == 2 ? 50. : 36.;

  // phi positions of the AT beams in rad
  const double phiPositions[8] = {0.392699, 1.289799, 1.851794, 2.748894, 3.645995, 4.319690, 5.216791, 5.778784};
  std::vector<double> beamPhiPositions(8, 0.);
  for (unsigned int aBeam = 0; aBeam < 8; ++aBeam)
    beamPhiPositions.at(aBeam) = phiPositions[aBeam];

  // the radii of the alignment tube beams for each halfbarrel.
  // the halfbarrels 1-6 are (see TkLasATModel TWiki): TEC+, TEC-, TIB+, TIB-. TOB+, TOB-
  // in TIB/TOB modules these radii differ from the beam radius..
  // ..due to the radial offsets (after the semitransparent mirrors)
  const double radii[6] = {564., 564., 514., 514., 600., 600.};
  std::vector<double> beamRadii(6, 0.);
  for (int aHalfbarrel = 0; aHalfbarrel < 6; ++aHalfbarrel)
    beamRadii.at(aHalfbarrel) = radii[aHalfbarrel];

  // reduced z positions of the beam spots ( z'_{k,j}, z"_{k,j} )
  double detReducedZ[2] = {0., 0.};
  // reduced beam splitter positions ( zt'_{k,j}, zt"_{k,j} )
  double beamReducedZ[2] = {0., 0.};

  // reduced module's z position with respect to the subdetector endfaces (zPrime, zPrimePrime)
  detReducedZ[0] = nominalCoordinates.GetTIBTOBEntry(det, beam, pos).GetZ() -
                   endFaceZPositions.at(det).at(theSide).at(0);  // = zPrime
  detReducedZ[0] /= (endFaceZPositions.at(det).at(theSide).at(1) - endFaceZPositions.at(det).at(theSide).at(0));
  detReducedZ[1] = endFaceZPositions.at(det).at(theSide).at(1) -
                   nominalCoordinates.GetTIBTOBEntry(det, beam, pos).GetZ();  // = zPrimePrime
  detReducedZ[1] /= (endFaceZPositions.at(det).at(theSide).at(1) - endFaceZPositions.at(det).at(theSide).at(0));

  // reduced module's z position with respect to the tec disks +-9 (for the beam parameters)
  beamReducedZ[0] = (nominalCoordinates.GetTIBTOBEntry(det, beam, pos).GetZ() - radialOffset) -
                    disk9EndFaceZPositions.at(0);  // = ZTPrime
  beamReducedZ[0] /= (disk9EndFaceZPositions.at(1) - disk9EndFaceZPositions.at(0));
  beamReducedZ[1] = disk9EndFaceZPositions.at(1) -
                    (nominalCoordinates.GetTIBTOBEntry(det, beam, pos).GetZ() - radialOffset);  // ZTPrimePrime
  beamReducedZ[1] /= (disk9EndFaceZPositions.at(1) - disk9EndFaceZPositions.at(0));

  // the correction to phi from the endcap algorithm;
  // it is defined such that the correction is to be subtracted ///////////////////////////////// ???
  double phiCorrection = 0.;

  // contribution from phi rotation of first end face
  phiCorrection += detReducedZ[1] * alignmentParameters.GetParameter(halfbarrel, 0, 0).first;

  // contribution from phi rotation of second end face
  phiCorrection += detReducedZ[0] * alignmentParameters.GetParameter(halfbarrel, 1, 0).first;

  // contribution from translation along x of first endface
  phiCorrection += detReducedZ[1] * sin(beamPhiPositions.at(beam)) *
                   alignmentParameters.GetParameter(halfbarrel, 0, 1).first / beamRadii.at(halfbarrel);

  // contribution from translation along x of second endface
  phiCorrection += detReducedZ[0] * sin(beamPhiPositions.at(beam)) *
                   alignmentParameters.GetParameter(halfbarrel, 1, 1).first / beamRadii.at(halfbarrel);

  // contribution from translation along y of first endface
  phiCorrection -= detReducedZ[1] * cos(beamPhiPositions.at(beam)) *
                   alignmentParameters.GetParameter(halfbarrel, 0, 2).first / beamRadii.at(halfbarrel);

  // contribution from translation along y of second endface
  phiCorrection -= detReducedZ[0] * cos(beamPhiPositions.at(beam)) *
                   alignmentParameters.GetParameter(halfbarrel, 1, 2).first / beamRadii.at(halfbarrel);

  // contribution from beam parameters;
  // originally, the contribution in meter is proportional to the radius of the beams: beamRadii.at( 0 )
  // the additional factor: beamRadii.at( halfbarrel ) converts from meter to radian on the module
  phiCorrection += beamReducedZ[1] * alignmentParameters.GetBeamParameter(beam, 0).first * beamRadii.at(0) /
                   beamRadii.at(halfbarrel);
  phiCorrection += beamReducedZ[0] * alignmentParameters.GetBeamParameter(beam, 1).first * beamRadii.at(0) /
                   beamRadii.at(halfbarrel);

  return phiCorrection;
}

///
/// get global phi correction from alignment parameters
/// for an alignment tube module in TEC(AT)
///
double LASAlignmentTubeAlgorithm::GetTEC2TECAlignmentParameterCorrection(
    int det,
    int beam,
    int disk,
    LASGlobalData<LASCoordinateSet>& nominalCoordinates,
    LASBarrelAlignmentParameterSet& alignmentParameters) {
  // INITIALIZATION;
  // ALL THIS IS DUPLICATED FOR THE MOMENT, SHOULD FINALLY BE CALCULATED ONLY ONCE
  // AND HARD CODED NUMBERS SHOULD CENTRALLY BE IMPORTED FROM src/LASConstants.h

  // the z positions of the halfbarrel_end_faces / outer_TEC_disks (in mm);
  // parameters are: det, side(0=+/1=-), z(0=lowerZ/1=higherZ). TECs have no side (use side = 0)
  std::vector<std::vector<std::vector<double> > > endFaceZPositions(
      4, std::vector<std::vector<double> >(2, std::vector<double>(2, 0.)));
  endFaceZPositions.at(0).at(0).at(0) = 1322.5;   // TEC+, *, disk1 ///
  endFaceZPositions.at(0).at(0).at(1) = 2667.5;   // TEC+, *, disk9 /// SIDE INFORMATION
  endFaceZPositions.at(1).at(0).at(0) = -2667.5;  // TEC-, *, disk9 /// MEANINGLESS FOR TEC -> USE .at(0)!
  endFaceZPositions.at(1).at(0).at(1) = -1322.5;  // TEC-, *, disk1 ///
  endFaceZPositions.at(2).at(1).at(0) = -700.;    // TIB,  -, outer
  endFaceZPositions.at(2).at(1).at(1) = -300.;    // TIB,  -, inner
  endFaceZPositions.at(2).at(0).at(0) = 300.;     // TIB,  +, inner
  endFaceZPositions.at(2).at(0).at(1) = 700.;     // TIB,  +, outer
  endFaceZPositions.at(3).at(1).at(0) = -1090.;   // TOB,  -, outer
  endFaceZPositions.at(3).at(1).at(1) = -300.;    // TOB,  -, inner
  endFaceZPositions.at(3).at(0).at(0) = 300.;     // TOB,  +, inner
  endFaceZPositions.at(3).at(0).at(1) = 1090.;    // TOB,  +, outer

  // the z positions of the virtual planes at which the beam parameters are measured
  std::vector<double> disk9EndFaceZPositions(2, 0.);
  disk9EndFaceZPositions.at(0) = -2667.5;  // TEC- disk9
  disk9EndFaceZPositions.at(1) = 2667.5;   // TEC+ disk9

  // for the tec, the halfbarrel numbers are equal to the det numbers...
  const int halfbarrel = det;

  // ...so there's no side distinction for the TEC
  const int theSide = 0;

  // also, there's no radial offset for the TEC
  const double radialOffset = 0.;

  // phi positions of the AT beams in rad
  const double phiPositions[8] = {0.392699, 1.289799, 1.851794, 2.748894, 3.645995, 4.319690, 5.216791, 5.778784};
  std::vector<double> beamPhiPositions(8, 0.);
  for (unsigned int aBeam = 0; aBeam < 8; ++aBeam)
    beamPhiPositions.at(aBeam) = phiPositions[aBeam];

  // the radii of the alignment tube beams for each halfbarrel.
  // the halfbarrels 1-6 are (see TkLasATModel TWiki): TEC+, TEC-, TIB+, TIB-. TOB+, TOB-
  // in TIB/TOB modules these radii differ from the beam radius..
  // ..due to the radial offsets (after the semitransparent mirrors)
  const double radii[6] = {564., 564., 514., 514., 600., 600.};
  std::vector<double> beamRadii(6, 0.);
  for (int aHalfbarrel = 0; aHalfbarrel < 6; ++aHalfbarrel)
    beamRadii.at(aHalfbarrel) = radii[aHalfbarrel];

  // reduced z positions of the beam spots ( z'_{k,j}, z"_{k,j} )
  double detReducedZ[2] = {0., 0.};
  // reduced beam splitter positions ( zt'_{k,j}, zt"_{k,j} )
  double beamReducedZ[2] = {0., 0.};

  // reduced module's z position with respect to the subdetector endfaces (zPrime, zPrimePrime)
  detReducedZ[0] = nominalCoordinates.GetTEC2TECEntry(det, beam, disk).GetZ() -
                   endFaceZPositions.at(det).at(theSide).at(0);  // = zPrime
  detReducedZ[0] /= (endFaceZPositions.at(det).at(theSide).at(1) - endFaceZPositions.at(det).at(theSide).at(0));
  detReducedZ[1] = endFaceZPositions.at(det).at(theSide).at(1) -
                   nominalCoordinates.GetTEC2TECEntry(det, beam, disk).GetZ();  // = zPrimePrime
  detReducedZ[1] /= (endFaceZPositions.at(det).at(theSide).at(1) - endFaceZPositions.at(det).at(theSide).at(0));

  // reduced module's z position with respect to the tec disks +-9 (for the beam parameters)
  beamReducedZ[0] = (nominalCoordinates.GetTEC2TECEntry(det, beam, disk).GetZ() - radialOffset) -
                    disk9EndFaceZPositions.at(0);  // = ZTPrime
  beamReducedZ[0] /= (disk9EndFaceZPositions.at(1) - disk9EndFaceZPositions.at(0));
  beamReducedZ[1] = disk9EndFaceZPositions.at(1) -
                    (nominalCoordinates.GetTEC2TECEntry(det, beam, disk).GetZ() - radialOffset);  // ZTPrimePrime
  beamReducedZ[1] /= (disk9EndFaceZPositions.at(1) - disk9EndFaceZPositions.at(0));

  // the correction to phi from the endcap algorithm;
  // it is defined such that the correction is to be subtracted ///////////////////////////////// ???
  double phiCorrection = 0.;

  // contribution from phi rotation of first end face
  phiCorrection += detReducedZ[1] * alignmentParameters.GetParameter(halfbarrel, 0, 0).first;

  // contribution from phi rotation of second end face
  phiCorrection += detReducedZ[0] * alignmentParameters.GetParameter(halfbarrel, 1, 0).first;

  // contribution from translation along x of first endface
  phiCorrection += detReducedZ[1] * sin(beamPhiPositions.at(beam)) *
                   alignmentParameters.GetParameter(halfbarrel, 0, 1).first / beamRadii.at(halfbarrel);

  // contribution from translation along x of second endface
  phiCorrection += detReducedZ[0] * sin(beamPhiPositions.at(beam)) *
                   alignmentParameters.GetParameter(halfbarrel, 1, 1).first / beamRadii.at(halfbarrel);

  // contribution from translation along y of first endface
  phiCorrection -= detReducedZ[1] * cos(beamPhiPositions.at(beam)) *
                   alignmentParameters.GetParameter(halfbarrel, 0, 2).first / beamRadii.at(halfbarrel);

  // contribution from translation along y of second endface
  phiCorrection -= detReducedZ[0] * cos(beamPhiPositions.at(beam)) *
                   alignmentParameters.GetParameter(halfbarrel, 1, 2).first / beamRadii.at(halfbarrel);

  // contribution from beam parameters;
  // originally, the contribution in meter is proportional to the radius of the beams: beamRadii.at( 0 )
  // the additional factor: beamRadii.at( halfbarrel ) converts from meter to radian on the module
  phiCorrection += beamReducedZ[1] * alignmentParameters.GetBeamParameter(beam, 0).first * beamRadii.at(0) /
                   beamRadii.at(halfbarrel);
  phiCorrection += beamReducedZ[0] * alignmentParameters.GetBeamParameter(beam, 1).first * beamRadii.at(0) /
                   beamRadii.at(halfbarrel);

  return phiCorrection;
}

///
/// allows to push in a simple simulated misalignment for quick internal testing purposes;
/// overwrites LASGlobalData<LASCoordinateSet>& measuredCoordinates;
/// call at beginning of LASBarrelAlgorithm::CalculateParameters method
///
/// one line per module,
/// format for TEC:              det ring beam disk phi phiErr
/// format for TEC(at) & TIBTOB: det beam   z  "-1" phi phiErr
///
void LASAlignmentTubeAlgorithm::ReadMisalignmentFromFile(const char* filename,
                                                         LASGlobalData<LASCoordinateSet>& measuredCoordinates,
                                                         LASGlobalData<LASCoordinateSet>& nominalCoordinates) {
  std::ifstream file(filename);
  if (file.bad()) {
    std::cerr << " [LASAlignmentTubeAlgorithm::ReadMisalignmentFromFile] ** ERROR: cannot open file \"" << filename
              << "\"." << std::endl;
    return;
  }

  std::cerr << " @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
               "@@@@@@@@@@@"
            << std::endl;
  std::cerr << " @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
               "@@@@@@@@@@@"
            << std::endl;
  std::cerr << " [LASAlignmentTubeAlgorithm::ReadMisalignmentFromFile] ** WARNING: you are reading a fake measurement "
               "from a file!"
            << std::endl;
  std::cerr << " @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
               "@@@@@@@@@@@"
            << std::endl;
  std::cerr << " @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@"
               "@@@@@@@@@@@"
            << std::endl;

  // the measured coordinates will finally be overwritten;
  // first, set them to the nominal values
  measuredCoordinates = nominalCoordinates;

  // and put large errors on all values;
  {
    LASGlobalLoop moduleLoop;
    int det, ring, beam, disk, pos;

    det = 0;
    ring = 0;
    beam = 0;
    disk = 0;
    do {
      measuredCoordinates.GetTECEntry(det, ring, beam, disk).SetPhiError(1000.);
    } while (moduleLoop.TECLoop(det, ring, beam, disk));

    det = 2;
    beam = 0;
    pos = 0;
    do {
      measuredCoordinates.GetTIBTOBEntry(det, beam, pos).SetPhiError(1000.);
    } while (moduleLoop.TIBTOBLoop(det, beam, pos));

    det = 0;
    beam = 0;
    disk = 0;
    do {
      measuredCoordinates.GetTEC2TECEntry(det, beam, disk).SetPhiError(1000.);
    } while (moduleLoop.TEC2TECLoop(det, beam, disk));
  }

  // buffers for read-in
  int det, beam, z, ring;
  double phi, phiError;

  while (!file.eof()) {
    file >> det;
    if (file.eof())
      break;  // do not read the last line twice, do not fill trash if file empty

    file >> beam;
    file >> z;
    file >> ring;
    file >> phi;
    file >> phiError;

    if (det > 1) {  // TIB/TOB
      measuredCoordinates.GetTIBTOBEntry(det, beam, z).SetPhi(phi);
      measuredCoordinates.GetTIBTOBEntry(det, beam, z).SetPhiError(phiError);
    } else {            // TEC or TEC(at)
      if (ring > -1) {  // TEC
        measuredCoordinates.GetTECEntry(det, ring, beam, z).SetPhi(phi);
        measuredCoordinates.GetTECEntry(det, ring, beam, z).SetPhiError(phiError);
      } else {  // TEC(at)
        measuredCoordinates.GetTEC2TECEntry(det, beam, z).SetPhi(phi);
        measuredCoordinates.GetTEC2TECEntry(det, beam, z).SetPhiError(phiError);
      }
    }
  }

  file.close();
}
