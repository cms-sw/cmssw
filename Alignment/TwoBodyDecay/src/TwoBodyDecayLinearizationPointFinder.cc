
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayLinearizationPointFinder.h"
#include "Alignment/TwoBodyDecay/interface/TwoBodyDecayModel.h"
#include "DataFormats/CLHEP/interface/AlgebraicObjects.h"

const TwoBodyDecayParameters TwoBodyDecayLinearizationPointFinder::getLinearizationPoint(
    const std::vector<RefCountedLinearizedTrackState> &tracks,
    const double primaryMass,
    const double secondaryMass) const {
  GlobalPoint linPoint = tracks[0]->linearizationPoint();
  PerigeeLinearizedTrackState *linTrack1 = dynamic_cast<PerigeeLinearizedTrackState *>(tracks[0].get());
  PerigeeLinearizedTrackState *linTrack2 = dynamic_cast<PerigeeLinearizedTrackState *>(tracks[1].get());
  if (!linTrack1 || !linTrack2)
    return TwoBodyDecayParameters();

  GlobalVector firstMomentum = linTrack1->predictedState().momentum();
  GlobalVector secondMomentum = linTrack2->predictedState().momentum();

  AlgebraicVector secondaryMomentum1(3);
  secondaryMomentum1[0] = firstMomentum.x();
  secondaryMomentum1[1] = firstMomentum.y();
  secondaryMomentum1[2] = firstMomentum.z();

  AlgebraicVector secondaryMomentum2(3);
  secondaryMomentum2[0] = secondMomentum.x();
  secondaryMomentum2[1] = secondMomentum.y();
  secondaryMomentum2[2] = secondMomentum.z();

  AlgebraicVector primaryMomentum = secondaryMomentum1 + secondaryMomentum2;

  TwoBodyDecayModel decayModel(primaryMass, secondaryMass);
  AlgebraicMatrix rotMat = decayModel.rotationMatrix(primaryMomentum[0], primaryMomentum[1], primaryMomentum[2]);
  AlgebraicMatrix invRotMat = rotMat.T();

  double p = primaryMomentum.norm();
  double pSquared = p * p;
  double gamma = sqrt(pSquared + primaryMass * primaryMass) / primaryMass;
  double betaGamma = p / primaryMass;
  AlgebraicSymMatrix lorentzTransformation(4, 1);
  lorentzTransformation[0][0] = gamma;
  lorentzTransformation[3][3] = gamma;
  lorentzTransformation[0][3] = -betaGamma;

  double p1 = secondaryMomentum1.norm();
  AlgebraicVector boostedLorentzMomentum1(4);
  boostedLorentzMomentum1[0] = sqrt(p1 * p1 + secondaryMass * secondaryMass);
  boostedLorentzMomentum1.sub(2, invRotMat * secondaryMomentum1);

  AlgebraicVector restFrameLorentzMomentum1 = lorentzTransformation * boostedLorentzMomentum1;
  AlgebraicVector restFrameMomentum1 = restFrameLorentzMomentum1.sub(2, 4);
  double perp1 = sqrt(restFrameMomentum1[0] * restFrameMomentum1[0] + restFrameMomentum1[1] * restFrameMomentum1[1]);
  double theta1 = atan2(perp1, restFrameMomentum1[2]);
  double phi1 = atan2(restFrameMomentum1[1], restFrameMomentum1[0]);

  double p2 = secondaryMomentum2.norm();
  AlgebraicVector boostedLorentzMomentum2(4);
  boostedLorentzMomentum2[0] = sqrt(p2 * p2 + secondaryMass * secondaryMass);
  boostedLorentzMomentum2.sub(2, invRotMat * secondaryMomentum2);

  AlgebraicVector restFrameLorentzMomentum2 = lorentzTransformation * boostedLorentzMomentum2;
  AlgebraicVector restFrameMomentum2 = restFrameLorentzMomentum2.sub(2, 4);
  double perp2 = sqrt(restFrameMomentum2[0] * restFrameMomentum2[0] + restFrameMomentum2[1] * restFrameMomentum2[1]);
  double theta2 = atan2(perp2, restFrameMomentum2[2]);
  double phi2 = atan2(restFrameMomentum2[1], restFrameMomentum2[0]);

  double pi = 3.141592654;
  double relSign = -1.;

  if (phi1 < 0)
    phi1 += 2 * pi;
  if (phi2 < 0)
    phi2 += 2 * pi;
  if (phi1 > phi2)
    relSign = 1.;

  double momentumSquared1 = secondaryMomentum1.normsq();
  double energy1 = sqrt(secondaryMass * secondaryMass + momentumSquared1);
  double momentumSquared2 = secondaryMomentum2.normsq();
  double energy2 = sqrt(secondaryMass * secondaryMass + momentumSquared2);
  double sumMomentaSquared = (secondaryMomentum1 + secondaryMomentum2).normsq();
  double sumEnergiesSquared = (energy1 + energy2) * (energy1 + energy2);
  double estimatedPrimaryMass = sqrt(sumEnergiesSquared - sumMomentaSquared);

  AlgebraicVector linParam(TwoBodyDecayParameters::dimension, 0);
  linParam[TwoBodyDecayParameters::x] = linPoint.x();
  linParam[TwoBodyDecayParameters::y] = linPoint.y();
  linParam[TwoBodyDecayParameters::z] = linPoint.z();
  linParam[TwoBodyDecayParameters::px] = primaryMomentum[0];
  linParam[TwoBodyDecayParameters::py] = primaryMomentum[1];
  linParam[TwoBodyDecayParameters::pz] = primaryMomentum[2];
  linParam[TwoBodyDecayParameters::theta] = 0.5 * (theta1 - theta2 + pi);
  linParam[TwoBodyDecayParameters::phi] = 0.5 * (phi1 + phi2 + relSign * pi);
  linParam[TwoBodyDecayParameters::mass] = estimatedPrimaryMass;

  return TwoBodyDecayParameters(linParam);
}
