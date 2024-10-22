/*
 *  See header file for a description of this class.
 *
 *
 *  \author Haiyun.Teng - Peking University
 *
 */

#include "RecoMuon/MuonSeedGenerator/src/RPCSeedPattern.h"
#include <RecoMuon/TrackingTools/interface/MuonPatternRecoDumper.h>
#include <MagneticField/Engine/interface/MagneticField.h>
#include <TrackingTools/TrajectoryState/interface/TrajectoryStateTransform.h>
#include <TrackingTools/DetLayers/interface/DetLayer.h>
#include <DataFormats/TrajectoryState/interface/PTrajectoryStateOnDet.h>
#include <DataFormats/Common/interface/OwnVector.h>
#include <DataFormats/MuonDetId/interface/RPCDetId.h>
#include <FWCore/Framework/interface/ESHandle.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <Geometry/CommonDetUnit/interface/GeomDet.h>
#include <Geometry/RPCGeometry/interface/RPCChamber.h>
#include <Geometry/RPCGeometry/interface/RPCGeometry.h>

#include "gsl/gsl_statistics.h"
#include "TH1F.h"
#include <cmath>

using namespace std;
using namespace edm;

RPCSeedPattern::RPCSeedPattern() {
  isPatternChecked = false;
  isConfigured = false;
  MagnecticFieldThreshold = 0.5;
}

RPCSeedPattern::~RPCSeedPattern() {}

void RPCSeedPattern::configure(const edm::ParameterSet& iConfig) {
  MaxRSD = iConfig.getParameter<double>("MaxRSD");
  deltaRThreshold = iConfig.getParameter<double>("deltaRThreshold");
  AlgorithmType = iConfig.getParameter<unsigned int>("AlgorithmType");
  autoAlgorithmChoose = iConfig.getParameter<bool>("autoAlgorithmChoose");
  ZError = iConfig.getParameter<double>("ZError");
  MinDeltaPhi = iConfig.getParameter<double>("MinDeltaPhi");
  MagnecticFieldThreshold = iConfig.getParameter<double>("MagnecticFieldThreshold");
  stepLength = iConfig.getParameter<double>("stepLength");
  sampleCount = iConfig.getParameter<unsigned int>("sampleCount");
  isConfigured = true;
}

RPCSeedPattern::weightedTrajectorySeed RPCSeedPattern::seed(const MagneticField& Field,
                                                            const RPCGeometry& rpcGeom,
                                                            int& isGoodSeed) {
  if (isConfigured == false) {
    cout << "Configuration not set yet" << endl;
    return createFakeSeed(isGoodSeed, Field);
  }

  // Check recHit number, if fail we return a fake seed and set pattern to "wrong"
  unsigned int NumberofHitsinSeed = nrhit();
  if (NumberofHitsinSeed < 3) {
    return createFakeSeed(isGoodSeed, Field);
  }
  // If only three recHits, we don't have other choice
  if (NumberofHitsinSeed == 3)
    ThreePointsAlgorithm();

  if (NumberofHitsinSeed > 3) {
    if (autoAlgorithmChoose == false) {
      cout << "computePtWithmorerecHits" << endl;
      if (AlgorithmType == 0)
        ThreePointsAlgorithm();
      if (AlgorithmType == 1)
        MiddlePointsAlgorithm();
      if (AlgorithmType == 2)
        SegmentAlgorithm();
      if (AlgorithmType == 3) {
        if (checkSegment()) {
          SegmentAlgorithmSpecial(Field);
        } else {
          cout << "Not enough recHits for Special Segment Algorithm" << endl;
          return createFakeSeed(isGoodSeed, Field);
        }
      }
    } else {
      if (checkSegment()) {
        AlgorithmType = 3;
        SegmentAlgorithmSpecial(Field);
      } else {
        AlgorithmType = 2;
        SegmentAlgorithm();
      }
    }
  }

  // Check the pattern
  if (isPatternChecked == false) {
    if (AlgorithmType != 3) {
      checkSimplePattern(Field);
    } else {
      checkSegmentAlgorithmSpecial(Field, rpcGeom);
    }
  }

  return createSeed(isGoodSeed, Field);
}

void RPCSeedPattern::ThreePointsAlgorithm() {
  cout << "computePtWith3recHits" << endl;
  unsigned int NumberofHitsinSeed = nrhit();
  // Check recHit number, if fail we set the pattern to "wrong"
  if (NumberofHitsinSeed < 3) {
    isPatternChecked = true;
    isGoodPattern = -1;
    return;
  }
  // Choose every 3 recHits to form a part
  unsigned int NumberofPart = NumberofHitsinSeed * (NumberofHitsinSeed - 1) * (NumberofHitsinSeed - 2) / (3 * 2);
  ;
  double* pt = new double[NumberofPart];
  double* pt_err = new double[NumberofPart];
  // Loop for each three-recHits part
  ConstMuonRecHitPointer precHit[3];
  unsigned int n = 0;
  unsigned int NumberofStraight = 0;
  for (unsigned int i = 0; i < (NumberofHitsinSeed - 2); i++)
    for (unsigned int j = (i + 1); j < (NumberofHitsinSeed - 1); j++)
      for (unsigned int k = (j + 1); k < NumberofHitsinSeed; k++) {
        precHit[0] = theRecHits[i];
        precHit[1] = theRecHits[j];
        precHit[2] = theRecHits[k];
        bool checkStraight = checkStraightwithThreerecHits(precHit, MinDeltaPhi);
        if (!checkStraight) {
          GlobalVector Center_temp = computePtwithThreerecHits(pt[n], pt_err[n], precHit);
          // For simple pattern
          Center += Center_temp;
        } else {
          // For simple pattern
          NumberofStraight++;
          pt[n] = upper_limit_pt;
          pt_err[n] = 0;
        }
        n++;
      }
  // For simple pattern, only one general parameter for pattern
  if (NumberofStraight == NumberofPart) {
    isStraight = true;
    meanRadius = -1;
  } else {
    isStraight = false;
    Center /= (NumberofPart - NumberofStraight);
    double meanR = 0.;
    for (ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin(); iter != theRecHits.end(); iter++)
      meanR += getDistance(*iter, Center);
    meanR /= NumberofHitsinSeed;
    meanRadius = meanR;
  }

  // Unset the pattern estimation signa
  isPatternChecked = false;

  //double ptmean0 = 0;
  //double sptmean0 = 0;
  //computeBestPt(pt, pt_err, ptmean0, sptmean0, (NumberofPart - NumberofStraight));

  delete[] pt;
  delete[] pt_err;
}

void RPCSeedPattern::MiddlePointsAlgorithm() {
  cout << "Using middle points algorithm" << endl;
  unsigned int NumberofHitsinSeed = nrhit();
  // Check recHit number, if fail we set the pattern to "wrong"
  if (NumberofHitsinSeed < 4) {
    isPatternChecked = true;
    isGoodPattern = -1;
    return;
  }
  double* X = new double[NumberofHitsinSeed];
  double* Y = new double[NumberofHitsinSeed];
  unsigned int n = 0;
  for (ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin(); iter != theRecHits.end(); iter++) {
    X[n] = (*iter)->globalPosition().x();
    Y[n] = (*iter)->globalPosition().y();
    cout << "X[" << n << "] = " << X[n] << ", Y[" << n << "]= " << Y[n] << endl;
    n++;
  }
  unsigned int NumberofPoints = NumberofHitsinSeed;
  while (NumberofPoints > 3) {
    for (unsigned int i = 0; i <= (NumberofPoints - 2); i++) {
      X[i] = (X[i] + X[i + 1]) / 2;
      Y[i] = (Y[i] + Y[i + 1]) / 2;
    }
    NumberofPoints--;
  }
  double x[3], y[3];
  for (unsigned int i = 0; i < 3; i++) {
    x[i] = X[i];
    y[i] = Y[i];
  }
  double pt = 0;
  double pt_err = 0;
  bool checkStraight = checkStraightwithThreerecHits(x, y, MinDeltaPhi);
  if (!checkStraight) {
    GlobalVector Center_temp = computePtWithThreerecHits(pt, pt_err, x, y);
    double meanR = 0.;
    for (ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin(); iter != theRecHits.end(); iter++)
      meanR += getDistance(*iter, Center_temp);
    meanR /= NumberofHitsinSeed;
    // For simple pattern
    isStraight = false;
    Center = Center_temp;
    meanRadius = meanR;
  } else {
    // For simple pattern
    isStraight = true;
    meanRadius = -1;
  }

  // Unset the pattern estimation signa
  isPatternChecked = false;

  delete[] X;
  delete[] Y;
}

void RPCSeedPattern::SegmentAlgorithm() {
  cout << "Using segments algorithm" << endl;
  unsigned int NumberofHitsinSeed = nrhit();
  // Check recHit number, if fail we set the pattern to "wrong"
  if (NumberofHitsinSeed < 4) {
    isPatternChecked = true;
    isGoodPattern = -1;
    return;
  }

  RPCSegment* Segment;
  unsigned int NumberofSegment = NumberofHitsinSeed - 2;
  Segment = new RPCSegment[NumberofSegment];
  unsigned int n = 0;
  for (ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin(); iter != (theRecHits.end() - 2); iter++) {
    Segment[n].first = (*iter);
    Segment[n].second = (*(iter + 2));
    n++;
  }
  unsigned int NumberofStraight = 0;
  for (unsigned int i = 0; i < NumberofSegment - 1; i++) {
    bool checkStraight = checkStraightwithSegment(Segment[i], Segment[i + 1], MinDeltaPhi);
    if (checkStraight == true) {
      // For simple patterm
      NumberofStraight++;
    } else {
      GlobalVector Center_temp = computePtwithSegment(Segment[i], Segment[i + 1]);
      // For simple patterm
      Center += Center_temp;
    }
  }
  // For simple pattern, only one general parameter for pattern
  if ((NumberofSegment - 1 - NumberofStraight) > 0) {
    isStraight = false;
    Center /= (NumberofSegment - 1 - NumberofStraight);
    double meanR = 0.;
    for (ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin(); iter != theRecHits.end(); iter++)
      meanR += getDistance(*iter, Center);
    meanR /= NumberofHitsinSeed;
    meanRadius = meanR;
  } else {
    isStraight = true;
    meanRadius = -1;
  }

  // Unset the pattern estimation signal
  isPatternChecked = false;

  delete[] Segment;
}

void RPCSeedPattern::SegmentAlgorithmSpecial(const MagneticField& Field) {
  //unsigned int NumberofHitsinSeed = nrhit();
  if (!checkSegment()) {
    isPatternChecked = true;
    isGoodPattern = -1;
    return;
  }

  // Get magnetice field sampling information, recHit's position is not the border of Chamber and Iron
  for (ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin(); iter != (theRecHits.end() - 1); iter++) {
    GlobalPoint gpFirst = (*iter)->globalPosition();
    GlobalPoint gpLast = (*(iter + 1))->globalPosition();
    GlobalPoint* gp = new GlobalPoint[sampleCount];
    double dx = (gpLast.x() - gpFirst.x()) / (sampleCount + 1);
    double dy = (gpLast.y() - gpFirst.y()) / (sampleCount + 1);
    double dz = (gpLast.z() - gpFirst.z()) / (sampleCount + 1);
    for (unsigned int index = 0; index < sampleCount; index++) {
      gp[index] = GlobalPoint(
          (gpFirst.x() + dx * (index + 1)), (gpFirst.y() + dy * (index + 1)), (gpFirst.z() + dz * (index + 1)));
      GlobalVector MagneticVec_temp = Field.inTesla(gp[index]);
      cout << "Sampling magnetic field : " << MagneticVec_temp << endl;
      //BValue.push_back(MagneticVec_temp);
    }
    delete[] gp;
  }

  // form two segments
  ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin();
  for (unsigned int n = 0; n <= 1; n++) {
    SegmentRB[n].first = (*iter);
    cout << "SegmentRB " << n << " recHit: " << (*iter)->globalPosition() << endl;
    iter++;
    SegmentRB[n].second = (*iter);
    cout << "SegmentRB " << n << " recHit: " << (*iter)->globalPosition() << endl;
    iter++;
  }
  GlobalVector segvec1 = (SegmentRB[0].second)->globalPosition() - (SegmentRB[0].first)->globalPosition();
  GlobalVector segvec2 = (SegmentRB[1].second)->globalPosition() - (SegmentRB[1].first)->globalPosition();

  // extrapolate the segment to find the Iron border which magnetic field is at large value
  entryPosition = (SegmentRB[0].second)->globalPosition();
  leavePosition = (SegmentRB[1].first)->globalPosition();
  while (fabs(Field.inTesla(entryPosition).z()) < MagnecticFieldThreshold) {
    cout << "Entry position is : " << entryPosition << ", and stepping into next point" << endl;
    entryPosition += segvec1.unit() * stepLength;
  }
  // Loop back for more accurate by stepLength/10
  while (fabs(Field.inTesla(entryPosition).z()) >= MagnecticFieldThreshold) {
    cout << "Entry position is : " << entryPosition << ", and stepping back into next point" << endl;
    entryPosition -= segvec1.unit() * stepLength / 10;
  }
  entryPosition += 0.5 * segvec1.unit() * stepLength / 10;
  cout << "Final entry position is : " << entryPosition << endl;

  while (fabs(Field.inTesla(leavePosition).z()) < MagnecticFieldThreshold) {
    cout << "Leave position is : " << leavePosition << ", and stepping into next point" << endl;
    leavePosition -= segvec2.unit() * stepLength;
  }
  // Loop back for more accurate by stepLength/10
  while (fabs(Field.inTesla(leavePosition).z()) >= MagnecticFieldThreshold) {
    cout << "Leave position is : " << leavePosition << ", and stepping back into next point" << endl;
    leavePosition += segvec2.unit() * stepLength / 10;
  }
  leavePosition -= 0.5 * segvec2.unit() * stepLength / 10;
  cout << "Final leave position is : " << leavePosition << endl;

  // Sampling magnetic field in Iron region
  GlobalPoint* gp = new GlobalPoint[sampleCount];
  double dx = (leavePosition.x() - entryPosition.x()) / (sampleCount + 1);
  double dy = (leavePosition.y() - entryPosition.y()) / (sampleCount + 1);
  double dz = (leavePosition.z() - entryPosition.z()) / (sampleCount + 1);
  std::vector<GlobalVector> BValue;
  BValue.clear();
  for (unsigned int index = 0; index < sampleCount; index++) {
    gp[index] = GlobalPoint((entryPosition.x() + dx * (index + 1)),
                            (entryPosition.y() + dy * (index + 1)),
                            (entryPosition.z() + dz * (index + 1)));
    GlobalVector MagneticVec_temp = Field.inTesla(gp[index]);
    cout << "Sampling magnetic field : " << MagneticVec_temp << endl;
    BValue.push_back(MagneticVec_temp);
  }
  delete[] gp;
  GlobalVector meanB2(0, 0, 0);
  for (std::vector<GlobalVector>::const_iterator BIter = BValue.begin(); BIter != BValue.end(); BIter++)
    meanB2 += (*BIter);
  meanB2 /= BValue.size();
  cout << "Mean B field is " << meanB2 << endl;
  meanMagneticField2 = meanB2;

  double meanBz2 = meanB2.z();
  double deltaBz2 = 0.;
  for (std::vector<GlobalVector>::const_iterator BIter = BValue.begin(); BIter != BValue.end(); BIter++)
    deltaBz2 += (BIter->z() - meanBz2) * (BIter->z() - meanBz2);
  deltaBz2 /= BValue.size();
  deltaBz2 = sqrt(deltaBz2);
  cout << "delta Bz is " << deltaBz2 << endl;

  // Distance of the initial 3 segment
  S = 0;
  bool checkStraight = checkStraightwithSegment(SegmentRB[0], SegmentRB[1], MinDeltaPhi);
  if (checkStraight == true) {
    // Just for complex pattern
    isStraight2 = checkStraight;
    Center2 = GlobalVector(0, 0, 0);
    meanRadius2 = -1;
    GlobalVector MomentumVec = (SegmentRB[1].second)->globalPosition() - (SegmentRB[0].first)->globalPosition();
    S += MomentumVec.perp();
    lastPhi = MomentumVec.phi().value();
  } else {
    GlobalVector seg1 = entryPosition - (SegmentRB[0].first)->globalPosition();
    S += seg1.perp();
    GlobalVector seg2 = (SegmentRB[1].second)->globalPosition() - leavePosition;
    S += seg2.perp();
    GlobalVector vecZ(0, 0, 1);
    GlobalVector gvec1 = seg1.cross(vecZ);
    GlobalVector gvec2 = seg2.cross(vecZ);
    double A1 = gvec1.x();
    double B1 = gvec1.y();
    double A2 = gvec2.x();
    double B2 = gvec2.y();
    double X1 = entryPosition.x();
    double Y1 = entryPosition.y();
    double X2 = leavePosition.x();
    double Y2 = leavePosition.y();
    double XO = (A1 * A2 * (Y2 - Y1) + A2 * B1 * X1 - A1 * B2 * X2) / (A2 * B1 - A1 * B2);
    double YO = (B1 * B2 * (X2 - X1) + B2 * A1 * Y1 - B1 * A2 * Y2) / (B2 * A1 - B1 * A2);
    GlobalVector Center_temp(XO, YO, 0);
    // Just for complex pattern
    isStraight2 = checkStraight;
    Center2 = Center_temp;

    cout << "entryPosition: " << entryPosition << endl;
    cout << "leavePosition: " << leavePosition << endl;
    cout << "Center2 is : " << Center_temp << endl;

    double R1 = GlobalVector((entryPosition.x() - Center_temp.x()),
                             (entryPosition.y() - Center_temp.y()),
                             (entryPosition.z() - Center_temp.z()))
                    .perp();
    double R2 = GlobalVector((leavePosition.x() - Center_temp.x()),
                             (leavePosition.y() - Center_temp.y()),
                             (leavePosition.z() - Center_temp.z()))
                    .perp();
    double meanR = (R1 + R2) / 2;
    double deltaR = sqrt(((R1 - meanR) * (R1 - meanR) + (R2 - meanR) * (R2 - meanR)) / 2);
    meanRadius2 = meanR;
    cout << "R1 is " << R1 << ", R2 is " << R2 << endl;
    cout << "Mean radius is " << meanR << endl;
    cout << "Delta R is " << deltaR << endl;
    double deltaPhi =
        fabs(((leavePosition - GlobalPoint(XO, YO, 0)).phi() - (entryPosition - GlobalPoint(XO, YO, 0)).phi()).value());
    S += meanR * deltaPhi;
    lastPhi = seg2.phi().value();
  }

  // Unset the pattern estimation signa
  isPatternChecked = false;
}

bool RPCSeedPattern::checkSegment() const {
  bool isFit = true;
  unsigned int count = 0;
  // first 4 recHits should be located in RB1 and RB2
  for (ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin(); iter != theRecHits.end(); iter++) {
    count++;
    const GeomDet* Detector = (*iter)->det();
    if (dynamic_cast<const RPCChamber*>(Detector) != nullptr) {
      const RPCChamber* RPCCh = dynamic_cast<const RPCChamber*>(Detector);
      RPCDetId RPCId = RPCCh->id();
      int Region = RPCId.region();
      unsigned int Station = RPCId.station();
      //int Layer = RPCId.layer();
      if (count <= 4) {
        if (Region != 0)
          isFit = false;
        if (Station > 2)
          isFit = false;
      }
    }
  }
  // more than 4 recHits for pattern building
  if (count <= 4)
    isFit = false;
  cout << "Check for segment fit: " << isFit << endl;
  return isFit;
}

MuonTransientTrackingRecHit::ConstMuonRecHitPointer RPCSeedPattern::FirstRecHit() const { return theRecHits.front(); }

MuonTransientTrackingRecHit::ConstMuonRecHitPointer RPCSeedPattern::BestRefRecHit() const {
  ConstMuonRecHitPointer best;
  int index = 0;
  // Use the last one for recHit on last layer has minmum delta Z for barrel or delta R for endcap while calculating the momentum
  // But for Algorithm 3 we use the 4th recHit on the 2nd segment for more accurate
  for (ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin(); iter != theRecHits.end(); iter++) {
    if (AlgorithmType != 3)
      best = (*iter);
    else if (index < 4)
      best = (*iter);
    index++;
  }
  return best;
}

double RPCSeedPattern::getDistance(const ConstMuonRecHitPointer& precHit, const GlobalVector& Center) const {
  return sqrt((precHit->globalPosition().x() - Center.x()) * (precHit->globalPosition().x() - Center.x()) +
              (precHit->globalPosition().y() - Center.y()) * (precHit->globalPosition().y() - Center.y()));
}

bool RPCSeedPattern::checkStraightwithThreerecHits(ConstMuonRecHitPointer (&precHit)[3], double MinDeltaPhi) const {
  GlobalVector segvec1 = precHit[1]->globalPosition() - precHit[0]->globalPosition();
  GlobalVector segvec2 = precHit[2]->globalPosition() - precHit[1]->globalPosition();
  double dPhi = (segvec2.phi() - segvec1.phi()).value();
  if (fabs(dPhi) > MinDeltaPhi) {
    cout << "Part is estimate to be not straight" << endl;
    return false;
  } else {
    cout << "Part is estimate to be straight" << endl;
    return true;
  }
}

GlobalVector RPCSeedPattern::computePtwithThreerecHits(double& pt,
                                                       double& pt_err,
                                                       ConstMuonRecHitPointer (&precHit)[3]) const {
  double x[3], y[3];
  x[0] = precHit[0]->globalPosition().x();
  y[0] = precHit[0]->globalPosition().y();
  x[1] = precHit[1]->globalPosition().x();
  y[1] = precHit[1]->globalPosition().y();
  x[2] = precHit[2]->globalPosition().x();
  y[2] = precHit[2]->globalPosition().y();
  double A = (y[2] - y[1]) / (x[2] - x[1]) - (y[1] - y[0]) / (x[1] - x[0]);
  double TYO = (x[2] - x[0]) / A + (y[2] * y[2] - y[1] * y[1]) / ((x[2] - x[1]) * A) -
               (y[1] * y[1] - y[0] * y[0]) / ((x[1] - x[0]) * A);
  double TXO = (x[2] + x[1]) + (y[2] * y[2] - y[1] * y[1]) / (x[2] - x[1]) - TYO * (y[2] - y[1]) / (x[2] - x[1]);
  double XO = 0.5 * TXO;
  double YO = 0.5 * TYO;
  double R2 = (x[0] - XO) * (x[0] - XO) + (y[0] - YO) * (y[0] - YO);
  cout << "R2 is " << R2 << endl;
  // How this algorithm get the pt without magnetic field??
  pt = 0.01 * sqrt(R2) * 2 * 0.3;
  cout << "pt is " << pt << endl;
  GlobalVector Center(XO, YO, 0);
  return Center;
}

bool RPCSeedPattern::checkStraightwithSegment(const RPCSegment& Segment1,
                                              const RPCSegment& Segment2,
                                              double MinDeltaPhi) const {
  GlobalVector segvec1 = (Segment1.second)->globalPosition() - (Segment1.first)->globalPosition();
  GlobalVector segvec2 = (Segment2.second)->globalPosition() - (Segment2.first)->globalPosition();
  GlobalVector segvec3 = (Segment2.first)->globalPosition() - (Segment1.first)->globalPosition();
  // compare segvec 1&2 for paralle, 1&3 for straight
  double dPhi1 = (segvec2.phi() - segvec1.phi()).value();
  double dPhi2 = (segvec3.phi() - segvec1.phi()).value();
  cout << "Checking straight with 2 segments. dPhi1: " << dPhi1 << ", dPhi2: " << dPhi2 << endl;
  cout << "Checking straight with 2 segments. dPhi1 in degree: " << dPhi1 * 180 / 3.1415926
       << ", dPhi2 in degree: " << dPhi2 * 180 / 3.1415926 << endl;
  if (fabs(dPhi1) > MinDeltaPhi || fabs(dPhi2) > MinDeltaPhi) {
    cout << "Segment is estimate to be not straight" << endl;
    return false;
  } else {
    cout << "Segment is estimate to be straight" << endl;
    return true;
  }
}

GlobalVector RPCSeedPattern::computePtwithSegment(const RPCSegment& Segment1, const RPCSegment& Segment2) const {
  GlobalVector segvec1 = (Segment1.second)->globalPosition() - (Segment1.first)->globalPosition();
  GlobalVector segvec2 = (Segment2.second)->globalPosition() - (Segment2.first)->globalPosition();
  GlobalPoint Point1(((Segment1.second)->globalPosition().x() + (Segment1.first)->globalPosition().x()) / 2,
                     ((Segment1.second)->globalPosition().y() + (Segment1.first)->globalPosition().y()) / 2,
                     ((Segment1.second)->globalPosition().z() + (Segment1.first)->globalPosition().z()) / 2);
  GlobalPoint Point2(((Segment2.second)->globalPosition().x() + (Segment2.first)->globalPosition().x()) / 2,
                     ((Segment2.second)->globalPosition().y() + (Segment2.first)->globalPosition().y()) / 2,
                     ((Segment2.second)->globalPosition().z() + (Segment2.first)->globalPosition().z()) / 2);
  GlobalVector vecZ(0, 0, 1);
  GlobalVector gvec1 = segvec1.cross(vecZ);
  GlobalVector gvec2 = segvec2.cross(vecZ);
  double A1 = gvec1.x();
  double B1 = gvec1.y();
  double A2 = gvec2.x();
  double B2 = gvec2.y();
  double X1 = Point1.x();
  double Y1 = Point1.y();
  double X2 = Point2.x();
  double Y2 = Point2.y();
  double XO = (A1 * A2 * (Y2 - Y1) + A2 * B1 * X1 - A1 * B2 * X2) / (A2 * B1 - A1 * B2);
  double YO = (B1 * B2 * (X2 - X1) + B2 * A1 * Y1 - B1 * A2 * Y2) / (B2 * A1 - B1 * A2);
  GlobalVector Center(XO, YO, 0);
  return Center;
}

bool RPCSeedPattern::checkStraightwithThreerecHits(double (&x)[3], double (&y)[3], double MinDeltaPhi) const {
  GlobalVector segvec1((x[1] - x[0]), (y[1] - y[0]), 0);
  GlobalVector segvec2((x[2] - x[1]), (y[2] - y[1]), 0);
  double dPhi = (segvec2.phi() - segvec1.phi()).value();
  if (fabs(dPhi) > MinDeltaPhi) {
    cout << "Part is estimate to be not straight" << endl;
    return false;
  } else {
    cout << "Part is estimate to be straight" << endl;
    return true;
  }
}

GlobalVector RPCSeedPattern::computePtWithThreerecHits(double& pt,
                                                       double& pt_err,
                                                       double (&x)[3],
                                                       double (&y)[3]) const {
  double A = (y[2] - y[1]) / (x[2] - x[1]) - (y[1] - y[0]) / (x[1] - x[0]);
  double TYO = (x[2] - x[0]) / A + (y[2] * y[2] - y[1] * y[1]) / ((x[2] - x[1]) * A) -
               (y[1] * y[1] - y[0] * y[0]) / ((x[1] - x[0]) * A);
  double TXO = (x[2] + x[1]) + (y[2] * y[2] - y[1] * y[1]) / (x[2] - x[1]) - TYO * (y[2] - y[1]) / (x[2] - x[1]);
  double XO = 0.5 * TXO;
  double YO = 0.5 * TYO;
  double R2 = (x[0] - XO) * (x[0] - XO) + (y[0] - YO) * (y[0] - YO);
  cout << "R2 is " << R2 << endl;
  // How this algorithm get the pt without magnetic field??
  pt = 0.01 * sqrt(R2) * 2 * 0.3;
  cout << "pt is " << pt << endl;
  GlobalVector Center(XO, YO, 0);
  return Center;
}

void RPCSeedPattern::checkSimplePattern(const MagneticField& Field) {
  if (isPatternChecked == true)
    return;

  unsigned int NumberofHitsinSeed = nrhit();

  // Print the recHit's position
  for (ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin(); iter != theRecHits.end(); iter++)
    cout << "Position of recHit is: " << (*iter)->globalPosition() << endl;

  // Get magnetice field information
  std::vector<double> BzValue;
  for (ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin(); iter != (theRecHits.end() - 1); iter++) {
    GlobalPoint gpFirst = (*iter)->globalPosition();
    GlobalPoint gpLast = (*(iter + 1))->globalPosition();
    GlobalPoint* gp = new GlobalPoint[sampleCount];
    double dx = (gpLast.x() - gpFirst.x()) / (sampleCount + 1);
    double dy = (gpLast.y() - gpFirst.y()) / (sampleCount + 1);
    double dz = (gpLast.z() - gpFirst.z()) / (sampleCount + 1);
    for (unsigned int index = 0; index < sampleCount; index++) {
      gp[index] = GlobalPoint(
          (gpFirst.x() + dx * (index + 1)), (gpFirst.y() + dy * (index + 1)), (gpFirst.z() + dz * (index + 1)));
      GlobalVector MagneticVec_temp = Field.inTesla(gp[index]);
      cout << "Sampling magnetic field : " << MagneticVec_temp << endl;
      BzValue.push_back(MagneticVec_temp.z());
    }
    delete[] gp;
  }
  meanBz = 0.;
  for (unsigned int index = 0; index < BzValue.size(); index++)
    meanBz += BzValue[index];
  meanBz /= BzValue.size();
  cout << "Mean Bz is " << meanBz << endl;
  deltaBz = 0.;
  for (unsigned int index = 0; index < BzValue.size(); index++)
    deltaBz += (BzValue[index] - meanBz) * (BzValue[index] - meanBz);
  deltaBz /= BzValue.size();
  deltaBz = sqrt(deltaBz);
  cout << "delata Bz is " << deltaBz << endl;

  // Set isGoodPattern to default true and check the failure
  isGoodPattern = 1;

  // Check the Z direction
  if (fabs((*(theRecHits.end() - 1))->globalPosition().z() - (*(theRecHits.begin()))->globalPosition().z()) > ZError) {
    if (((*(theRecHits.end() - 1))->globalPosition().z() - (*(theRecHits.begin()))->globalPosition().z()) > ZError)
      isParralZ = 1;
    else
      isParralZ = -1;
  } else
    isParralZ = 0;

  cout << " Check isParralZ is :" << isParralZ << endl;
  for (ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin(); iter != (theRecHits.end() - 1); iter++) {
    if (isParralZ == 0) {
      if (fabs((*(iter + 1))->globalPosition().z() - (*iter)->globalPosition().z()) > ZError) {
        cout << "Pattern find error in Z direction: wrong perpendicular direction" << endl;
        isGoodPattern = 0;
      }
    } else {
      if ((int)(((*(iter + 1))->globalPosition().z() - (*iter)->globalPosition().z()) / ZError) * isParralZ < 0) {
        cout << "Pattern find error in Z direction: wrong Z direction" << endl;
        isGoodPattern = 0;
      }
    }
  }

  // Check pattern
  if (isStraight == false) {
    // Check clockwise direction
    GlobalVector* vec = new GlobalVector[NumberofHitsinSeed];
    unsigned int index = 0;
    for (ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin(); iter != theRecHits.end(); iter++) {
      GlobalVector vec_temp(((*iter)->globalPosition().x() - Center.x()),
                            ((*iter)->globalPosition().y() - Center.y()),
                            ((*iter)->globalPosition().z() - Center.z()));
      vec[index] = vec_temp;
      index++;
    }
    isClockwise = 0;
    for (unsigned int index = 0; index < (NumberofHitsinSeed - 1); index++) {
      // Check phi direction, all sub-dphi direction should be the same
      if ((vec[index + 1].phi() - vec[index].phi()) > 0)
        isClockwise--;
      else
        isClockwise++;
      cout << "Current isClockwise is : " << isClockwise << endl;
    }
    cout << "Check isClockwise is : " << isClockwise << endl;
    if ((unsigned int)abs(isClockwise) != (NumberofHitsinSeed - 1)) {
      cout << "Pattern find error in Phi direction" << endl;
      isGoodPattern = 0;
      isClockwise = 0;
    } else
      isClockwise /= abs(isClockwise);
    delete[] vec;

    // Get meanPt and meanSpt
    double deltaRwithBz = fabs(deltaBz * meanRadius / meanBz);
    cout << "deltaR with Bz is " << deltaRwithBz << endl;

    if (isClockwise == 0)
      meanPt = upper_limit_pt;
    else
      meanPt = 0.01 * meanRadius * meanBz * 0.3 * isClockwise;
    if (fabs(meanPt) > upper_limit_pt)
      meanPt = upper_limit_pt * meanPt / fabs(meanPt);
    cout << " meanRadius is " << meanRadius << endl;
    cout << " meanPt is " << meanPt << endl;

    double deltaR = 0.;
    for (ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin(); iter != theRecHits.end(); iter++) {
      deltaR += (getDistance(*iter, Center) - meanRadius) * (getDistance(*iter, Center) - meanRadius);
    }
    deltaR = deltaR / NumberofHitsinSeed;
    deltaR = sqrt(deltaR);
    //meanSpt = 0.01 * deltaR * meanBz * 0.3;
    meanSpt = deltaR;
    cout << "DeltaR is " << deltaR << endl;
    if (deltaR > deltaRThreshold) {
      cout << "Pattern find error: deltaR over threshold" << endl;
      isGoodPattern = 0;
    }
  } else {
    // Just set pattern to be straight
    isClockwise = 0;
    meanPt = upper_limit_pt;
    // Set the straight pattern with lowest priority among good pattern
    meanSpt = deltaRThreshold;
  }
  cout << "III--> Seed Pt : " << meanPt << endl;
  cout << "III--> Pattern is: " << isGoodPattern << endl;

  // Set the pattern estimation signal
  isPatternChecked = true;
}

void RPCSeedPattern::checkSegmentAlgorithmSpecial(MagneticField const& Field, RPCGeometry const& rpcGeom) {
  if (isPatternChecked == true)
    return;

  if (!checkSegment()) {
    isPatternChecked = true;
    isGoodPattern = -1;
    return;
  }

  // Set isGoodPattern to default true and check the failure
  isGoodPattern = 1;

  // Print the recHit's position
  for (ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin(); iter != theRecHits.end(); iter++)
    cout << "Position of recHit is: " << (*iter)->globalPosition() << endl;

  // Check the Z direction
  if (fabs((*(theRecHits.end() - 1))->globalPosition().z() - (*(theRecHits.begin()))->globalPosition().z()) > ZError) {
    if (((*(theRecHits.end() - 1))->globalPosition().z() - (*(theRecHits.begin()))->globalPosition().z()) > ZError)
      isParralZ = 1;
    else
      isParralZ = -1;
  } else
    isParralZ = 0;

  cout << " Check isParralZ is :" << isParralZ << endl;
  for (ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin(); iter != (theRecHits.end() - 1); iter++) {
    if (isParralZ == 0) {
      if (fabs((*(iter + 1))->globalPosition().z() - (*iter)->globalPosition().z()) > ZError) {
        cout << "Pattern find error in Z direction: wrong perpendicular direction" << endl;
        isGoodPattern = 0;
      }
    } else {
      if ((int)(((*(iter + 1))->globalPosition().z() - (*iter)->globalPosition().z()) / ZError) * isParralZ < 0) {
        cout << "Pattern find error in Z direction: wrong Z direction" << endl;
        isGoodPattern = 0;
      }
    }
  }

  // Check the pattern
  if (isStraight2 == true) {
    // Set pattern to be straight
    isClockwise = 0;
    meanPt = upper_limit_pt;
    // Set the straight pattern with lowest priority among good pattern
    meanSpt = deltaRThreshold;

    // Extrapolate to other recHits and check deltaR
    GlobalVector startSegment = (SegmentRB[1].second)->globalPosition() - (SegmentRB[1].first)->globalPosition();
    GlobalPoint startPosition = (SegmentRB[1].first)->globalPosition();
    GlobalVector startMomentum = startSegment * (meanPt / startSegment.perp());
    unsigned int index = 0;

    for (ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin(); iter != theRecHits.end(); iter++) {
      if (index < 4) {
        index++;
        continue;
      }
      double tracklength = 0;
      cout << "Now checking recHit " << index << endl;
      double Distance = extropolateStep(startPosition, startMomentum, iter, isClockwise, tracklength, Field, rpcGeom);
      cout << "Final distance is " << Distance << endl;
      if (Distance > MaxRSD) {
        cout << "Pattern find error in distance for other recHits: " << Distance << endl;
        isGoodPattern = 0;
      }
      index++;
    }
  } else {
    // Get clockwise direction
    GlobalVector vec[2];
    vec[0] = GlobalVector(
        (entryPosition.x() - Center2.x()), (entryPosition.y() - Center2.y()), (entryPosition.z() - Center2.z()));
    vec[1] = GlobalVector(
        (leavePosition.x() - Center2.x()), (leavePosition.y() - Center2.y()), (leavePosition.z() - Center2.z()));
    isClockwise = 0;
    if ((vec[1].phi() - vec[0].phi()).value() > 0)
      isClockwise = -1;
    else
      isClockwise = 1;

    cout << "Check isClockwise is : " << isClockwise << endl;

    // Get meanPt
    meanPt = 0.01 * meanRadius2 * meanMagneticField2.z() * 0.3 * isClockwise;
    //meanPt = 0.01 * meanRadius2[0] * (-3.8) * 0.3 * isClockwise;
    cout << " meanRadius is " << meanRadius2 << ", with meanBz " << meanMagneticField2.z() << endl;
    cout << " meanPt is " << meanPt << endl;
    if (fabs(meanPt) > upper_limit_pt)
      meanPt = upper_limit_pt * meanPt / fabs(meanPt);

    // Check the initial 3 segments
    cout << "entryPosition: " << entryPosition << endl;
    cout << "leavePosition: " << leavePosition << endl;
    cout << "Center2 is : " << Center2 << endl;
    double R1 = vec[0].perp();
    double R2 = vec[1].perp();
    double deltaR = sqrt(((R1 - meanRadius2) * (R1 - meanRadius2) + (R2 - meanRadius2) * (R2 - meanRadius2)) / 2);
    meanSpt = deltaR;
    cout << "R1 is " << R1 << ", R2 is " << R2 << endl;
    cout << "Delta R for the initial 3 segments is " << deltaR << endl;
    if (deltaR > deltaRThreshold) {
      cout << "Pattern find error in delta R for the initial 3 segments" << endl;
      isGoodPattern = 0;
    }

    // Extrapolate to other recHits and check deltaR
    GlobalVector startSegment = (SegmentRB[1].second)->globalPosition() - (SegmentRB[1].first)->globalPosition();
    GlobalPoint startPosition = (SegmentRB[1].first)->globalPosition();
    GlobalVector startMomentum = startSegment * (fabs(meanPt) / startSegment.perp());
    unsigned int index = 0;
    for (ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin(); iter != theRecHits.end(); iter++) {
      if (index < 4) {
        index++;
        continue;
      }
      double tracklength = 0;
      cout << "Now checking recHit " << index << endl;
      double Distance = extropolateStep(startPosition, startMomentum, iter, isClockwise, tracklength, Field, rpcGeom);
      cout << "Final distance is " << Distance << endl;
      if (Distance > MaxRSD) {
        cout << "Pattern find error in distance for other recHits: " << Distance << endl;
        isGoodPattern = 0;
      }
      index++;
    }
  }

  cout << "Checking finish, isGoodPattern now is " << isGoodPattern << endl;
  // Set the pattern estimation signal
  isPatternChecked = true;
}

double RPCSeedPattern::extropolateStep(const GlobalPoint& startPosition,
                                       const GlobalVector& startMomentum,
                                       ConstMuonRecHitContainer::const_iterator iter,
                                       const int ClockwiseDirection,
                                       double& tracklength,
                                       const MagneticField& Field,
                                       const RPCGeometry& rpcGeometry) {
  cout << "Extrapolating the track to check the pattern" << endl;
  tracklength = 0;
  // Get the iter recHit's detector geometry
  DetId hitDet = (*iter)->hit()->geographicalId();
  RPCDetId RPCId = RPCDetId(hitDet.rawId());
  //const RPCChamber* hitRPC = dynamic_cast<const RPCChamber*>(hitDet);

  const BoundPlane RPCSurface = rpcGeometry.chamber(RPCId)->surface();
  double startSide = RPCSurface.localZ(startPosition);
  cout << "Start side : " << startSide;

  GlobalPoint currentPosition = startPosition;
  GlobalVector currentMomentum = startMomentum;
  GlobalVector ZDirection(0, 0, 1);

  // Use the perp other than mag, since initial segment might have small value while final recHit have large difference value at Z direction
  double currentDistance = ((GlobalVector)(currentPosition - (*iter)->globalPosition())).perp();
  cout << "Start current position is : " << currentPosition << endl;
  cout << "Start current Momentum is: " << currentMomentum.mag() << ", in vector: " << currentMomentum << endl;
  cout << "Start current distance is " << currentDistance << endl;
  cout << "Start current radius is " << currentPosition.perp() << endl;
  cout << "Destination radius is " << (*iter)->globalPosition().perp() << endl;

  // Judge roughly if the stepping cross the Det surface of the recHit
  //while((currentPosition.perp() < ((*iter)->globalPosition().perp())))
  double currentDistance_next = currentDistance;
  do {
    currentDistance = currentDistance_next;
    if (ClockwiseDirection == 0) {
      currentPosition += currentMomentum.unit() * stepLength;
    } else {
      double Bz = Field.inTesla(currentPosition).z();
      double Radius = currentMomentum.perp() / fabs(Bz * 0.01 * 0.3);
      double deltaPhi = (stepLength * currentMomentum.perp() / currentMomentum.mag()) / Radius;

      // Get the center for current step
      GlobalVector currentPositiontoCenter = currentMomentum.unit().cross(ZDirection);
      currentPositiontoCenter *= Radius;
      // correction of ClockwiseDirection correction
      currentPositiontoCenter *= ClockwiseDirection;
      // continue to get the center for current step
      GlobalPoint currentCenter = currentPosition;
      currentCenter += currentPositiontoCenter;

      // Get the next step position
      GlobalVector CentertocurrentPosition = (GlobalVector)(currentPosition - currentCenter);
      double Phi = CentertocurrentPosition.phi().value();
      Phi += deltaPhi * (-1) * ClockwiseDirection;
      double deltaZ = stepLength * currentMomentum.z() / currentMomentum.mag();
      GlobalVector CentertonewPosition(GlobalVector::Cylindrical(CentertocurrentPosition.perp(), Phi, deltaZ));
      double PtPhi = currentMomentum.phi().value();
      PtPhi += deltaPhi * (-1) * ClockwiseDirection;
      currentMomentum = GlobalVector(GlobalVector::Cylindrical(currentMomentum.perp(), PtPhi, currentMomentum.z()));
      currentPosition = currentCenter + CentertonewPosition;
    }

    // count the total step length
    tracklength += stepLength * currentMomentum.perp() / currentMomentum.mag();

    // Get the next step distance
    double currentSide = RPCSurface.localZ(currentPosition);
    cout << "Stepping current side : " << currentSide << endl;
    cout << "Stepping current position is: " << currentPosition << endl;
    cout << "Stepping current Momentum is: " << currentMomentum.mag() << ", in vector: " << currentMomentum << endl;
    currentDistance_next = ((GlobalVector)(currentPosition - (*iter)->globalPosition())).perp();
    cout << "Stepping current distance is " << currentDistance << endl;
    cout << "Stepping current radius is " << currentPosition.perp() << endl;
  } while (currentDistance_next < currentDistance);

  return currentDistance;
}

RPCSeedPattern::weightedTrajectorySeed RPCSeedPattern::createFakeSeed(int& isGoodSeed, const MagneticField& Field) {
  // Create a fake seed and return
  cout << "Now create a fake seed" << endl;
  isPatternChecked = true;
  isGoodPattern = -1;
  isStraight = true;
  meanPt = upper_limit_pt;
  meanSpt = 0;
  Charge = 0;
  isClockwise = 0;
  isParralZ = 0;
  meanRadius = -1;
  //return createSeed(isGoodSeed, eSetup);

  // Get the reference recHit, DON'T use the recHit on 1st layer(inner most layer)
  const ConstMuonRecHitPointer best = BestRefRecHit();

  Momentum = GlobalVector(0, 0, 0);
  LocalPoint segPos = best->localPosition();
  LocalVector segDirFromPos = best->det()->toLocal(Momentum);
  LocalTrajectoryParameters param(segPos, segDirFromPos, Charge);

  //AlgebraicVector t(4);
  AlgebraicSymMatrix mat(5, 0);
  mat = best->parametersError().similarityT(best->projectionMatrix());
  mat[0][0] = meanSpt;
  LocalTrajectoryError error(asSMatrix<5>(mat));

  TrajectoryStateOnSurface tsos(param, error, best->det()->surface(), &Field);

  DetId id = best->geographicalId();

  PTrajectoryStateOnDet seedTSOS = trajectoryStateTransform::persistentState(tsos, id.rawId());

  edm::OwnVector<TrackingRecHit> container;
  for (ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin(); iter != theRecHits.end(); iter++)
    container.push_back((*iter)->hit()->clone());

  TrajectorySeed theSeed(seedTSOS, container, alongMomentum);
  weightedTrajectorySeed theweightedSeed;
  theweightedSeed.first = theSeed;
  theweightedSeed.second = meanSpt;
  isGoodSeed = isGoodPattern;

  return theweightedSeed;
}

RPCSeedPattern::weightedTrajectorySeed RPCSeedPattern::createSeed(int& isGoodSeed, const MagneticField& Field) {
  if (isPatternChecked == false || isGoodPattern == -1) {
    cout << "Pattern is not yet checked! Create a fake seed instead!" << endl;
    return createFakeSeed(isGoodSeed, Field);
  }

  MuonPatternRecoDumper debug;

  //double theMinMomentum = 3.0;
  //if(fabs(meanPt) < lower_limit_pt)
  //meanPt = lower_limit_pt * meanPt / fabs(meanPt);

  // For pattern we use is Clockwise other than isStraight to estimate charge
  if (isClockwise == 0)
    Charge = 0;
  else
    Charge = (int)(meanPt / fabs(meanPt));

  // Get the reference recHit, DON'T use the recHit on 1st layer(inner most layer)
  const ConstMuonRecHitPointer best = BestRefRecHit();
  const ConstMuonRecHitPointer first = FirstRecHit();

  if (isClockwise != 0) {
    if (AlgorithmType != 3) {
      // Get the momentum on reference recHit
      GlobalVector vecRef1((first->globalPosition().x() - Center.x()),
                           (first->globalPosition().y() - Center.y()),
                           (first->globalPosition().z() - Center.z()));
      GlobalVector vecRef2((best->globalPosition().x() - Center.x()),
                           (best->globalPosition().y() - Center.y()),
                           (best->globalPosition().z() - Center.z()));

      double deltaPhi = (vecRef2.phi() - vecRef1.phi()).value();
      double deltaS = meanRadius * fabs(deltaPhi);
      double deltaZ = best->globalPosition().z() - first->globalPosition().z();

      GlobalVector vecZ(0, 0, 1);
      GlobalVector vecPt = (vecRef2.unit()).cross(vecZ);
      if (isClockwise == -1)
        vecPt *= -1;
      vecPt *= deltaS;
      Momentum = GlobalVector(0, 0, deltaZ);
      Momentum += vecPt;
      Momentum *= fabs(meanPt / deltaS);
    } else {
      double deltaZ = best->globalPosition().z() - first->globalPosition().z();
      Momentum = GlobalVector(GlobalVector::Cylindrical(S, lastPhi, deltaZ));
      Momentum *= fabs(meanPt / S);
    }
  } else {
    Momentum = best->globalPosition() - first->globalPosition();
    double deltaS = Momentum.perp();
    Momentum *= fabs(meanPt / deltaS);
  }
  LocalPoint segPos = best->localPosition();
  LocalVector segDirFromPos = best->det()->toLocal(Momentum);
  LocalTrajectoryParameters param(segPos, segDirFromPos, Charge);

  LocalTrajectoryError error = getSpecialAlgorithmErrorMatrix(first, best);

  TrajectoryStateOnSurface tsos(param, error, best->det()->surface(), &Field);
  cout << "Trajectory State on Surface before the extrapolation" << endl;
  cout << debug.dumpTSOS(tsos);
  DetId id = best->geographicalId();
  cout << "The RecSegment relies on: " << endl;
  cout << debug.dumpMuonId(id);
  cout << debug.dumpTSOS(tsos);

  PTrajectoryStateOnDet const& seedTSOS = trajectoryStateTransform::persistentState(tsos, id.rawId());

  edm::OwnVector<TrackingRecHit> container;
  for (ConstMuonRecHitContainer::const_iterator iter = theRecHits.begin(); iter != theRecHits.end(); iter++) {
    // This casting withou clone will cause memory overflow when used in push_back
    // Since container's deconstructor functiion free the pointer menber!
    //TrackingRecHit* pt = dynamic_cast<TrackingRecHit*>(&*(*iter));
    //cout << "Push recHit type " << pt->getType() << endl;
    container.push_back((*iter)->hit()->clone());
  }

  TrajectorySeed theSeed(seedTSOS, container, alongMomentum);
  weightedTrajectorySeed theweightedSeed;
  theweightedSeed.first = theSeed;
  theweightedSeed.second = meanSpt;
  isGoodSeed = isGoodPattern;

  return theweightedSeed;
}

LocalTrajectoryError RPCSeedPattern::getSpecialAlgorithmErrorMatrix(const ConstMuonRecHitPointer& first,
                                                                    const ConstMuonRecHitPointer& best) {
  LocalTrajectoryError Error;
  double dXdZ = 0;
  double dYdZ = 0;
  double dP = 0;
  AlgebraicSymMatrix mat(5, 0);
  mat = best->parametersError().similarityT(best->projectionMatrix());
  if (AlgorithmType != 3) {
    GlobalVector vecRef1((first->globalPosition().x() - Center.x()),
                         (first->globalPosition().y() - Center.y()),
                         (first->globalPosition().z() - Center.z()));
    GlobalVector vecRef2((best->globalPosition().x() - Center.x()),
                         (best->globalPosition().y() - Center.y()),
                         (best->globalPosition().z() - Center.z()));
    double deltaPhi = (vecRef2.phi() - vecRef1.phi()).value();
    double L = meanRadius * fabs(deltaPhi);
    double N = nrhit();
    double A_N = 180 * N * N * N / ((N - 1) * (N + 1) * (N + 2) * (N + 3));
    double sigma_x = sqrt(mat[3][3]);
    double betaovergame = Momentum.mag() / 0.1066;
    double beta = sqrt((betaovergame * betaovergame) / (1 + betaovergame * betaovergame));
    double dPt = meanPt * (0.0136 * sqrt(1 / 100) * sqrt(4 * A_N / N) / (beta * 0.3 * meanBz * L) +
                           sigma_x * meanPt * sqrt(4 * A_N) / (0.3 * meanBz * L * L));
    double dP = dPt * Momentum.mag() / meanPt;
    mat[0][0] = (dP * dP) / (Momentum.mag() * Momentum.mag() * Momentum.mag() * Momentum.mag());
    mat[1][1] = dXdZ * dXdZ;
    mat[2][2] = dYdZ * dYdZ;
    Error = LocalTrajectoryError(asSMatrix<5>(mat));
  } else {
    AlgebraicSymMatrix mat0(5, 0);
    mat0 = (SegmentRB[0].first)->parametersError().similarityT((SegmentRB[0].first)->projectionMatrix());
    double dX0 = sqrt(mat0[3][3]);
    double dY0 = sqrt(mat0[4][4]);
    AlgebraicSymMatrix mat1(5, 0);
    mat1 = (SegmentRB[0].second)->parametersError().similarityT((SegmentRB[0].second)->projectionMatrix());
    double dX1 = sqrt(mat1[3][3]);
    double dY1 = sqrt(mat1[4][4]);
    AlgebraicSymMatrix mat2(5, 0);
    mat2 = (SegmentRB[1].first)->parametersError().similarityT((SegmentRB[1].first)->projectionMatrix());
    double dX2 = sqrt(mat2[3][3]);
    double dY2 = sqrt(mat2[4][4]);
    AlgebraicSymMatrix mat3(5, 0);
    mat3 = (SegmentRB[1].second)->parametersError().similarityT((SegmentRB[1].second)->projectionMatrix());
    double dX3 = sqrt(mat3[3][3]);
    double dY3 = sqrt(mat3[4][4]);
    cout << "Local error for 4 recHits are: " << dX0 << ", " << dY0 << ", " << dX1 << ", " << dY1 << ", " << dX2 << ", "
         << dY2 << ", " << dX3 << ", " << dY3 << endl;
    const GeomDetUnit* refRPC1 = (SegmentRB[0].second)->detUnit();
    LocalPoint recHit0 = refRPC1->toLocal((SegmentRB[0].first)->globalPosition());
    LocalPoint recHit1 = refRPC1->toLocal((SegmentRB[0].second)->globalPosition());
    LocalVector localSegment00 = (LocalVector)(recHit1 - recHit0);
    LocalVector localSegment01 = LocalVector(localSegment00.x() + dX0 + dX1, localSegment00.y(), localSegment00.z());
    LocalVector localSegment02 = LocalVector(localSegment00.x() - dX0 - dX1, localSegment00.y(), localSegment00.z());
    GlobalVector globalSegment00 = refRPC1->toGlobal(localSegment00);
    GlobalVector globalSegment01 = refRPC1->toGlobal(localSegment01);
    GlobalVector globalSegment02 = refRPC1->toGlobal(localSegment02);

    const GeomDetUnit* refRPC2 = (SegmentRB[1].first)->detUnit();
    LocalPoint recHit2 = refRPC2->toLocal((SegmentRB[1].first)->globalPosition());
    LocalPoint recHit3 = refRPC2->toLocal((SegmentRB[1].second)->globalPosition());
    LocalVector localSegment10 = (LocalVector)(recHit3 - recHit2);
    LocalVector localSegment11 = LocalVector(localSegment10.x() + dX2 + dX3, localSegment10.y(), localSegment10.z());
    LocalVector localSegment12 = LocalVector(localSegment10.x() - dX2 - dX3, localSegment10.y(), localSegment10.z());
    GlobalVector globalSegment10 = refRPC2->toGlobal(localSegment10);
    GlobalVector globalSegment11 = refRPC2->toGlobal(localSegment11);
    GlobalVector globalSegment12 = refRPC2->toGlobal(localSegment12);

    if (isClockwise != 0) {
      GlobalVector vec[2];
      vec[0] = GlobalVector(
          (entryPosition.x() - Center2.x()), (entryPosition.y() - Center2.y()), (entryPosition.z() - Center2.z()));
      vec[1] = GlobalVector(
          (leavePosition.x() - Center2.x()), (leavePosition.y() - Center2.y()), (leavePosition.z() - Center2.z()));
      double halfPhiCenter = fabs((vec[1].phi() - vec[0].phi()).value()) / 2;
      // dPhi0 shoule be the same clockwise direction, while dPhi1 should be opposite clockwise direction, w.r.t to the track clockwise
      double dPhi0 = (((globalSegment00.phi() - globalSegment01.phi()).value() * isClockwise) > 0)
                         ? fabs((globalSegment00.phi() - globalSegment01.phi()).value())
                         : fabs((globalSegment00.phi() - globalSegment02.phi()).value());
      double dPhi1 = (((globalSegment10.phi() - globalSegment11.phi()).value() * isClockwise) < 0)
                         ? fabs((globalSegment10.phi() - globalSegment11.phi()).value())
                         : fabs((globalSegment10.phi() - globalSegment12.phi()).value());
      // For the deltaR should be kept small, we assume the delta Phi0/Phi1 should be in a same limit value
      double dPhi = (dPhi0 <= dPhi1) ? dPhi0 : dPhi1;
      cout << "DPhi for new Segment is about " << dPhi << endl;
      // Check the variance of halfPhiCenter
      double newhalfPhiCenter = ((halfPhiCenter - dPhi) > 0 ? (halfPhiCenter - dPhi) : 0);
      if (newhalfPhiCenter != 0) {
        double newmeanPt = meanPt * halfPhiCenter / newhalfPhiCenter;
        if (fabs(newmeanPt) > upper_limit_pt)
          newmeanPt = upper_limit_pt * meanPt / fabs(meanPt);
        cout << "The error is inside range. Max meanPt could be " << newmeanPt << endl;
        dP = fabs(Momentum.mag() * (newmeanPt - meanPt) / meanPt);
      } else {
        double newmeanPt = upper_limit_pt * meanPt / fabs(meanPt);
        cout << "The error is outside range. Max meanPt could be " << newmeanPt << endl;
        dP = fabs(Momentum.mag() * (newmeanPt - meanPt) / meanPt);
      }
    } else {
      double dPhi0 = (fabs((globalSegment00.phi() - globalSegment01.phi()).value()) <=
                      fabs((globalSegment00.phi() - globalSegment02.phi()).value()))
                         ? fabs((globalSegment00.phi() - globalSegment01.phi()).value())
                         : fabs((globalSegment00.phi() - globalSegment02.phi()).value());
      double dPhi1 = (fabs((globalSegment10.phi() - globalSegment11.phi()).value()) <=
                      fabs((globalSegment10.phi() - globalSegment12.phi()).value()))
                         ? fabs((globalSegment10.phi() - globalSegment11.phi()).value())
                         : fabs((globalSegment10.phi() - globalSegment12.phi()).value());
      double dPhi = (dPhi0 <= dPhi1) ? dPhi0 : dPhi1;
      GlobalVector middleSegment = leavePosition - entryPosition;
      double halfDistance = middleSegment.perp() / 2;
      double newmeanPt = halfDistance / dPhi;
      cout << "The error is for straight. Max meanPt could be " << newmeanPt << endl;
      dP = fabs(Momentum.mag() * (newmeanPt - meanPt) / meanPt);
    }

    double dXdZ1 = globalSegment11.x() / globalSegment11.z() - globalSegment10.x() / globalSegment10.z();
    double dXdZ2 = globalSegment12.x() / globalSegment12.z() - globalSegment10.x() / globalSegment10.z();
    dXdZ = (fabs(dXdZ1) >= fabs(dXdZ2)) ? dXdZ1 : dXdZ2;

    LocalVector localSegment13 = LocalVector(localSegment10.x(), localSegment10.y() + dY2 + dY3, localSegment10.z());
    LocalVector localSegment14 = LocalVector(localSegment10.x(), localSegment10.y() - dY2 - dY3, localSegment10.z());
    GlobalVector globalSegment13 = refRPC2->toGlobal(localSegment13);
    GlobalVector globalSegment14 = refRPC2->toGlobal(localSegment14);
    double dYdZ1 = globalSegment13.y() / globalSegment13.z() - globalSegment10.y() / globalSegment10.z();
    double dYdZ2 = globalSegment14.y() / globalSegment14.z() - globalSegment10.y() / globalSegment10.z();
    dYdZ = (fabs(dYdZ1) >= fabs(dYdZ2)) ? dYdZ1 : dYdZ2;

    mat[0][0] = (dP * dP) / (Momentum.mag() * Momentum.mag() * Momentum.mag() * Momentum.mag());
    mat[1][1] = dXdZ * dXdZ;
    mat[2][2] = dYdZ * dYdZ;
    Error = LocalTrajectoryError(asSMatrix<5>(mat));
  }
  return Error;
}
