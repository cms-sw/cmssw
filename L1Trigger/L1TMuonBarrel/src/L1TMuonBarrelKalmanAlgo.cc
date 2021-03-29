#include <cmath>
#include "L1Trigger/L1TMuonBarrel/interface/L1TMuonBarrelKalmanAlgo.h"
#include "ap_int.h"
#include "ap_fixed.h"

L1TMuonBarrelKalmanAlgo::L1TMuonBarrelKalmanAlgo(const edm::ParameterSet& settings)
    : verbose_(settings.getParameter<bool>("verbose")),
      lutService_(new L1TMuonBarrelKalmanLUTs(settings.getParameter<std::string>("lutFile"))),
      initK_(settings.getParameter<std::vector<double> >("initialK")),
      initK2_(settings.getParameter<std::vector<double> >("initialK2")),
      eLoss_(settings.getParameter<std::vector<double> >("eLoss")),
      aPhi_(settings.getParameter<std::vector<double> >("aPhi")),
      aPhiB_(settings.getParameter<std::vector<double> >("aPhiB")),
      aPhiBNLO_(settings.getParameter<std::vector<double> >("aPhiBNLO")),
      bPhi_(settings.getParameter<std::vector<double> >("bPhi")),
      bPhiB_(settings.getParameter<std::vector<double> >("bPhiB")),
      phiAt2_(settings.getParameter<double>("phiAt2")),

      chiSquare_(settings.getParameter<std::vector<double> >("chiSquare")),
      chiSquareCutPattern_(settings.getParameter<std::vector<int> >("chiSquareCutPattern")),
      chiSquareCutCurv_(settings.getParameter<std::vector<int> >("chiSquareCutCurvMax")),
      chiSquareCut_(settings.getParameter<std::vector<int> >("chiSquareCut")),
      trackComp_(settings.getParameter<std::vector<double> >("trackComp")),
      trackCompErr1_(settings.getParameter<std::vector<double> >("trackCompErr1")),
      trackCompErr2_(settings.getParameter<std::vector<double> >("trackCompErr2")),
      trackCompPattern_(settings.getParameter<std::vector<int> >("trackCompCutPattern")),
      trackCompCutCurv_(settings.getParameter<std::vector<int> >("trackCompCutCurvMax")),
      trackCompCut_(settings.getParameter<std::vector<int> >("trackCompCut")),
      chiSquareCutTight_(settings.getParameter<std::vector<int> >("chiSquareCutTight")),
      combos4_(settings.getParameter<std::vector<int> >("combos4")),
      combos3_(settings.getParameter<std::vector<int> >("combos3")),
      combos2_(settings.getParameter<std::vector<int> >("combos2")),
      combos1_(settings.getParameter<std::vector<int> >("combos1")),

      useOfflineAlgo_(settings.getParameter<bool>("useOfflineAlgo")),
      mScatteringPhi_(settings.getParameter<std::vector<double> >("mScatteringPhi")),
      mScatteringPhiB_(settings.getParameter<std::vector<double> >("mScatteringPhiB")),
      pointResolutionPhi_(settings.getParameter<double>("pointResolutionPhi")),
      pointResolutionPhiB_(settings.getParameter<double>("pointResolutionPhiB")),
      pointResolutionPhiBH_(settings.getParameter<std::vector<double> >("pointResolutionPhiBH")),
      pointResolutionPhiBL_(settings.getParameter<std::vector<double> >("pointResolutionPhiBL")),
      pointResolutionVertex_(settings.getParameter<double>("pointResolutionVertex"))

{}

std::pair<bool, uint> L1TMuonBarrelKalmanAlgo::getByCode(const L1MuKBMTrackCollection& tracks, int mask) {
  for (uint i = 0; i < tracks.size(); ++i) {
    printf("Code=%d, track=%d\n", tracks[i].hitPattern(), mask);
    if (tracks[i].hitPattern() == mask)
      return std::make_pair(true, i);
  }
  return std::make_pair(false, 0);
}

l1t::RegionalMuonCand L1TMuonBarrelKalmanAlgo::convertToBMTF(const L1MuKBMTrack& track) {
  //  int  K = fabs(track.curvatureAtVertex());

  //calibration
  int sign, signValid;

  if (track.curvatureAtVertex() == 0) {
    sign = 0;
    signValid = 0;
  } else if (track.curvatureAtVertex() > 0) {
    sign = 0;
    signValid = 1;
  } else {
    sign = 1;
    signValid = 1;
  }

  // if (K<22)
  //   K=22;

  // if (K>4095)
  //   K=4095;

  int pt = ptLUT(track.curvatureAtVertex());

  // int  K2 = fabs(track.curvatureAtMuon());
  // if (K2<22)
  //   K2=22;

  // if (K2>4095)
  //   K2=4095;
  int pt2 = ptLUT(track.curvatureAtMuon()) / 2;
  int eta = track.hasFineEta() ? track.fineEta() : track.coarseEta();

  //  int phi2 = track.phiAtMuon()>>2;
  //  float phi_f = float(phi2);
  //double kPhi = 57.2958/0.625/1024.;
  //int phi = 24+int(floor(kPhi*phi_f));
  //  if (phi >  69) phi =  69;
  //  if (phi < -8) phi = -8;
  int phi2 = track.phiAtMuon() >> 2;
  int tmp = fp_product(0.0895386, phi2, 14);
  int phi = 24 + tmp;

  int processor = track.sector();
  int HF = track.hasFineEta();

  int quality = 12 | (rank(track) >> 6);

  int dxy = abs(track.dxy()) >> 8;
  if (dxy > 3)
    dxy = 3;

  int trackAddr;
  std::map<int, int> addr = trackAddress(track, trackAddr);

  l1t::RegionalMuonCand muon(pt, phi, eta, sign, signValid, quality, processor, l1t::bmtf, addr);
  muon.setHwHF(HF);
  muon.setHwPtUnconstrained(pt2);
  muon.setHwDXY(dxy);

  //nw the words!
  uint32_t word1 = pt;
  word1 = word1 | quality << 9;
  word1 = word1 | (twosCompToBits(eta)) << 13;
  word1 = word1 | HF << 22;
  word1 = word1 | (twosCompToBits(phi)) << 23;

  uint32_t word2 = sign;
  word2 = word2 | signValid << 1;
  word2 = word2 | dxy << 2;
  word2 = word2 | trackAddr << 4;
  word2 = word2 | (twosCompToBits(track.wheel())) << 20;
  word2 = word2 | pt2 << 23;
  muon.setDataword(word2, word1);
  return muon;
}

void L1TMuonBarrelKalmanAlgo::addBMTFMuon(int bx,
                                          const L1MuKBMTrack& track,
                                          std::unique_ptr<l1t::RegionalMuonCandBxCollection>& out) {
  out->push_back(bx, convertToBMTF(track));
}

// std::pair<bool,uint> L1TMuonBarrelKalmanAlgo::match(const L1MuKBMTCombinedStubRef& seed, const L1MuKBMTCombinedStubRefVector& stubs,int step) {
//   L1MuKBMTCombinedStubRefVector selected;

//   bool found=false;
//   uint best=0;
//   int distance=100000;
//   uint N=0;
//   for (const auto& stub :stubs)  {
//     N=N+1;
//     if (stub->stNum()!=step)
//       continue;

//     int d = fabs(wrapAround(((correctedPhi(seed,seed->scNum())-correctedPhi(stub,seed->scNum()))>>3),1024));
//     if (d<distance) {
//       distance = d;
//       best=N-1;
//       found=true;
//     }
//   }
//   return std::make_pair(found,best);
// }

uint L1TMuonBarrelKalmanAlgo::matchAbs(std::map<uint, uint>& info, uint i, uint j) {
  if (info[i] < info[j])
    return i;
  else
    return j;
}

std::pair<bool, uint> L1TMuonBarrelKalmanAlgo::match(const L1MuKBMTCombinedStubRef& seed,
                                                     const L1MuKBMTCombinedStubRefVector& stubs,
                                                     int step) {
  L1MuKBMTCombinedStubRefVector selected;

  std::map<uint, uint> diffInfo;
  for (uint i = 0; i < 12; ++i) {
    diffInfo[i] = 60000;
  }

  std::map<uint, uint> stubInfo;

  int sector = seed->scNum();
  int previousSector = sector - 1;
  int nextSector = sector + 1;
  if (sector == 0) {
    previousSector = 11;
  }
  if (sector == 11) {
    nextSector = 0;
  }

  int wheel = seed->whNum();
  int innerWheel = 0;
  if (wheel == -2)
    innerWheel = -1;
  if (wheel == -1)
    innerWheel = 0;
  if (wheel == 0)
    innerWheel = 1982;
  if (wheel == 1)
    innerWheel = 0;
  if (wheel == 2)
    innerWheel = 1;

  //First align the data
  uint N = 0;
  for (const auto& stub : stubs) {
    N = N + 1;

    if (stub->stNum() != step)
      continue;

    uint distance =
        fabs(wrapAround(((correctedPhi(seed, seed->scNum()) - correctedPhi(stub, seed->scNum())) >> 3), 1024));

    if (stub->scNum() == previousSector) {
      if (stub->whNum() == wheel) {
        if (!stub->tag()) {
          diffInfo[0] = distance;
          stubInfo[0] = N - 1;
        } else {
          diffInfo[1] = distance;
          stubInfo[1] = N - 1;
        }
      } else if (stub->whNum() == innerWheel) {
        if (!stub->tag()) {
          diffInfo[2] = distance;
          stubInfo[2] = N - 1;
        } else {
          diffInfo[3] = distance;
          stubInfo[3] = N - 1;
        }
      }
    } else if (stub->scNum() == sector) {
      if (stub->whNum() == wheel) {
        if (!stub->tag()) {
          diffInfo[4] = distance;
          stubInfo[4] = N - 1;
        } else {
          diffInfo[5] = distance;
          stubInfo[5] = N - 1;
        }
      } else if (stub->whNum() == innerWheel) {
        if (!stub->tag()) {
          diffInfo[6] = distance;
          stubInfo[6] = N - 1;
        } else {
          diffInfo[7] = distance;
          stubInfo[7] = N - 1;
        }
      }
    } else if (stub->scNum() == nextSector) {
      if (stub->whNum() == wheel) {
        if (!stub->tag()) {
          diffInfo[8] = distance;
          stubInfo[8] = N - 1;
        } else {
          diffInfo[9] = distance;
          stubInfo[9] = N - 1;
        }
      } else if (stub->whNum() == innerWheel) {
        if (!stub->tag()) {
          diffInfo[10] = distance;
          stubInfo[10] = N - 1;
        } else {
          diffInfo[11] = distance;
          stubInfo[11] = N - 1;
        }
      }
    }
  }

  uint s1_1 = matchAbs(diffInfo, 0, 1);
  uint s1_2 = matchAbs(diffInfo, 2, 3);
  uint s1_3 = matchAbs(diffInfo, 4, 5);
  uint s1_4 = matchAbs(diffInfo, 6, 7);
  uint s1_5 = matchAbs(diffInfo, 8, 9);
  uint s1_6 = matchAbs(diffInfo, 10, 11);

  uint s2_1 = matchAbs(diffInfo, s1_1, s1_2);
  uint s2_2 = matchAbs(diffInfo, s1_3, s1_4);
  uint s2_3 = matchAbs(diffInfo, s1_5, s1_6);

  uint s3_1 = matchAbs(diffInfo, s2_1, s2_2);

  uint s4 = matchAbs(diffInfo, s3_1, s2_3);

  if (diffInfo[s4] != 60000)
    return std::make_pair(true, stubInfo[s4]);
  else
    return std::make_pair(false, 0);
}

int L1TMuonBarrelKalmanAlgo::correctedPhiB(const L1MuKBMTCombinedStubRef& stub) {
  //Promote phiB to 12 bits
  return 8 * stub->phiB();
}

int L1TMuonBarrelKalmanAlgo::correctedPhi(const L1MuKBMTCombinedStubRef& stub, int sector) {
  if (stub->scNum() == sector) {
    return stub->phi();
  } else if ((stub->scNum() == sector - 1) || (stub->scNum() == 11 && sector == 0)) {
    return stub->phi() - 2144;
  } else if ((stub->scNum() == sector + 1) || (stub->scNum() == 0 && sector == 11)) {
    return stub->phi() + 2144;
  }
  return stub->phi();
}

int L1TMuonBarrelKalmanAlgo::hitPattern(const L1MuKBMTrack& track) {
  unsigned int mask = 0;
  for (const auto& stub : track.stubs()) {
    mask = mask + round(pow(2, stub->stNum() - 1));
  }
  return mask;
}

int L1TMuonBarrelKalmanAlgo::customBitmask(unsigned int bit1, unsigned int bit2, unsigned int bit3, unsigned int bit4) {
  return bit1 * 1 + bit2 * 2 + bit3 * 4 + bit4 * 8;
}

bool L1TMuonBarrelKalmanAlgo::getBit(int bitmask, int pos) { return (bitmask & (1 << pos)) >> pos; }

void L1TMuonBarrelKalmanAlgo::propagate(L1MuKBMTrack& track) {
  int K = track.curvature();
  int phi = track.positionAngle();
  int phiB = track.bendingAngle();
  unsigned int step = track.step();

  //energy loss term only for MU->VERTEX
  //int offset=int(charge*eLoss_[step-1]*K*K);
  //  if (fabs(offset)>4096)
  //      offset=4096*offset/fabs(offset);
  int charge = 1;
  if (K != 0)
    charge = K / fabs(K);

  int KBound = K;

  if (KBound > 4095)
    KBound = 4095;
  if (KBound < -4095)
    KBound = -4095;

  int deltaK = 0;
  int KNew = 0;
  if (step == 1) {
    int addr = KBound / 2;
    if (addr < 0)
      addr = (-KBound) / 2;
    deltaK = 2 * addr - int(2 * addr / (1 + eLoss_[step - 1] * addr));

    if (verbose_)
      printf("propagate to vertex K=%d deltaK=%d addr=%d\n", K, deltaK, addr);
  }

  if (K >= 0)
    KNew = K - deltaK;
  else
    KNew = K + deltaK;

  //phi propagation
  ap_fixed<BITSCURV, BITSCURV> phi11 = ap_fixed<BITSPARAM + 1, 2>(aPhi_[step - 1]) * ap_fixed<BITSCURV, BITSCURV>(K);
  ap_fixed<BITSPHIB, BITSPHIB> phi12 =
      ap_fixed<BITSPARAM + 1, 2>(-bPhi_[step - 1]) * ap_fixed<BITSPHIB, BITSPHIB>(phiB);

  if (verbose_) {
    printf("phi prop = %d * %f = %d, %d * %f = %d\n",
           K,
           ap_fixed<BITSPARAM + 1, 2>(aPhi_[step - 1]).to_float(),
           phi11.to_int(),
           phiB,
           ap_fixed<BITSPARAM + 1, 2>(-bPhi_[step - 1]).to_float(),
           phi12.to_int());
  }
  int phiNew = ap_fixed<BITSPHI, BITSPHI>(phi + phi11 + phi12);

  //phiB propagation
  ap_fixed<BITSCURV, BITSCURV> phiB11 = ap_fixed<BITSPARAM, 1>(aPhiB_[step - 1]) * ap_fixed<BITSCURV, BITSCURV>(K);
  ap_fixed<BITSPHIB + 1, BITSPHIB + 1> phiB12 =
      ap_ufixed<BITSPARAM + 1, 1>(bPhiB_[step - 1]) * ap_fixed<BITSPHIB, BITSPHIB>(phiB);
  int phiBNew = ap_fixed<13, 13>(phiB11 + phiB12);
  if (verbose_) {
    printf("phiB prop = %d * %f = %d, %d * %f = %d\n",
           K,
           ap_fixed<BITSPARAM + 1, 2>(aPhiB_[step - 1]).to_float(),
           phiB11.to_int(),
           phiB,
           ap_ufixed<BITSPARAM + 1, 1>(bPhiB_[step - 1]).to_float(),
           phiB12.to_int());
  }

  //Only for the propagation to vertex we use the LUT for better precision and the full function
  if (step == 1) {
    int addr = KBound / 2;
    // Extra steps to mimic firmware for vertex prop
    ap_ufixed<11, 11> dxyOffset = (int)fabs(aPhiB_[step - 1] * addr / (1 + charge * aPhiBNLO_[step - 1] * addr));
    ap_fixed<12, 12> DXY;
    if (addr > 0)
      DXY = -dxyOffset;
    else
      DXY = dxyOffset;
    phiBNew = ap_fixed<BITSPHIB, BITSPHIB>(DXY - ap_fixed<BITSPHIB, BITSPHIB>(phiB));
    if (verbose_) {
      printf("Vertex phiB prop = %d - %d = %d\n", DXY.to_int(), ap_fixed<BITSPHIB, BITSPHIB>(phiB).to_int(), phiBNew);
    }
  }
  ///////////////////////////////////////////////////////
  //Rest of the stuff  is for the offline version only
  //where we want to check what is happening in the covariaznce matrix

  //Create the transformation matrix
  double a[9];
  a[0] = 1.;
  a[1] = 0.0;
  a[2] = 0.0;
  a[3] = aPhi_[step - 1];
  //  a[3] = 0.0;
  a[4] = 1.0;
  a[5] = -bPhi_[step - 1];
  //a[6]=0.0;
  a[6] = aPhiB_[step - 1];
  if (step == 1)
    a[6] = aPhiB_[step - 1] / 2.0;

  a[7] = 0.0;
  a[8] = bPhiB_[step - 1];

  ROOT::Math::SMatrix<double, 3> P(a, 9);

  const std::vector<double>& covLine = track.covariance();
  L1MuKBMTrack::CovarianceMatrix cov(covLine.begin(), covLine.end());
  cov = ROOT::Math::Similarity(P, cov);

  //Add the multiple scattering
  double phiRMS = mScatteringPhi_[step - 1] * K * K;
  double phiBRMS = mScatteringPhiB_[step - 1] * K * K;

  std::vector<double> b(6);
  b[0] = 0;
  b[1] = 0;
  b[2] = phiRMS;
  b[3] = 0;
  b[4] = 0;
  b[5] = phiBRMS;

  reco::Candidate::CovarianceMatrix MS(b.begin(), b.end());

  cov = cov + MS;

  if (verbose_) {
    printf("Covariance term for phiB = %f\n", cov(2, 2));
    printf("Multiple scattering term for phiB = %f\n", MS(2, 2));
  }

  track.setCovariance(cov);
  track.setCoordinates(step - 1, KNew, phiNew, phiBNew);
}

bool L1TMuonBarrelKalmanAlgo::update(L1MuKBMTrack& track, const L1MuKBMTCombinedStubRef& stub, int mask, int seedQual) {
  updateEta(track, stub);
  if (useOfflineAlgo_) {
    if (mask == 3 || mask == 5 || mask == 9 || mask == 6 || mask == 10 || mask == 12)
      return updateOffline(track, stub);
    else
      return updateOffline1D(track, stub);

  } else
    return updateLUT(track, stub, mask, seedQual);
}

bool L1TMuonBarrelKalmanAlgo::updateOffline(L1MuKBMTrack& track, const L1MuKBMTCombinedStubRef& stub) {
  int trackK = track.curvature();
  int trackPhi = track.positionAngle();
  int trackPhiB = track.bendingAngle();

  int phi = correctedPhi(stub, track.sector());
  int phiB = correctedPhiB(stub);

  Vector2 residual;
  residual[0] = phi - trackPhi;
  residual[1] = phiB - trackPhiB;

  Matrix23 H;
  H(0, 0) = 0.0;
  H(0, 1) = 1.0;
  H(0, 2) = 0.0;
  H(1, 0) = 0.0;
  H(1, 1) = 0.0;
  H(1, 2) = 1.0;

  CovarianceMatrix2 R;
  R(0, 0) = pointResolutionPhi_;
  R(0, 1) = 0.0;
  R(1, 0) = 0.0;
  if (stub->quality() < 4)
    R(1, 1) = pointResolutionPhiBL_[track.step() - 1];
  else
    R(1, 1) = pointResolutionPhiBH_[track.step() - 1];

  const std::vector<double>& covLine = track.covariance();
  L1MuKBMTrack::CovarianceMatrix cov(covLine.begin(), covLine.end());

  CovarianceMatrix2 S = ROOT::Math::Similarity(H, cov) + R;
  if (!S.Invert())
    return false;
  Matrix32 Gain = cov * ROOT::Math::Transpose(H) * S;

  track.setKalmanGain(
      track.step(), fabs(trackK), Gain(0, 0), Gain(0, 1), Gain(1, 0), Gain(1, 1), Gain(2, 0), Gain(2, 1));

  int KNew = (trackK + int(Gain(0, 0) * residual(0) + Gain(0, 1) * residual(1)));
  if (fabs(KNew) > 8192)
    return false;

  int phiNew = wrapAround(trackPhi + residual(0), 8192);
  int phiBNew = wrapAround(trackPhiB + int(Gain(2, 0) * residual(0) + Gain(2, 1) * residual(1)), 4096);

  track.setResidual(stub->stNum() - 1, fabs(phi - phiNew) + fabs(phiB - phiBNew) / 8);

  if (verbose_) {
    printf("residual %d - %d = %d %d - %d = %d\n", phi, trackPhi, int(residual[0]), phiB, trackPhiB, int(residual[1]));
    printf("Gains offline: %f %f %f %f\n", Gain(0, 0), Gain(0, 1), Gain(2, 0), Gain(2, 1));
    printf(" K = %d + %f * %f + %f * %f\n", trackK, Gain(0, 0), residual(0), Gain(0, 1), residual(1));
    printf(" phiB = %d + %f * %f + %f * %f\n", trackPhiB, Gain(2, 0), residual(0), Gain(2, 1), residual(1));
  }

  track.setCoordinates(track.step(), KNew, phiNew, phiBNew);
  Matrix33 covNew = cov - Gain * (H * cov);
  L1MuKBMTrack::CovarianceMatrix c;

  c(0, 0) = covNew(0, 0);
  c(0, 1) = covNew(0, 1);
  c(0, 2) = covNew(0, 2);
  c(1, 0) = covNew(1, 0);
  c(1, 1) = covNew(1, 1);
  c(1, 2) = covNew(1, 2);
  c(2, 0) = covNew(2, 0);
  c(2, 1) = covNew(2, 1);
  c(2, 2) = covNew(2, 2);
  if (verbose_) {
    printf("Post Fit Covariance Matrix %f %f %f\n", cov(0, 0), cov(1, 1), cov(2, 2));
  }

  track.setCovariance(c);
  track.addStub(stub);
  track.setHitPattern(hitPattern(track));

  return true;
}

bool L1TMuonBarrelKalmanAlgo::updateOffline1D(L1MuKBMTrack& track, const L1MuKBMTCombinedStubRef& stub) {
  int trackK = track.curvature();
  int trackPhi = track.positionAngle();
  int trackPhiB = track.bendingAngle();

  int phi = correctedPhi(stub, track.sector());

  double residual = phi - trackPhi;

  if (verbose_)
    printf("residuals %d - %d = %d\n", phi, trackPhi, int(residual));

  Matrix13 H;
  H(0, 0) = 0.0;
  H(0, 1) = 1.0;
  H(0, 2) = 0.0;

  const std::vector<double>& covLine = track.covariance();
  L1MuKBMTrack::CovarianceMatrix cov(covLine.begin(), covLine.end());

  double S = ROOT::Math::Similarity(H, cov)(0, 0) + pointResolutionPhi_;

  if (S == 0.0)
    return false;
  Matrix31 Gain = cov * ROOT::Math::Transpose(H) / S;

  track.setKalmanGain(track.step(), fabs(trackK), Gain(0, 0), 0.0, Gain(1, 0), 0.0, Gain(2, 0), 0.0);
  if (verbose_)
    printf("Gains: %f %f\n", Gain(0, 0), Gain(2, 0));

  int KNew = wrapAround(trackK + int(Gain(0, 0) * residual), 8192);
  int phiNew = wrapAround(trackPhi + residual, 8192);
  int phiBNew = wrapAround(trackPhiB + int(Gain(2, 0) * residual), 4096);
  track.setCoordinates(track.step(), KNew, phiNew, phiBNew);
  Matrix33 covNew = cov - Gain * (H * cov);
  L1MuKBMTrack::CovarianceMatrix c;

  if (verbose_) {
    printf("phiUpdate: %d %d\n", int(Gain(0, 0) * residual), int(Gain(2, 0) * residual));
  }

  c(0, 0) = covNew(0, 0);
  c(0, 1) = covNew(0, 1);
  c(0, 2) = covNew(0, 2);
  c(1, 0) = covNew(1, 0);
  c(1, 1) = covNew(1, 1);
  c(1, 2) = covNew(1, 2);
  c(2, 0) = covNew(2, 0);
  c(2, 1) = covNew(2, 1);
  c(2, 2) = covNew(2, 2);
  track.setCovariance(c);
  track.addStub(stub);
  track.setHitPattern(hitPattern(track));

  return true;
}

bool L1TMuonBarrelKalmanAlgo::updateLUT(L1MuKBMTrack& track,
                                        const L1MuKBMTCombinedStubRef& stub,
                                        int mask,
                                        int seedQual) {
  int trackK = track.curvature();
  int trackPhi = track.positionAngle();
  int trackPhiB = track.bendingAngle();

  int phi = correctedPhi(stub, track.sector());
  int phiB = correctedPhiB(stub);

  Vector2 residual;
  ap_fixed<BITSPHI + 1, BITSPHI + 1> residualPhi = phi - trackPhi;
  ap_fixed<BITSPHIB + 1, BITSPHIB + 1> residualPhiB = phiB - trackPhiB;

  if (verbose_)
    printf("residual %d - %d = %d %d - %d = %d\n",
           phi,
           trackPhi,
           residualPhi.to_int(),
           phiB,
           trackPhiB,
           residualPhiB.to_int());

  uint absK = fabs(trackK);
  if (absK > 4095)
    absK = 4095;

  std::vector<float> GAIN;
  //For the three stub stuff use only gains 0 and 4
  if (!(mask == 3 || mask == 5 || mask == 9 || mask == 6 || mask == 10 || mask == 12)) {
    GAIN = lutService_->trackGain(track.step(), track.hitPattern(), absK / 4);
    GAIN[1] = 0.0;
    GAIN[3] = 0.0;

  } else {
    GAIN = lutService_->trackGain2(track.step(), track.hitPattern(), absK / 8, seedQual, stub->quality());
  }
  if (verbose_) {
    printf("Gains (fp): %f %f %f %f\n", GAIN[0], GAIN[1], GAIN[2], GAIN[3]);
    if (!(mask == 3 || mask == 5 || mask == 9 || mask == 6 || mask == 10 || mask == 12))
      printf("Addr=%d   gain0=%f gain4=-%f\n",
             absK / 4,
             ap_ufixed<GAIN_0, GAIN_0INT>(GAIN[0]).to_float(),
             ap_ufixed<GAIN_4, GAIN_4INT>(GAIN[2]).to_float());
    else
      printf("Addr=%d   %f -%f %f %f\n",
             absK / 4,
             ap_fixed<GAIN2_0, GAIN2_0INT>(GAIN[0]).to_float(),
             ap_ufixed<GAIN2_1, GAIN2_1INT>(GAIN[1]).to_float(),
             ap_ufixed<GAIN2_4, GAIN2_4INT>(GAIN[2]).to_float(),
             ap_ufixed<GAIN2_5, GAIN2_5INT>(GAIN[3]).to_float());
  }

  track.setKalmanGain(track.step(), fabs(trackK), GAIN[0], GAIN[1], 1, 0, GAIN[2], GAIN[3]);

  int KNew;
  if (!(mask == 3 || mask == 5 || mask == 9 || mask == 6 || mask == 10 || mask == 12)) {
    KNew = ap_fixed<BITSPHI + 9, BITSPHI + 9>(ap_fixed<BITSCURV, BITSCURV>(trackK) +
                                              ap_ufixed<GAIN_0, GAIN_0INT>(GAIN[0]) * residualPhi);
  } else {
    ap_fixed<BITSPHI + 7, BITSPHI + 7> k11 = ap_fixed<GAIN2_0, GAIN2_0INT>(GAIN[0]) * residualPhi;
    ap_fixed<BITSPHIB + 4, BITSPHIB + 4> k12 = ap_ufixed<GAIN2_1, GAIN2_1INT>(GAIN[1]) * residualPhiB;
    KNew = ap_fixed<BITSPHI + 9, BITSPHI + 9>(ap_fixed<BITSCURV, BITSCURV>(trackK) + k11 - k12);
  }
  if (fabs(KNew) >= 8191)
    return false;
  KNew = wrapAround(KNew, 8192);
  int phiNew = phi;

  //different products for different firmware logic
  ap_fixed<BITSPHI + 5, BITSPHI + 5> pbdouble_0 = ap_ufixed<GAIN2_4, GAIN2_4INT>(GAIN[2]) * residualPhi;
  ap_fixed<BITSPHIB + 24, BITSPHIB + 4> pb_1 = ap_ufixed<GAIN2_5, GAIN2_5INT>(GAIN[3]) * residualPhiB;
  ap_fixed<BITSPHI + 9, BITSPHI + 5> pb_0 = ap_ufixed<GAIN_4, GAIN_4INT>(GAIN[2]) * residualPhi;

  if (verbose_) {
    printf("phiupdate %f %f %f\n", pb_0.to_float(), pb_1.to_float(), pbdouble_0.to_float());
  }

  int phiBNew;
  if (!(mask == 3 || mask == 5 || mask == 9 || mask == 6 || mask == 10 || mask == 12)) {
    phiBNew = ap_fixed<BITSPHI + 8, BITSPHI + 8>(ap_fixed<BITSPHIB, BITSPHIB>(trackPhiB) -
                                                 ap_ufixed<GAIN_4, GAIN_4INT>(GAIN[2]) * residualPhi);

    if (fabs(phiBNew) >= 4095)
      return false;
  } else {
    phiBNew = ap_fixed<BITSPHI + 7, BITSPHI + 7>(ap_fixed<BITSPHIB, BITSPHIB>(trackPhiB) + pb_1 - pbdouble_0);
    if (fabs(phiBNew) >= 4095)
      return false;
  }
  track.setCoordinates(track.step(), KNew, phiNew, phiBNew);
  track.addStub(stub);
  track.setHitPattern(hitPattern(track));
  return true;
}

void L1TMuonBarrelKalmanAlgo::updateEta(L1MuKBMTrack& track, const L1MuKBMTCombinedStubRef& stub) {}

void L1TMuonBarrelKalmanAlgo::vertexConstraint(L1MuKBMTrack& track) {
  if (useOfflineAlgo_)
    vertexConstraintOffline(track);
  else
    vertexConstraintLUT(track);
}

void L1TMuonBarrelKalmanAlgo::vertexConstraintOffline(L1MuKBMTrack& track) {
  double residual = -track.dxy();
  Matrix13 H;
  H(0, 0) = 0;
  H(0, 1) = 0;
  H(0, 2) = 1;

  const std::vector<double>& covLine = track.covariance();
  L1MuKBMTrack::CovarianceMatrix cov(covLine.begin(), covLine.end());

  double S = (ROOT::Math::Similarity(H, cov))(0, 0) + pointResolutionVertex_;
  S = 1.0 / S;
  Matrix31 Gain = cov * (ROOT::Math::Transpose(H)) * S;
  track.setKalmanGain(track.step(), fabs(track.curvature()), Gain(0, 0), Gain(1, 0), Gain(2, 0));

  if (verbose_) {
    printf("sigma3=%f sigma6=%f\n", cov(0, 3), cov(3, 3));
    printf(" K = %d + %f * %f\n", track.curvature(), Gain(0, 0), residual);
  }

  int KNew = wrapAround(int(track.curvature() + Gain(0, 0) * residual), 8192);
  int phiNew = wrapAround(int(track.positionAngle() + Gain(1, 0) * residual), 8192);
  int dxyNew = wrapAround(int(track.dxy() + Gain(2, 0) * residual), 8192);
  if (verbose_)
    printf("Post fit impact parameter=%d\n", dxyNew);
  track.setCoordinatesAtVertex(KNew, phiNew, -residual);
  Matrix33 covNew = cov - Gain * (H * cov);
  L1MuKBMTrack::CovarianceMatrix c;
  c(0, 0) = covNew(0, 0);
  c(0, 1) = covNew(0, 1);
  c(0, 2) = covNew(0, 2);
  c(1, 0) = covNew(1, 0);
  c(1, 1) = covNew(1, 1);
  c(1, 2) = covNew(1, 2);
  c(2, 0) = covNew(2, 0);
  c(2, 1) = covNew(2, 1);
  c(2, 2) = covNew(2, 2);
  track.setCovariance(c);
  //  track.covariance = track.covariance - Gain*H*track.covariance;
}

void L1TMuonBarrelKalmanAlgo::vertexConstraintLUT(L1MuKBMTrack& track) {
  double residual = -track.dxy();
  uint absK = fabs(track.curvature());
  if (absK > 2047)
    absK = 2047;

  std::pair<float, float> GAIN = lutService_->vertexGain(track.hitPattern(), absK / 2);
  track.setKalmanGain(track.step(), fabs(track.curvature()), GAIN.first, GAIN.second, -1);

  ap_fixed<BITSCURV, BITSCURV> k_0 =
      -(ap_ufixed<GAIN_V0, GAIN_V0INT>(fabs(GAIN.first))) * ap_fixed<BITSPHIB, BITSPHIB>(residual);
  int KNew = ap_fixed<BITSCURV, BITSCURV>(k_0 + ap_fixed<BITSCURV, BITSCURV>(track.curvature()));

  if (verbose_) {
    printf("VERTEX GAIN(%d)= -%f * %d = %d\n",
           absK / 2,
           ap_ufixed<GAIN_V0, GAIN_V0INT>(fabs(GAIN.first)).to_float(),
           ap_fixed<BITSPHIB, BITSPHIB>(residual).to_int(),
           k_0.to_int());
  }

  int p_0 = fp_product(GAIN.second, int(residual), 7);
  int phiNew = wrapAround(track.positionAngle() + p_0, 8192);
  track.setCoordinatesAtVertex(KNew, phiNew, -residual);
}

void L1TMuonBarrelKalmanAlgo::setFloatingPointValues(L1MuKBMTrack& track, bool vertex) {
  int K, etaINT;

  if (track.hasFineEta())
    etaINT = track.fineEta();
  else
    etaINT = track.coarseEta();

  double lsb = 1.25 / float(1 << 13);
  double lsbEta = 0.010875;

  if (vertex) {
    int charge = 1;
    if (track.curvatureAtVertex() < 0)
      charge = -1;
    double pt = double(ptLUT(track.curvatureAtVertex())) / 2.0;

    double phi = track.sector() * M_PI / 6.0 + track.phiAtVertex() * M_PI / (6 * 2048.) - 2 * M_PI;

    double eta = etaINT * lsbEta;
    track.setPtEtaPhi(pt, eta, phi);
    track.setCharge(charge);
  } else {
    K = track.curvatureAtMuon();
    if (K == 0)
      K = 1;

    if (fabs(K) < 46)
      K = 46 * K / fabs(K);
    double pt = 1.0 / (lsb * fabs(K));
    if (pt < 1.6)
      pt = 1.6;
    track.setPtUnconstrained(pt);
  }
}

std::pair<bool, L1MuKBMTrack> L1TMuonBarrelKalmanAlgo::chain(const L1MuKBMTCombinedStubRef& seed,
                                                             const L1MuKBMTCombinedStubRefVector& stubs) {
  L1MuKBMTrackCollection pretracks;
  std::vector<int> combinatorics;
  int seedQual;
  switch (seed->stNum()) {
    case 1:
      combinatorics = combos1_;
      break;
    case 2:
      combinatorics = combos2_;
      break;

    case 3:
      combinatorics = combos3_;
      break;

    case 4:
      combinatorics = combos4_;
      break;

    default:
      printf("Something really bad happend\n");
  }

  L1MuKBMTrack nullTrack(seed, correctedPhi(seed, seed->scNum()), correctedPhiB(seed));
  seedQual = seed->quality();
  for (const auto& mask : combinatorics) {
    L1MuKBMTrack track(seed, correctedPhi(seed, seed->scNum()), correctedPhiB(seed));
    int phiB = correctedPhiB(seed);
    int charge;
    if (phiB == 0)
      charge = 0;
    else
      charge = phiB / fabs(phiB);

    int address = phiB;
    if (track.step() == 4 && (fabs(seed->phiB()) > 15))
      address = charge * 15 * 8;

    if (track.step() == 3 && (fabs(seed->phiB()) > 30))
      address = charge * 30 * 8;
    if (track.step() == 2 && (fabs(seed->phiB()) > 127))
      address = charge * 127 * 8;
    int initialK = int(initK_[seed->stNum() - 1] * address / (1 + initK2_[seed->stNum() - 1] * charge * address));
    if (initialK > 8191)
      initialK = 8191;
    if (initialK < -8191)
      initialK = -8191;

    track.setCoordinates(seed->stNum(), initialK, correctedPhi(seed, seed->scNum()), phiB);
    if (seed->quality() < 4) {
      track.setCoordinates(seed->stNum(), 0, correctedPhi(seed, seed->scNum()), 0);
    }

    track.setHitPattern(hitPattern(track));
    //set covariance
    L1MuKBMTrack::CovarianceMatrix covariance;

    float DK = 512 * 512.;
    covariance(0, 0) = DK;
    covariance(0, 1) = 0;
    covariance(0, 2) = 0;
    covariance(1, 0) = 0;
    covariance(1, 1) = float(pointResolutionPhi_);
    covariance(1, 2) = 0;
    covariance(2, 0) = 0;
    covariance(2, 1) = 0;
    if (!(mask == 3 || mask == 5 || mask == 9 || mask == 6 || mask == 10 || mask == 12))
      covariance(2, 2) = float(pointResolutionPhiB_);
    else {
      if (seed->quality() < 4)
        covariance(2, 2) = float(pointResolutionPhiBL_[seed->stNum() - 1]);
      else
        covariance(2, 2) = float(pointResolutionPhiBH_[seed->stNum() - 1]);
    }
    track.setCovariance(covariance);

    //
    if (verbose_) {
      printf("New Kalman fit staring at step=%d, phi=%d,phiB=%d with curvature=%d\n",
             track.step(),
             track.positionAngle(),
             track.bendingAngle(),
             track.curvature());
      printf("BITMASK:");
      for (unsigned int i = 0; i < 4; ++i)
        printf("%d", getBit(mask, i));
      printf("\n");
      printf("------------------------------------------------------\n");
      printf("------------------------------------------------------\n");
      printf("------------------------------------------------------\n");
      printf("stubs:\n");
      for (const auto& stub : stubs)
        printf("station=%d phi=%d phiB=%d qual=%d tag=%d sector=%d wheel=%d fineEta= %d %d\n",
               stub->stNum(),
               correctedPhi(stub, seed->scNum()),
               correctedPhiB(stub),
               stub->quality(),
               stub->tag(),
               stub->scNum(),
               stub->whNum(),
               stub->eta1(),
               stub->eta2());
      printf("------------------------------------------------------\n");
      printf("------------------------------------------------------\n");
    }

    int phiAtStation2 = 0;

    while (track.step() > 0) {
      // muon station 1
      if (track.step() == 1) {
        track.setCoordinatesAtMuon(track.curvature(), track.positionAngle(), track.bendingAngle());
        phiAtStation2 = phiAt2(track);
        bool passed = estimateChiSquare(track);
        if (!passed)
          break;
        calculateEta(track);
        setFloatingPointValues(track, false);
        //calculate coarse eta
        //////////////////////

        if (verbose_)
          printf("Unconstrained PT  in Muon System: pt=%f\n", track.ptUnconstrained());
      }

      propagate(track);
      if (verbose_)
        printf("propagated Coordinates step:%d,phi=%d,phiB=%d,K=%d\n",
               track.step(),
               track.positionAngle(),
               track.bendingAngle(),
               track.curvature());

      if (track.step() > 0)
        if (getBit(mask, track.step() - 1)) {
          std::pair<bool, uint> bestStub = match(seed, stubs, track.step());
          if ((!bestStub.first) || (!update(track, stubs[bestStub.second], mask, seedQual)))
            break;
          if (verbose_) {
            printf("updated Coordinates step:%d,phi=%d,phiB=%d,K=%d\n",
                   track.step(),
                   track.positionAngle(),
                   track.bendingAngle(),
                   track.curvature());
          }
        }

      if (track.step() == 0) {
        track.setCoordinatesAtVertex(track.curvature(), track.positionAngle(), track.bendingAngle());
        if (verbose_)
          printf(" Coordinates before vertex constraint step:%d,phi=%d,dxy=%d,K=%d\n",
                 track.step(),
                 track.phiAtVertex(),
                 track.dxy(),
                 track.curvatureAtVertex());
        if (verbose_)
          printf("Chi Square = %d\n", track.approxChi2());

        vertexConstraint(track);
        estimateCompatibility(track);
        if (verbose_) {
          printf(" Coordinates after vertex constraint step:%d,phi=%d,dxy=%d,K=%d  maximum local chi2=%d\n",
                 track.step(),
                 track.phiAtVertex(),
                 track.dxy(),
                 track.curvatureAtVertex(),
                 track.approxChi2());
          printf("------------------------------------------------------\n");
          printf("------------------------------------------------------\n");
        }
        setFloatingPointValues(track, true);
        //set the coordinates at muon to include phi at station 2
        track.setCoordinatesAtMuon(track.curvatureAtMuon(), phiAtStation2, track.phiBAtMuon());
        track.setRank(rank(track));
        if (verbose_)
          printf("Floating point coordinates at vertex: pt=%f, eta=%f phi=%f\n", track.pt(), track.eta(), track.phi());
        pretracks.push_back(track);
      }
    }
  }

  //Now for all the pretracks we need only one
  L1MuKBMTrackCollection cleaned = clean(pretracks, seed->stNum());

  if (!cleaned.empty()) {
    return std::make_pair(true, cleaned[0]);
  }
  return std::make_pair(false, nullTrack);
}

bool L1TMuonBarrelKalmanAlgo::estimateChiSquare(L1MuKBMTrack& track) {
  //here we have a simplification of the algorithm for the sake of the emulator - rsult is identical
  // we apply cuts on the firmware as |u -u'|^2 < a+b *K^2
  int K = track.curvatureAtMuon();

  uint chi = 0;

  // const double PHI[4]={0.0,0.249,0.543,0.786};
  // const double DROR[4]={0.0,0.182,0.430,0.677};

  int coords = wrapAround((track.phiAtMuon() + track.phiBAtMuon()) >> 4, 512);
  for (const auto& stub : track.stubs()) {
    int AK = wrapAround(fp_product(-chiSquare_[stub->stNum() - 1], K >> 4, 8), 256);
    int stubCoords = wrapAround((correctedPhi(stub, track.sector()) >> 4) + (stub->phiB() >> 1), 512);
    int diff1 = wrapAround(stubCoords - coords, 1024);
    uint delta = wrapAround(abs(diff1 + AK), 2048);
    chi = chi + delta;
    if (verbose_)
      printf("Chi Square stub for track with pattern=%d coords=%d -> AK=%d stubCoords=%d diff=%d delta=%d\n",
             track.hitPattern(),
             coords,
             AK,
             stubCoords,
             diff1,
             delta);
  }

  if (chi > 127)
    chi = 127;
  // for (const auto& stub: track.stubs()) {
  //   int deltaPhi = (correctedPhi(stub,track.sector())-track.phiAtMuon())>>3;
  //   int AK =  fp_product(PHI[stub->stNum()-1],K>>3,8);
  //   int BPB = fp_product(DROR[stub->stNum()-1],track.phiBAtMuon()>>3,8);
  //   chi=chi+abs(deltaPhi-AK-BPB);
  // }
  // //  }

  track.setApproxChi2(chi);
  for (uint i = 0; i < chiSquareCutPattern_.size(); ++i) {
    if (track.hitPattern() == chiSquareCutPattern_[i] && fabs(K) < chiSquareCutCurv_[i] &&
        track.approxChi2() > chiSquareCut_[i])
      return false;
  }
  return true;
}

void L1TMuonBarrelKalmanAlgo::estimateCompatibility(L1MuKBMTrack& track) {
  int K = track.curvatureAtVertex() >> 4;

  if (track.stubs().size() != 2) {
    track.setTrackCompatibility(0);
    return;
  }

  uint stubSel = 1;
  if (track.stubs()[0]->quality() > track.stubs()[1]->quality())
    stubSel = 0;
  const L1MuKBMTCombinedStubRef& stub = track.stubs()[stubSel];

  if (verbose_) {
    printf("stubsel %d phi=%d phiB=%d\n", stubSel, stub->phi(), stub->phiB());
  }

  ap_ufixed<BITSCURV - 5, BITSCURV - 5> absK;
  if (K < 0)
    absK = -K;
  else
    absK = K;

  ap_fixed<12, 12> diff = ap_int<10>(stub->phiB()) -
                          ap_ufixed<5, 1>(trackComp_[stub->stNum() - 1]) * ap_fixed<BITSCURV - 4, BITSCURV - 4>(K);
  ap_ufixed<11, 11> delta;
  if (diff.is_neg())
    delta = -diff;
  else
    delta = diff;

  ap_ufixed<BITSCURV - 5, BITSCURV - 5> err =
      ap_uint<3>(trackCompErr1_[stub->stNum() - 1]) + ap_ufixed<5, 0>(trackCompErr2_[stub->stNum() - 1]) * absK;
  track.setTrackCompatibility(((int)delta) / ((int)err));
  for (uint i = 0; i < trackCompPattern_.size(); ++i) {
    int deltaMax = ap_ufixed<BITSCURV, BITSCURV>(err * trackCompCut_[i]);
    if (verbose_) {
      if (track.hitPattern() == trackCompPattern_[i]) {
        printf("delta = %d = abs(%d - %f*%d\n", delta.to_int(), stub->phiB(), trackComp_[stub->stNum() - 1], K);
        printf("err = %d = %f + %f*%d\n",
               err.to_int(),
               trackCompErr1_[stub->stNum() - 1],
               trackCompErr2_[stub->stNum() - 1],
               absK.to_int());
        printf("deltaMax = %d = %d*%d\n", deltaMax, err.to_int(), trackCompCut_[i]);
      }
    }
    if ((track.hitPattern() == trackCompPattern_[i]) && ((int)absK < trackCompCutCurv_[i]) &&
        ((track.approxChi2() > chiSquareCutTight_[i]) || (delta > deltaMax))) {
      track.setCoordinatesAtVertex(8191, track.phiAtVertex(), track.dxy());
      break;
    }
  }
}

int L1TMuonBarrelKalmanAlgo::rank(const L1MuKBMTrack& track) {
  //    int offset=0;
  uint chi = track.approxChi2() > 127 ? 127 : track.approxChi2();
  if (hitPattern(track) == customBitmask(0, 0, 1, 1)) {
    return 60;
  }
  //    return offset+(track.stubs().size()*2+track.quality())*80-track.approxChi2();
  return 160 + (track.stubs().size()) * 20 - chi;
}

int L1TMuonBarrelKalmanAlgo::wrapAround(int value, int maximum) {
  if (value > maximum - 1)
    return wrapAround(value - 2 * maximum, maximum);
  if (value < -maximum)
    return wrapAround(value + 2 * maximum, maximum);
  return value;
}

int L1TMuonBarrelKalmanAlgo::encode(bool ownwheel, int sector, bool tag) {
  if (ownwheel) {
    if (sector == 0) {
      if (tag)
        return 9;
      else
        return 8;
    } else if (sector == 1) {
      if (tag)
        return 11;
      else
        return 10;

    } else {
      if (tag)
        return 13;
      else
        return 12;
    }

  } else {
    if (sector == 0) {
      if (tag)
        return 1;
      else
        return 0;
    } else if (sector == 1) {
      if (tag)
        return 3;
      else
        return 2;

    } else {
      if (tag)
        return 5;
      else
        return 4;
    }
  }
  return 15;
}

std::map<int, int> L1TMuonBarrelKalmanAlgo::trackAddress(const L1MuKBMTrack& track, int& word) {
  std::map<int, int> out;

  out[l1t::RegionalMuonCand::kWheelSide] = track.wheel() < 0;
  if (track.wheel() == -2)
    out[l1t::RegionalMuonCand::kWheelNum] = 2;
  else if (track.wheel() == -1)
    out[l1t::RegionalMuonCand::kWheelNum] = 1;
  else if (track.wheel() == 0)
    out[l1t::RegionalMuonCand::kWheelNum] = 0;
  else if (track.wheel() == 1)
    out[l1t::RegionalMuonCand::kWheelNum] = 1;
  else if (track.wheel() == 2)
    out[l1t::RegionalMuonCand::kWheelNum] = 2;
  else
    out[l1t::RegionalMuonCand::kWheelNum] = 0;
  out[l1t::RegionalMuonCand::kStat1] = 15;
  out[l1t::RegionalMuonCand::kStat2] = 15;
  out[l1t::RegionalMuonCand::kStat3] = 15;
  out[l1t::RegionalMuonCand::kStat4] = 3;
  out[l1t::RegionalMuonCand::kSegSelStat1] = 0;
  out[l1t::RegionalMuonCand::kSegSelStat2] = 0;
  out[l1t::RegionalMuonCand::kSegSelStat3] = 0;
  out[l1t::RegionalMuonCand::kSegSelStat4] = 0;
  //out[l1t::RegionalMuonCand::kNumBmtfSubAddr]=0; // This is commented out for better data/MC agreement

  for (const auto& stub : track.stubs()) {
    bool ownwheel = stub->whNum() == track.wheel();
    int sector = 0;
    if ((stub->scNum() == track.sector() + 1) || (stub->scNum() == 0 && track.sector() == 11))
      sector = +1;
    if ((stub->scNum() == track.sector() - 1) || (stub->scNum() == 11 && track.sector() == 0))
      sector = -1;
    int addr = encode(ownwheel, sector, stub->tag());

    if (stub->stNum() == 4) {
      if (stub->tag())
        addr = 1;
      else
        addr = 2;
      out[l1t::RegionalMuonCand::kStat4] = addr;
    }
    if (stub->stNum() == 3) {
      out[l1t::RegionalMuonCand::kStat3] = addr;
    }
    if (stub->stNum() == 2) {
      out[l1t::RegionalMuonCand::kStat2] = addr;
    }
    if (stub->stNum() == 1) {
      out[l1t::RegionalMuonCand::kStat1] = addr;
    }
  }

  word = 0;
  word = word | out[l1t::RegionalMuonCand::kStat4] << 12;
  word = word | out[l1t::RegionalMuonCand::kStat3] << 8;
  word = word | out[l1t::RegionalMuonCand::kStat2] << 4;
  word = word | out[l1t::RegionalMuonCand::kStat1];

  return out;
}

uint L1TMuonBarrelKalmanAlgo::twosCompToBits(int q) {
  if (q >= 0)
    return q;
  else
    return (~q) + 1;
}

int L1TMuonBarrelKalmanAlgo::fp_product(float a, int b, uint bits) {
  //  return long(a*(1<<bits)*b)>>bits;
  return (long((a * (1 << bits)) * b)) >> bits;
}

int L1TMuonBarrelKalmanAlgo::ptLUT(int K) {
  int charge = (K >= 0) ? +1 : -1;
  float lsb = 1.25 / float(1 << 13);
  float FK = fabs(K);

  if (FK > 2047)
    FK = 2047.;
  if (FK < 26)
    FK = 26.;

  FK = FK * lsb;

  //step 1 -material and B-field
  FK = .8569 * FK / (1.0 + 0.1144 * FK);
  //step 2 - misalignment
  FK = FK - charge * 1.23e-03;
  //Get to BMTF scale
  FK = FK / 1.17;

  int pt = 0;
  if (FK != 0)
    pt = int(2.0 / FK);

  if (pt > 511)
    pt = 511;

  if (pt < 8)
    pt = 8;

  return pt;
}

L1MuKBMTrackCollection L1TMuonBarrelKalmanAlgo::clean(const L1MuKBMTrackCollection& tracks, uint seed) {
  L1MuKBMTrackCollection out;

  std::map<uint, int> infoRank;
  std::map<uint, L1MuKBMTrack> infoTrack;
  for (uint i = 3; i <= 15; ++i) {
    if (i == 4 || i == 8)
      continue;
    infoRank[i] = -1;
  }

  for (const auto& track : tracks) {
    infoRank[track.hitPattern()] = rank(track);
    infoTrack[track.hitPattern()] = track;
  }

  int selected = 15;
  if (seed == 4)  //station 4 seeded
  {
    int sel6 = infoRank[10] >= infoRank[12] ? 10 : 12;
    int sel5 = infoRank[14] >= infoRank[9] ? 14 : 9;
    int sel4 = infoRank[11] >= infoRank[13] ? 11 : 13;
    int sel3 = infoRank[sel6] >= infoRank[sel5] ? sel6 : sel5;
    int sel2 = infoRank[sel4] >= infoRank[sel3] ? sel4 : sel3;
    selected = infoRank[15] >= infoRank[sel2] ? 15 : sel2;
  }
  if (seed == 3)  //station 3 seeded
  {
    int sel2 = infoRank[5] >= infoRank[6] ? 5 : 6;
    selected = infoRank[7] >= infoRank[sel2] ? 7 : sel2;
  }
  if (seed == 2)  //station 2 seeded
    selected = 3;

  auto search = infoTrack.find(selected);
  if (search != infoTrack.end())
    out.push_back(search->second);

  return out;
}

uint L1TMuonBarrelKalmanAlgo::etaStubRank(const L1MuKBMTCombinedStubRef& stub) {
  if (stub->qeta1() != 0 && stub->qeta2() != 0) {
    return 0;
  }
  if (stub->qeta1() == 0) {
    return 0;
  }
  //  return (stub->qeta1()*4+stub->stNum());
  return (stub->qeta1());
}

void L1TMuonBarrelKalmanAlgo::calculateEta(L1MuKBMTrack& track) {
  uint pattern = track.hitPattern();
  int wheel = track.stubs()[0]->whNum();
  uint awheel = fabs(wheel);
  int sign = 1;
  if (wheel < 0)
    sign = -1;
  uint nstubs = track.stubs().size();
  uint mask = 0;
  for (unsigned int i = 0; i < track.stubs().size(); ++i) {
    if (fabs(track.stubs()[i]->whNum()) != awheel)
      mask = mask | (1 << i);
  }
  mask = (awheel << nstubs) | mask;
  track.setCoarseEta(sign * lutService_->coarseEta(pattern, mask));
  int sumweights = 0;
  int sums = 0;

  for (const auto& stub : track.stubs()) {
    uint rank = etaStubRank(stub);
    if (rank == 0)
      continue;
    //    printf("Stub station=%d rank=%d values=%d %d\n",stub->stNum(),rank,stub->eta1(),stub->eta2());
    sumweights += rank;
    sums += rank * stub->eta1();
  }
  //0.5  0.332031 0.25 0.199219 0.164063
  float factor;
  if (sumweights == 1)
    factor = 1.0;
  else if (sumweights == 2)
    factor = 0.5;
  else if (sumweights == 3)
    factor = 0.332031;
  else if (sumweights == 4)
    factor = 0.25;
  else if (sumweights == 5)
    factor = 0.199219;
  else if (sumweights == 6)
    factor = 0.164063;
  else
    factor = 0.0;

  int eta = 0;
  if (sums > 0)
    eta = fp_product(factor, sums, 10);
  else
    eta = -fp_product(factor, fabs(sums), 10);

  //int eta=int(factor*sums);
  //    printf("Eta debug %f *%d=%d\n",factor,sums,eta);

  if (sumweights > 0)
    track.setFineEta(eta);
}

int L1TMuonBarrelKalmanAlgo::phiAt2(const L1MuKBMTrack& track) {
  //If there is stub at station 2 use this else propagate from 1
  for (const auto& stub : track.stubs())
    if (stub->stNum() == 2)
      return correctedPhi(stub, track.sector());

  ap_fixed<BITSPHI, BITSPHI> phi = track.phiAtMuon();
  ap_fixed<BITSPHIB, BITSPHIB> phiB = track.phiBAtMuon();
  ap_fixed<BITSPARAM, 1> phiAt2 = phiAt2_;
  int phiNew = ap_fixed<BITSPHI + 1, BITSPHI + 1, AP_TRN_ZERO, AP_SAT>(phi + phiAt2 * phiB);

  if (verbose_)
    printf("Phi at second station=%d\n", phiNew);
  return phiNew;
}
