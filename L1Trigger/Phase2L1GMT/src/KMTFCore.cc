#include "L1Trigger/Phase2L1GMT/interface/KMTFCore.h"
using namespace Phase2L1GMT;
KMTFCore::KMTFCore(const edm::ParameterSet& settings)
    : lutService_(new KMTFLUTs(settings.getParameter<std::string>("lutFile"))),
      verbose_(settings.getParameter<bool>("verbose")),
      initK_(settings.getParameter<std::vector<double> >("initialK")),
      initK2_(settings.getParameter<std::vector<double> >("initialK2")),
      eLoss_(settings.getParameter<std::vector<double> >("eLoss")),
      aPhi_(settings.getParameter<std::vector<double> >("aPhi")),
      aPhiB_(settings.getParameter<std::vector<double> >("aPhiB")),
      aPhiBNLO_(settings.getParameter<std::vector<double> >("aPhiBNLO")),
      bPhi_(settings.getParameter<std::vector<double> >("bPhi")),
      bPhiB_(settings.getParameter<std::vector<double> >("bPhiB")),
      phiAt2_(settings.getParameter<double>("phiAt2")),

      chiSquareDisp1_(settings.getParameter<std::vector<double> >("chiSquareDisp1")),
      chiSquareDisp2_(settings.getParameter<std::vector<double> >("chiSquareDisp2")),
      chiSquareDisp3_(settings.getParameter<std::vector<double> >("chiSquareDisp3")),
      chiSquareErrADisp1_(settings.getParameter<std::vector<int> >("chiSquareErrADisp1")),
      chiSquareErrADisp2_(settings.getParameter<std::vector<int> >("chiSquareErrADisp2")),
      chiSquareErrADisp3_(settings.getParameter<std::vector<int> >("chiSquareErrADisp3")),
      chiSquareErrBDisp1_(settings.getParameter<std::vector<double> >("chiSquareErrBDisp1")),
      chiSquareErrBDisp2_(settings.getParameter<std::vector<double> >("chiSquareErrBDisp2")),
      chiSquareErrBDisp3_(settings.getParameter<std::vector<double> >("chiSquareErrBDisp3")),

      chiSquarePrompt1_(settings.getParameter<std::vector<double> >("chiSquarePrompt1")),
      chiSquarePrompt2_(settings.getParameter<std::vector<double> >("chiSquarePrompt2")),
      chiSquarePrompt3_(settings.getParameter<std::vector<double> >("chiSquarePrompt3")),
      chiSquareErrAPrompt1_(settings.getParameter<std::vector<int> >("chiSquareErrAPrompt1")),
      chiSquareErrAPrompt2_(settings.getParameter<std::vector<int> >("chiSquareErrAPrompt2")),
      chiSquareErrAPrompt3_(settings.getParameter<std::vector<int> >("chiSquareErrAPrompt3")),
      chiSquareErrBPrompt1_(settings.getParameter<std::vector<double> >("chiSquareErrBPrompt1")),
      chiSquareErrBPrompt2_(settings.getParameter<std::vector<double> >("chiSquareErrBPrompt2")),
      chiSquareErrBPrompt3_(settings.getParameter<std::vector<double> >("chiSquareErrBPrompt3")),

      chiSquareCutDispPattern_(settings.getParameter<std::vector<int> >("chiSquareCutDispPattern")),
      chiSquareCutOffDisp_(settings.getParameter<std::vector<int> >("chiSquareCutOffDisp")),
      chiSquareCutDisp_(settings.getParameter<std::vector<int> >("chiSquareCutDisp")),

      chiSquareCutPromptPattern_(settings.getParameter<std::vector<int> >("chiSquareCutPromptPattern")),
      chiSquareCutOffPrompt_(settings.getParameter<std::vector<int> >("chiSquareCutOffPrompt")),
      chiSquareCutPrompt_(settings.getParameter<std::vector<int> >("chiSquareCutPrompt")),

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
      pointResolutionVertex_(settings.getParameter<double>("pointResolutionVertex")),
      curvResolution1_(settings.getParameter<std::vector<double> >("curvResolution1")),
      curvResolution2_(settings.getParameter<std::vector<double> >("curvResolution2")) {}

std::pair<l1t::KMTFTrack, l1t::KMTFTrack> KMTFCore::chain(const l1t::MuonStubRef& seed,
                                                          const l1t::MuonStubRefVector& stubs) {
  std::vector<l1t::KMTFTrack> pretracks;
  std::vector<int> combinatorics;
  int seedQual;
  switch (seed->depthRegion()) {
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
      throw cms::Exception("KMTFCore") << "Something really bad happend\n";
  }

  l1t::KMTFTrack nullTrack(seed, seed->coord1(), correctedPhiB(seed));
  seedQual = seed->quality();
  for (const auto& mask : combinatorics) {
    l1t::KMTFTrack track(seed, seed->coord1(), correctedPhiB(seed));
    int phiB = correctedPhiB(seed);
    int charge;
    if (phiB == 0)
      charge = 0;
    else
      charge = phiB / fabs(phiB);

    int address = phiB;
    if (track.step() == 4 && (fabs(seed->coord2()) > 15))
      address = charge * 15 * ap_ufixed<PHIBSCALE, PHIBSCALE_INT>(28.5205658);
    if (track.step() == 3 && (fabs(seed->coord2()) > 100))
      address = charge * 100 * ap_ufixed<PHIBSCALE, PHIBSCALE_INT>(28.5205658);
    if (track.step() == 2 && (fabs(seed->coord2()) > 250))
      address = charge * 250 * ap_ufixed<PHIBSCALE, PHIBSCALE_INT>(28.5205658);
    int initialK =
        int(initK_[seed->depthRegion() - 1] * address / (1 + initK2_[seed->depthRegion() - 1] * charge * address));
    if (initialK >= pow(2, BITSCURV - 1))
      initialK = pow(2, BITSCURV - 1) - 1;
    if (initialK <= -pow(2, BITSCURV - 1))
      initialK = -pow(2, BITSCURV - 1) + 1;
    track.setCoordinates(seed->depthRegion(), initialK, seed->coord1(), phiB);
    if (seed->quality() < 6) {
      track.setCoordinates(seed->depthRegion(), initialK, seed->coord1(), 0);
    }
    if (verbose_) {
      edm::LogInfo("KMTFCore") << "Initial state: phiB=" << phiB << " addr=" << address << " K=" << initialK;
    }
    track.setHitPattern(hitPattern(track));
    //set covariance
    l1t::KMTFTrack::CovarianceMatrix covariance;
    float DK = curvResolution1_[track.step() - 1] + curvResolution2_[track.step() - 1] * initialK * initialK;
    if (seed->quality() < 6)
      DK = pow(2, 22);
    //    DK = pow(2,24);
    covariance(0, 0) = DK * 4;
    covariance(0, 1) = 0;
    covariance(0, 2) = 0;
    covariance(1, 0) = 0;
    covariance(1, 1) = float(pointResolutionPhi_);
    covariance(1, 2) = 0;
    covariance(2, 0) = 0;
    covariance(2, 1) = 0;
    if (!(mask == 1 || mask == 2 || mask == 3 || mask == 4 || mask == 5 || mask == 9 || mask == 6 || mask == 10 ||
          mask == 12))
      covariance(2, 2) = float(pointResolutionPhiB_);
    else {
      if (seed->quality() < 6)
        covariance(2, 2) = float(pointResolutionPhiBL_[seed->depthRegion() - 1]);
      else
        covariance(2, 2) = float(pointResolutionPhiBH_[seed->depthRegion() - 1]);
    }
    track.setCovariance(covariance);

    //
    if (verbose_) {
      edm::LogInfo("KMTFCore") << "New Kalman fit staring at step=" << track.step() << ", phi=" << track.positionAngle()
                               << ", phiB=" << track.bendingAngle() << ", with curvature=" << track.curvature();
      edm::LogInfo("KMTFCore") << "BITMASK:" << std::flush;
      for (unsigned int i = 0; i < 4; ++i)
        edm::LogInfo("KMTFCore") << getBit(mask, i) << std::flush;
      edm::LogInfo("KMTFCore") << std::endl;
      edm::LogInfo("KMTFCore") << "------------------------------------------------------";
      edm::LogInfo("KMTFCore") << "------------------------------------------------------";
      edm::LogInfo("KMTFCore") << "------------------------------------------------------";
      edm::LogInfo("KMTFCore") << "stubs:";
      for (const auto& stub : stubs)
        edm::LogInfo("KMTFCore") << "station=" << stub->depthRegion() << " phi=" << stub->coord1()
                                 << " phiB=" << correctedPhiB(stub) << " qual=" << stub->quality()
                                 << " tag=" << stub->id() << " sector=" << stub->phiRegion()
                                 << " wheel=" << stub->etaRegion() << " fineEta= " << stub->eta1() << " "
                                 << stub->eta2();
      edm::LogInfo("KMTFCore") << "------------------------------------------------------";
      edm::LogInfo("KMTFCore") << "------------------------------------------------------";
    }

    bool passedU = false;
    bool passedV = false;
    while (track.step() > 0) {
      // muon station 1
      if (track.step() == 1) {
        track.setCoordinatesAtMuon(track.curvature(), track.positionAngle(), track.bendingAngle());
        passedU = estimateChiSquare(track, false);
        setRank(track, false);

        if (verbose_)
          edm::LogInfo("KMTFCore") << "Calculated Chi2 for displaced track =" << track.approxDispChi2()
                                   << "  Passed Cut=" << passedU;
        calculateEta(track);
        setFourVectors(track);
        //calculate coarse eta
        //////////////////////
        if (verbose_)
          edm::LogInfo("KMTFCore") << "Unconstrained PT  in Muon System: pt=" << track.displacedP4().pt();
        calculateEta(track);
      }

      propagate(track);
      if (verbose_)
        edm::LogInfo("KMTFCore") << "propagated Coordinates step:" << track.step() << "phi=" << track.positionAngle()
                                 << "phiB=" << track.bendingAngle() << "K=" << track.curvature();
      if (track.step() > 0)
        if (getBit(mask, track.step() - 1)) {
          std::pair<bool, uint> bestStub = match(seed, stubs, track.step());
          if (verbose_)
            edm::LogInfo("KMTFCore") << "Found match =" << bestStub.first << " index=" << bestStub.second
                                     << " number of all stubs=" << stubs.size();

          if ((!bestStub.first) || (!update(track, stubs[bestStub.second], mask, seedQual)))
            break;
          if (verbose_) {
            edm::LogInfo("KMTFCore") << "updated Coordinates step:" << track.step() << " phi=" << track.positionAngle()
                                     << " phiB=" << track.bendingAngle() << " K=" << track.curvature();
          }
        }

      if (track.step() == 0) {
        track.setCoordinatesAtVertex(track.curvature(), track.positionAngle(), track.bendingAngle());
        if (verbose_)
          edm::LogInfo("KMTFCore") << " Coordinates before vertex constraint step:" << track.step()
                                   << " phi=" << track.phiAtVertex() << " dxy=" << track.dxy()
                                   << " K=" << track.curvatureAtVertex();

        //apply vertex constraint for non single tracks
        if (track.stubs().size() > 1)
          vertexConstraint(track);

        passedV = estimateChiSquare(track, true);
        setRank(track, true);

        if (verbose_)
          edm::LogInfo("KMTFCore") << "Calculated Chi2 for prompt track =" << track.approxPromptChi2()
                                   << "  Passed Cut=" << passedV;

        if (verbose_) {
          edm::LogInfo("KMTFCore") << " Coordinates after vertex constraint step:" << track.step()
                                   << " phi=" << track.phiAtVertex() << " dxy=" << track.dxy()
                                   << " K=" << track.curvatureAtVertex()
                                   << "  maximum local chi2=" << track.approxPromptChi2();
          edm::LogInfo("KMTFCore") << "------------------------------------------------------";
          edm::LogInfo("KMTFCore") << "------------------------------------------------------";
        }
        setFourVectors(track);
        //finally set the displaced or prompt ID
        track.setIDFlag(passedV, passedU);

        if (verbose_)
          edm::LogInfo("KMTFCore") << "Floating point coordinates at vertex: pt=" << track.pt()
                                   << ", eta=" << track.eta() << " phi=" << track.phi();

        pretracks.push_back(track);
      }
    }
  }
  if (verbose_) {
    if (!pretracks.empty())
      edm::LogInfo("KMTFCore") << "-----Kalman Algo at station " << seed->depthRegion() << " (uncleaned)-----";
    for (const auto& track : pretracks)
      edm::LogInfo("KMTFCore") << "Kalman Track charge=" << track.charge() << " pt=" << track.pt()
                               << " hit pattern = " << track.hitPattern() << " eta=" << track.eta()
                               << " phi=" << track.phi() << " curvature=" << track.curvatureAtVertex()
                               << " curvature STA =" << track.curvatureAtMuon()
                               << " stubs=" << int(track.stubs().size()) << " chi2=" << track.approxPromptChi2() << ","
                               << track.approxDispChi2() << " rank=" << track.rankPrompt() << "," << track.rankDisp()
                               << " pts=" << track.pt() << " " << track.displacedP4().pt() << "   ID=" << track.id();
  }
  //Now for all the pretracks we need only one vertex constrained and one vertex unconstrained
  //so we clean twice
  if (verbose_)
    edm::LogInfo("KMTFCore") << "Chain Reconstructed " << pretracks.size()
                             << " pretracks, now cleaning them separately";

  std::vector<l1t::KMTFTrack> cleanedPrompt = clean(pretracks, seed->depthRegion(), true);
  std::vector<l1t::KMTFTrack> cleanedDisp = clean(pretracks, seed->depthRegion(), false);
  if (verbose_)
    edm::LogInfo("KMTFCore") << "Cleaned Chain tracks prompt=" << cleanedPrompt.size()
                             << " displaced=" << cleanedDisp.size();

  if (cleanedPrompt.empty() && cleanedDisp.empty())
    return std::make_pair(nullTrack, nullTrack);
  else if ((!cleanedPrompt.empty()) && cleanedDisp.empty())
    return std::make_pair(cleanedPrompt[0], nullTrack);
  else if (cleanedPrompt.empty() && (!cleanedDisp.empty()))
    return std::make_pair(nullTrack, cleanedDisp[0]);
  else
    return std::make_pair(cleanedPrompt[0], cleanedDisp[0]);
}

std::vector<l1t::KMTFTrack> KMTFCore::clean(const std::vector<l1t::KMTFTrack>& tracks, uint seed, bool vertex) {
  std::vector<l1t::KMTFTrack> out;

  std::map<uint, int> infoRank;
  std::map<uint, l1t::KMTFTrack> infoTrack;
  for (uint i = 1; i <= 15; ++i) {
    infoRank[i] = -1;
  }

  for (const auto& track : tracks) {
    if (vertex) {
      if ((track.id() & 0x1) == 0)
        continue;
      if (verbose_)
        edm::LogInfo("KMTFCore") << "Chain Cleaning : Pre Track = pattern = " << track.rankPrompt()
                                 << " rank=" << track.hitPattern();
      infoRank[track.hitPattern()] = track.rankPrompt();
      infoTrack[track.hitPattern()] = track;

    } else {
      if ((track.id() & 0x2) == 0)
        continue;
      infoRank[track.hitPattern()] = track.rankDisp();
      infoTrack[track.hitPattern()] = track;
    }
  }

  int selected = 15;
  if (seed == 4)  //station 4 seeded
  {
    int sel6 = infoRank[10] >= infoRank[12] ? 10 : 12;
    int sel5 = infoRank[14] >= infoRank[9] ? 14 : 9;
    int sel4 = infoRank[11] >= infoRank[13] ? 11 : 13;
    int sel3 = infoRank[sel6] >= infoRank[sel5] ? sel6 : sel5;
    int sel2 = infoRank[sel4] >= infoRank[sel3] ? sel4 : sel3;
    int sel1 = infoRank[15] >= infoRank[sel2] ? 15 : sel2;
    if (vertex)
      selected = infoRank[sel1] > 0 ? sel1 : 8;
    else
      selected = sel1;
  }
  if (seed == 3)  //station 3 seeded
  {
    int sel2 = infoRank[5] >= infoRank[6] ? 5 : 6;
    int sel1 = infoRank[7] >= infoRank[sel2] ? 7 : sel2;
    if (vertex)
      selected = infoRank[sel1] > 0 ? sel1 : 4;
    else
      selected = sel1;
  }
  if (seed == 2)  //station 2 seeded
  {
    if (vertex)
      selected = infoRank[3] > 0 ? 3 : 2;
    else
      selected = 3;
  }
  if (seed == 1)  //station 1 seeded
    selected = 1;

  auto search = infoTrack.find(selected);
  if (search != infoTrack.end())
    out.push_back(search->second);

  return out;
}

std::pair<bool, uint> KMTFCore::match(const l1t::MuonStubRef& seed, const l1t::MuonStubRefVector& stubs, int step) {
  l1t::MuonStubRefVector selected;
  std::map<uint, uint> diffInfo;
  for (uint i = 0; i < 32; ++i) {
    diffInfo[i] = 60000;
  }

  int wheel = seed->etaRegion();
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
  //calculate the distance of all stubs

  for (unsigned int N = 0; N < stubs.size(); ++N) {
    const l1t::MuonStubRef& stub = stubs[N];
    //Should not be stubs with tag=4 but there are, so skip those
    if (verbose_)
      edm::LogInfo("KMTFCore") << "testing stub on depth=" << stub->depthRegion() << " for step=" << step;

    if (stub->depthRegion() != step)
      continue;
    if (verbose_)
      edm::LogInfo("KMTFCore") << "Passed";

    uint distance = fabs(wrapAround((seed->coord1() - stub->coord1()) >> 3, 32768));
    //if the wheels are not adjacent make this huge
    if (!((stub->etaRegion() == wheel) || (stub->etaRegion() == innerWheel)))
      distance = 60000;

    diffInfo[N] = distance;
  }

  uint s1_1 = matchAbs(diffInfo, 0, 1);
  uint s1_2 = matchAbs(diffInfo, 2, 3);
  uint s1_3 = matchAbs(diffInfo, 4, 5);
  uint s1_4 = matchAbs(diffInfo, 6, 7);
  uint s1_5 = matchAbs(diffInfo, 8, 9);
  uint s1_6 = matchAbs(diffInfo, 10, 11);
  uint s1_7 = matchAbs(diffInfo, 12, 13);
  uint s1_8 = matchAbs(diffInfo, 14, 15);
  uint s1_9 = matchAbs(diffInfo, 16, 17);
  uint s1_10 = matchAbs(diffInfo, 18, 19);
  uint s1_11 = matchAbs(diffInfo, 20, 21);
  uint s1_12 = matchAbs(diffInfo, 22, 23);
  uint s1_13 = matchAbs(diffInfo, 24, 25);
  uint s1_14 = matchAbs(diffInfo, 26, 27);
  uint s1_15 = matchAbs(diffInfo, 28, 29);
  uint s1_16 = matchAbs(diffInfo, 30, 31);

  uint s2_1 = matchAbs(diffInfo, s1_1, s1_2);
  uint s2_2 = matchAbs(diffInfo, s1_3, s1_4);
  uint s2_3 = matchAbs(diffInfo, s1_5, s1_6);
  uint s2_4 = matchAbs(diffInfo, s1_7, s1_8);
  uint s2_5 = matchAbs(diffInfo, s1_9, s1_10);
  uint s2_6 = matchAbs(diffInfo, s1_11, s1_12);
  uint s2_7 = matchAbs(diffInfo, s1_13, s1_14);
  uint s2_8 = matchAbs(diffInfo, s1_15, s1_16);

  uint s3_1 = matchAbs(diffInfo, s2_1, s2_2);
  uint s3_2 = matchAbs(diffInfo, s2_3, s2_4);
  uint s3_3 = matchAbs(diffInfo, s2_5, s2_6);
  uint s3_4 = matchAbs(diffInfo, s2_7, s2_8);

  uint s4_1 = matchAbs(diffInfo, s3_1, s3_2);
  uint s4_2 = matchAbs(diffInfo, s3_3, s3_4);

  uint s5 = matchAbs(diffInfo, s4_1, s4_2);

  if (diffInfo[s5] != 60000)
    return std::make_pair(true, s5);
  else
    return std::make_pair(false, 0);
}

int KMTFCore::correctedPhiB(const l1t::MuonStubRef& stub) {
  return ap_fixed<BITSPHIB, BITSPHIB, AP_TRN_ZERO, AP_SAT>(ap_ufixed<PHIBSCALE, PHIBSCALE_INT>(28.5205658) *
                                                           stub->coord2());
}

void KMTFCore::propagate(l1t::KMTFTrack& track) {
  int K = track.curvature();
  int phi = track.positionAngle();
  int phiB = track.bendingAngle();
  unsigned int step = track.step();

  int charge = 1;
  if (K != 0)
    charge = K / fabs(K);

  int KBound = K;
  if (KBound > pow(2, BITSCURV - 3) - 1)
    KBound = pow(2, BITSCURV - 3) - 1;
  if (KBound < -(pow(2, BITSCURV - 3) - 1))
    KBound = -(pow(2, BITSCURV - 3) - 1);

  int deltaK = 0;
  int KNew = K;
  if (step == 1) {
    ap_ufixed<17, 0> eLoss = ap_ufixed<17, 0>(eLoss_[step - 1]);
    ap_fixed<BITSCURV - 2, BITSCURV - 2> Kint = ap_fixed<BITSCURV - 2, BITSCURV - 2>(KBound);
    ap_fixed<16, 5> eK = eLoss * Kint;
    if (charge < 0)
      eK = -eK;
    ap_fixed<BITSCURV, BITSCURV> KnewInt = ap_fixed<BITSCURV, BITSCURV>(K) - eK * Kint;
    KNew = KnewInt;
    if (verbose_)
      edm::LogInfo("KMTFCore") << "propagate to vertex Kint=" << Kint.to_int() << " ek=" << eK.to_float()
                               << " Knew=" << KNew;
  }

  //phi propagation
  ap_fixed<BITSCURV, BITSCURV> phi11 = ap_fixed<BITSPARAM + 1, 2>(aPhi_[step - 1]) * ap_fixed<BITSCURV, BITSCURV>(K);
  ap_fixed<BITSPHIB, BITSPHIB> phi12 =
      ap_fixed<BITSPARAM + 1, 2>(-bPhi_[step - 1]) * ap_fixed<BITSPHIB, BITSPHIB>(phiB);

  if (verbose_) {
    edm::LogInfo("KMTFCore") << "phi prop = " << K << " * " << ap_fixed<BITSPARAM + 1, 2>(aPhi_[step - 1]).to_float()
                             << " = " << phi11.to_int() << ", " << phiB << " * "
                             << ap_fixed<BITSPARAM + 1, 2>(-bPhi_[step - 1]).to_float() << " = " << phi12.to_int();
  }
  int phiNew = ap_fixed<BITSPHI, BITSPHI>(phi + phi11 + phi12);

  //phiB propagation
  ap_fixed<BITSCURV, BITSCURV> phiB11 = ap_fixed<BITSPARAM, 1>(aPhiB_[step - 1]) * ap_fixed<BITSCURV, BITSCURV>(K);
  ap_fixed<BITSPHIB + 1, BITSPHIB + 1> phiB12 =
      ap_ufixed<BITSPARAM + 1, 1>(bPhiB_[step - 1]) * ap_fixed<BITSPHIB, BITSPHIB>(phiB);
  int phiBNew = ap_fixed<BITSPHIB, BITSPHIB>(phiB11 + phiB12);
  if (verbose_) {
    edm::LogInfo("KMTFCore") << "phiB prop = " << K << " * " << ap_fixed<BITSPARAM + 1, 2>(aPhiB_[step - 1]).to_float()
                             << " = " << phiB11.to_int() << ", " << phiB << " * "
                             << ap_ufixed<BITSPARAM + 1, 1>(bPhiB_[step - 1]).to_float() << " = " << phiB12.to_int();
  }

  //Only for the propagation to vertex we use second order;
  if (step == 1) {
    ap_fixed<10, 4> aPhiB = aPhiB_[step - 1];
    ap_ufixed<16, 0> aPhiBNLO = aPhiBNLO_[step - 1];

    ap_fixed<BITSCURV - 2, BITSCURV - 2> Kint = ap_fixed<BITSCURV - 2, BITSCURV - 2>(KBound);
    ap_fixed<BITSCURV - 1, BITSCURV - 1> aK = aPhiB * Kint;
    ap_fixed<16, 5> eK = aPhiBNLO * Kint;
    if (charge < 0)
      eK = -eK;
    ap_fixed<BITSPHIB + 2, BITSPHIB + 2> DXY = aK + eK * Kint;
    //ap_fixed<BITSPHIB+3,BITSPHIB+3> diff = DXY - ap_fixed<BITSPHIB, BITSPHIB>(phiB);

    ap_fixed<BITSPHIB, BITSPHIB, AP_TRN_ZERO, AP_SAT_SYM> diff = DXY - ap_fixed<BITSPHIB, BITSPHIB>(phiB);

    phiBNew = ap_fixed<BITSPHIB, BITSPHIB>(diff);
    if (verbose_) {
      edm::LogInfo("KMTFCore") << "Vertex phiB prop = " << DXY.to_int() << "(=" << aK.to_int() << " +"
                               << (eK * Kint).to_int() << ") - " << ap_fixed<BITSPHIB, BITSPHIB>(phiB).to_int() << " = "
                               << phiBNew;
    }
  }
  ///////////////////////////////////////////////////////
  //Rest of the stuff  is for the offline version only
  //where we want to check what is happening in the covariance matrix

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
  l1t::KMTFTrack::CovarianceMatrix cov(covLine.begin(), covLine.end());
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
    edm::LogInfo("KMTFCore") << "Covariance term for phiB = " << cov(2, 2);
    edm::LogInfo("KMTFCore") << "Multiple scattering term for phiB = " << MS(2, 2);
  }

  track.setCovariance(cov);
  track.setCoordinates(step - 1, KNew, phiNew, phiBNew);
}

bool KMTFCore::update(l1t::KMTFTrack& track, const l1t::MuonStubRef& stub, int mask, int seedQual) {
  //  updateEta(track, stub);
  if (useOfflineAlgo_) {
    if (mask == 3 || mask == 5 || mask == 9 || mask == 6 || mask == 10 || mask == 12)
      return updateOffline(track, stub);
    else
      return updateOffline1D(track, stub);

  } else
    return updateLUT(track, stub, mask, seedQual);
}

bool KMTFCore::updateOffline(l1t::KMTFTrack& track, const l1t::MuonStubRef& stub) {
  int trackK = track.curvature();
  int trackPhi = track.positionAngle();
  int trackPhiB = track.bendingAngle();

  int phi = stub->coord1();
  int phiB = correctedPhiB(stub);

  Vector2 residual;
  residual[0] = ap_fixed<BITSPHI, BITSPHI>(phi - trackPhi);
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
  if (stub->quality() < 6)
    R(1, 1) = pointResolutionPhiBL_[track.step() - 1];
  else
    R(1, 1) = pointResolutionPhiBH_[track.step() - 1];

  const std::vector<double>& covLine = track.covariance();
  l1t::KMTFTrack::CovarianceMatrix cov(covLine.begin(), covLine.end());
  CovarianceMatrix2 S = ROOT::Math::Similarity(H, cov) + R;
  if (!S.Invert())
    return false;
  Matrix32 Gain = cov * ROOT::Math::Transpose(H) * S;

  track.setKalmanGain(
      track.step(), fabs(trackK), Gain(0, 0), Gain(0, 1), Gain(1, 0), Gain(1, 1), Gain(2, 0), Gain(2, 1));

  int KNew = (trackK + int(Gain(0, 0) * residual(0) + Gain(0, 1) * residual(1)));
  if (fabs(KNew) > pow(2, BITSCURV - 1))
    return false;

  int phiNew = wrapAround(trackPhi + residual(0), pow(2, BITSPHI - 1));
  int phiBNew = wrapAround(trackPhiB + int(Gain(2, 0) * residual(0) + Gain(2, 1) * residual(1)), pow(2, BITSPHIB - 1));

  track.setResidual(stub->depthRegion() - 1, fabs(phi - phiNew) + fabs(phiB - phiBNew));

  if (verbose_) {
    edm::LogInfo("KMTFCore") << "residual " << phi << " - " << trackPhi << " = " << int(residual[0]) << " " << phiB
                             << " - " << trackPhiB << " = " << int(residual[1]);
    edm::LogInfo("KMTFCore") << "Gains offline: " << Gain(0, 0) << " " << Gain(0, 1) << " " << Gain(2, 0) << " "
                             << Gain(2, 1);
    edm::LogInfo("KMTFCore") << " K = " << trackK << " + " << Gain(0, 0) << " * " << residual(0) << " + " << Gain(0, 1)
                             << " * " << residual(1);
    edm::LogInfo("KMTFCore") << " phiB = " << trackPhiB << " + " << Gain(2, 0) << " * " << residual(0) << " + "
                             << Gain(2, 1) << " * " << residual(1);
  }

  track.setCoordinates(track.step(), KNew, phiNew, phiBNew);
  Matrix33 covNew = cov - Gain * (H * cov);
  l1t::KMTFTrack::CovarianceMatrix c;

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
    edm::LogInfo("KMTFCore") << "Post Fit Covariance Matrix " << cov(0, 0) << " " << cov(1, 1) << " " << cov(2, 2);
  }

  track.setCovariance(c);
  track.addStub(stub);
  track.setHitPattern(hitPattern(track));

  return true;
}

bool KMTFCore::updateOffline1D(l1t::KMTFTrack& track, const l1t::MuonStubRef& stub) {
  int trackK = track.curvature();
  int trackPhi = track.positionAngle();
  int trackPhiB = track.bendingAngle();

  int phi = stub->coord1();

  double residual = ap_fixed<BITSPHI, BITSPHI>(phi - trackPhi);

  if (verbose_)
    edm::LogInfo("KMTFCore") << "residuals " << phi << " - " << trackPhi << " = " << int(residual);

  Matrix13 H;
  H(0, 0) = 0.0;
  H(0, 1) = 1.0;
  H(0, 2) = 0.0;

  const std::vector<double>& covLine = track.covariance();
  l1t::KMTFTrack::CovarianceMatrix cov(covLine.begin(), covLine.end());

  double S = ROOT::Math::Similarity(H, cov)(0, 0) + pointResolutionPhi_;
  if (S == 0.0)
    return false;
  Matrix31 Gain = cov * ROOT::Math::Transpose(H) / S;

  track.setKalmanGain(track.step(), fabs(trackK), Gain(0, 0), 0.0, Gain(1, 0), 0.0, Gain(2, 0), 0.0);

  int KNew = wrapAround(trackK + int(Gain(0, 0) * residual), pow(2, BITSCURV - 1));
  int phiNew = wrapAround(trackPhi + residual, pow(2, BITSPHI - 1));
  int phiBNew = wrapAround(trackPhiB + int(Gain(2, 0) * residual), pow(2, BITSPHIB - 1));
  track.setCoordinates(track.step(), KNew, phiNew, phiBNew);
  Matrix33 covNew = cov - Gain * (H * cov);
  l1t::KMTFTrack::CovarianceMatrix c;

  if (verbose_) {
    edm::LogInfo("KMTFCore") << "phiUpdate: " << int(Gain(0, 0) * residual) << " " << int(Gain(2, 0) * residual);
    edm::LogInfo("KMTFCore") << " K = " << trackK << " + " << Gain(0, 0) << " * " << residual;
    edm::LogInfo("KMTFCore") << " phiBNew = " << trackPhiB << " + " << Gain(2, 0) << " * " << residual;
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

bool KMTFCore::updateLUT(l1t::KMTFTrack& track, const l1t::MuonStubRef& stub, int mask, int seedQual) {
  int trackK = track.curvature();
  int trackPhi = track.positionAngle();
  int trackPhiB = track.bendingAngle();

  int phi = stub->coord1();
  int phiB = correctedPhiB(stub);

  Vector2 residual;
  ap_fixed<BITSPHI, BITSPHI> residualPhi = phi - trackPhi;
  ap_fixed<BITSPHIB + 1, BITSPHIB + 1> residualPhiB = phiB - trackPhiB;

  if (verbose_)
    edm::LogInfo("KMTFCore") << "residual " << phi << " - " << trackPhi << " = " << residualPhi.to_int() << " " << phiB
                             << " - " << trackPhiB << " = " << residualPhiB.to_int();

  uint absK = fabs(trackK);
  if (absK > pow(2, BITSCURV - 2) - 1)
    absK = pow(2, BITSCURV - 2) - 1;

  std::vector<float> GAIN;
  if (verbose_) {
    edm::LogInfo("KMTFCore") << "Looking up LUTs for mask=" << mask << " with hit pattern=" << track.hitPattern();
  }
  //For the three stub stuff use only gains 0 and 4
  if (!(mask == 3 || mask == 5 || mask == 9 || mask == 6 || mask == 10 || mask == 12)) {
    GAIN = lutService_->trackGain(track.step(), track.hitPattern(), absK / 16);
    GAIN[1] = 0.0;
    GAIN[3] = 0.0;

  } else {
    GAIN = lutService_->trackGain2(track.step(), track.hitPattern(), absK / 32, seedQual, stub->quality());
  }
  if (verbose_) {
    edm::LogInfo("KMTFCore") << "Gains (fp): " << GAIN[0] << " " << GAIN[1] << " " << GAIN[2] << " " << GAIN[3];

    if (!(mask == 3 || mask == 5 || mask == 9 || mask == 6 || mask == 10 || mask == 12))
      edm::LogInfo("KMTFCore") << "Addr=" << absK / 16
                               << "   gain0=" << ap_ufixed<GAIN_0, GAIN_0INT>(GAIN[0]).to_float() << " gain4=-"
                               << ap_ufixed<GAIN_4, GAIN_4INT>(GAIN[2]).to_float();
    else
      edm::LogInfo("KMTFCore") << "Addr=" << absK / 32 << "   " << ap_ufixed<GAIN2_0, GAIN2_0INT>(GAIN[0]).to_float()
                               << " -" << ap_ufixed<GAIN2_1, GAIN2_1INT>(GAIN[1]).to_float() << " "
                               << ap_ufixed<GAIN2_4, GAIN2_4INT>(GAIN[2]).to_float() << " "
                               << ap_ufixed<GAIN2_5, GAIN2_5INT>(GAIN[3]).to_float();
  }

  track.setKalmanGain(track.step(), fabs(trackK), GAIN[0], GAIN[1], 1, 0, GAIN[2], GAIN[3]);

  int KNew;
  if (!(mask == 3 || mask == 5 || mask == 9 || mask == 6 || mask == 10 || mask == 12)) {
    KNew = ap_fixed<BITSPHI + 9, BITSPHI + 9>(ap_fixed<BITSCURV, BITSCURV>(trackK) +
                                              ap_ufixed<GAIN_0, GAIN_0INT>(GAIN[0]) * residualPhi);
    if (verbose_) {
      edm::LogInfo("KMTFCore") << "K = " << KNew << " = " << ap_fixed<BITSCURV, BITSCURV>(trackK).to_int() << " + "
                               << ap_ufixed<GAIN_0, GAIN_0INT>(GAIN[0]).to_float() << "*" << residualPhi.to_int();
    }
  } else {
    ap_fixed<BITSPHI + 7, BITSPHI + 7> k11 = ap_ufixed<GAIN2_0, GAIN2_0INT>(GAIN[0]) * residualPhi;
    ap_fixed<BITSPHIB + 4, BITSPHIB + 4> k12 = ap_ufixed<GAIN2_1, GAIN2_1INT>(GAIN[1]) * residualPhiB;

    KNew = ap_fixed<BITSPHI + 9, BITSPHI + 9>(ap_fixed<BITSCURV, BITSCURV>(trackK) + k11 - k12);
    if (verbose_) {
      edm::LogInfo("KMTFCore") << "K = " << KNew << " = " << ap_fixed<BITSCURV, BITSCURV>(trackK).to_int() << " + "
                               << k11.to_int() << " + " << k12.to_int();
    }
  }
  if ((KNew > (pow(2, BITSCURV - 1) - 1)) || (KNew < -(pow(2, BITSCURV - 1)))) {
    if (verbose_)
      edm::LogInfo("KMTFCore") << "K has saturated, track has extremely low energy";
    return false;
  }
  KNew = wrapAround(KNew, pow(2, BITSCURV - 1));
  int phiNew = phi;

  //different products for different firmware logic
  ap_fixed<BITSPHI + 5, BITSPHI + 5> pbdouble_0 = ap_ufixed<GAIN2_4, GAIN2_4INT>(GAIN[2]) * residualPhi;
  ap_fixed<BITSPHIB + 4, BITSPHIB + 4> pb_1 = ap_ufixed<GAIN2_5, GAIN2_5INT>(GAIN[3]) * residualPhiB;
  ap_fixed<BITSPHI + 9, BITSPHI + 5> pb_0 = ap_ufixed<GAIN_4, GAIN_4INT>(GAIN[2]) * residualPhi;

  if (verbose_) {
    edm::LogInfo("KMTFCore") << "phiupdate " << pb_0.to_float() << " " << pb_1.to_float() << " "
                             << pbdouble_0.to_float();
  }

  int phiBNew;
  if (!(mask == 3 || mask == 5 || mask == 9 || mask == 6 || mask == 10 || mask == 12)) {
    phiBNew = ap_fixed<BITSPHI + 8, BITSPHI + 8>(ap_fixed<BITSPHIB, BITSPHIB>(trackPhiB) -
                                                 ap_ufixed<GAIN_4, GAIN_4INT>(GAIN[2]) * residualPhi);
  } else {
    phiBNew = ap_fixed<BITSPHI + 7, BITSPHI + 7>(ap_fixed<BITSPHIB, BITSPHIB>(trackPhiB) + pb_1 - pbdouble_0);
  }

  if ((phiBNew > (pow(2, BITSPHIB - 1) - 1)) || (phiBNew < (-pow(2, BITSPHIB - 1))))
    return false;

  track.setCoordinates(track.step(), KNew, phiNew, phiBNew);
  track.addStub(stub);
  track.setHitPattern(hitPattern(track));
  if (verbose_) {
    edm::LogInfo("KMTFCore") << "Stub station =" << stub->depthRegion();

    edm::LogInfo("KMTFCore") << "Updated Hit Pattern =" << track.hitPattern();
  }

  return true;
}

void KMTFCore::vertexConstraint(l1t::KMTFTrack& track) {
  if (useOfflineAlgo_)
    vertexConstraintOffline(track);
  else
    vertexConstraintLUT(track);
}

void KMTFCore::vertexConstraintOffline(l1t::KMTFTrack& track) {
  double residual = -track.dxy();
  Matrix13 H;
  H(0, 0) = 0;
  H(0, 1) = 0;
  H(0, 2) = 1;

  const std::vector<double>& covLine = track.covariance();
  l1t::KMTFTrack::CovarianceMatrix cov(covLine.begin(), covLine.end());

  double S = (ROOT::Math::Similarity(H, cov))(0, 0) + pointResolutionVertex_;
  S = 1.0 / S;
  Matrix31 Gain = cov * (ROOT::Math::Transpose(H)) * S;
  track.setKalmanGain(track.step(), fabs(track.curvature()), Gain(0, 0), Gain(1, 0), Gain(2, 0));

  if (verbose_) {
    edm::LogInfo("KMTFCore") << "sigma3=" << cov(0, 3) << " sigma6=" << cov(3, 3);
    edm::LogInfo("KMTFCore") << " K = " << track.curvature() << " + " << Gain(0, 0) << " * " << residual;
  }

  int KNew = wrapAround(int(track.curvature() + Gain(0, 0) * residual), pow(2, BITSCURV - 1));
  int phiNew = wrapAround(int(track.positionAngle() + Gain(1, 0) * residual), pow(2, BITSPHI));
  int dxyNew = wrapAround(int(track.dxy() + Gain(2, 0) * residual), pow(2, BITSPHIB));
  if (verbose_)
    edm::LogInfo("KMTFCore") << "Post fit impact parameter=" << dxyNew;
  track.setCoordinatesAtVertex(KNew, phiNew, -residual);
  Matrix33 covNew = cov - Gain * (H * cov);
  l1t::KMTFTrack::CovarianceMatrix c;
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

void KMTFCore::vertexConstraintLUT(l1t::KMTFTrack& track) {
  double residual = -track.dxy();
  uint absK = fabs(track.curvature());
  if (absK > pow(2, BITSCURV - 4) - 1)
    absK = pow(2, BITSCURV - 4) - 1;

  std::pair<float, float> GAIN = lutService_->vertexGain(track.hitPattern(), absK / 4);
  track.setKalmanGain(track.step(), fabs(track.curvature()), GAIN.first, GAIN.second, -1);

  ap_fixed<BITSCURV, BITSCURV> k_0 =
      -(ap_ufixed<GAIN_V0, GAIN_V0INT>(fabs(GAIN.first))) * ap_fixed<BITSPHIB, BITSPHIB>(residual);
  int KNew = ap_fixed<BITSCURV, BITSCURV>(k_0 + ap_fixed<BITSCURV, BITSCURV>(track.curvature()));

  if (verbose_) {
    edm::LogInfo("KMTFCore") << "VERTEX GAIN(" << absK / 4 << ")= -"
                             << ap_ufixed<GAIN_V0, GAIN_V0INT>(fabs(GAIN.first)).to_float() << " * "
                             << ap_fixed<BITSPHIB, BITSPHIB>(residual).to_int() << " = " << k_0.to_int();
  }

  //int p_0 = fp_product(GAIN.second, int(residual), 7);
  int p_0 = GAIN.second * int(residual);
  int phiNew = wrapAround(track.positionAngle() + p_0, pow(2, BITSPHI - 1));
  track.setCoordinatesAtVertex(KNew, phiNew, -residual);
}

int KMTFCore::hitPattern(const l1t::KMTFTrack& track) {
  unsigned int mask = 0;
  for (const auto& stub : track.stubs()) {
    mask = mask + round(pow(2, stub->depthRegion() - 1));
  }
  return mask;
}

int KMTFCore::customBitmask(unsigned int bit1, unsigned int bit2, unsigned int bit3, unsigned int bit4) {
  return bit1 * 1 + bit2 * 2 + bit3 * 4 + bit4 * 8;
}

bool KMTFCore::getBit(int bitmask, int pos) { return (bitmask & (1 << pos)) >> pos; }

void KMTFCore::setFourVectors(l1t::KMTFTrack& track) {
  int etaINT = track.coarseEta();
  double lsbEta = M_PI / pow(2, 12);

  int charge = 1;
  if (track.curvatureAtVertex() < 0)
    charge = -1;

  int ptC = ptLUT(track.curvatureAtVertex());
  int ptU = ptLUT(track.curvatureAtMuon());

  //if only one stub return the phi of the stub.
  //Also set PT =0 and dxy=0
  if (track.stubs().size() == 1) {
    ptC = 0;
    track.setCoordinatesAtMuon(track.curvatureAtMuon(), track.stubs()[0]->coord1(), track.phiBAtMuon());
    track.setCoordinatesAtVertex(track.curvatureAtVertex(), track.phiAtVertex(), 0);
  }
  track.setPt(ptC, ptU);
  //shift the dxy by 10 bits
  track.setCoordinatesAtVertex(track.curvatureAtVertex(), track.phiAtVertex(), track.dxy() / 1024);

  //vertex
  double pt = ptC * 0.03125;
  double phi = (track.phiAtMuon() / 32) * M_PI / pow(2, 12);
  double eta = etaINT * lsbEta;
  track.setPtEtaPhi(pt, eta, phi);
  track.setCharge(charge);
  pt = double(ptLUT(track.curvatureAtMuon())) * 0.03125;
  track.setPtEtaPhiDisplaced(pt, eta, phi);
}

bool KMTFCore::estimateChiSquare(l1t::KMTFTrack& track, bool vertex) {
  int K;
  uint chi = 0;
  uint chiErr = 0;

  //exception for 1 stub tracks and vertex constraint
  //displaced track not allowed / prompt are allowed
  if (track.stubs().size() == 1) {
    if (!vertex)
      return false;
    else {
      track.setApproxChi2(127, chiErr, vertex);
      return true;
    }
  }

  std::vector<double> prop;
  std::vector<double> propErrB;
  std::vector<int> propErrA;
  std::vector<int> cut;

  const l1t::MuonStubRef& innerStub = track.stubs()[track.stubs().size() - 1];

  if (vertex) {
    K = track.curvatureAtVertex();
    if (innerStub->depthRegion() == 1) {
      prop = chiSquarePrompt1_;
      propErrA = chiSquareErrAPrompt1_;
      propErrB = chiSquareErrBPrompt1_;
    } else if (innerStub->depthRegion() == 2) {
      prop = chiSquarePrompt2_;
      propErrA = chiSquareErrAPrompt2_;
      propErrB = chiSquareErrBPrompt2_;

    } else if (innerStub->depthRegion() == 3) {
      prop = chiSquarePrompt3_;
      propErrA = chiSquareErrAPrompt3_;
      propErrB = chiSquareErrBPrompt3_;
    }
  } else {
    K = track.curvatureAtMuon();
    if (innerStub->depthRegion() == 1) {
      prop = chiSquareDisp1_;
      propErrA = chiSquareErrADisp1_;
      propErrB = chiSquareErrBDisp1_;

    } else if (innerStub->depthRegion() == 2) {
      prop = chiSquareDisp2_;
      propErrA = chiSquareErrADisp2_;
      propErrB = chiSquareErrBDisp2_;

    } else if (innerStub->depthRegion() == 3) {
      prop = chiSquareDisp3_;
      propErrA = chiSquareErrADisp3_;
      propErrB = chiSquareErrBDisp3_;
    }
  }

  ap_fixed<BITSCURV, BITSCURV> Kdig = K;
  ap_fixed<BITSCURV - 3, BITSCURV - 3> Kshifted = Kdig >> 4;
  ap_ufixed<BITSCURV - 4, BITSCURV - 4> absK = 0;

  if (Kshifted < 0)
    absK = (-Kshifted);
  else
    absK = (Kshifted);

  for (unsigned int i = 0; i < (track.stubs().size() - 1); i++) {
    const l1t::MuonStubRef& stub = track.stubs()[i];

    int diffPhi = ap_fixed<BITSPHI, BITSPHI>(stub->coord1() - innerStub->coord1()) >> 4;
    int diffPhiB = ap_fixed<BITSPHI, BITSPHI>(correctedPhiB(stub) - correctedPhiB(innerStub)) >> 4;
    if (vertex)
      diffPhiB = 0;

    if (verbose_)
      edm::LogInfo("KMTFCore") << "Error propagation coefficients A="
                               << propErrA[stub->depthRegion() - innerStub->depthRegion() - 1]
                               << " B=" << propErrB[stub->depthRegion() - innerStub->depthRegion() - 1] << " BK = "
                               << uint(ap_fixed<8, 2>(propErrB[stub->depthRegion() - innerStub->depthRegion() - 1]) *
                                       (absK >> 4));
    uint positionError = propErrA[stub->depthRegion() - innerStub->depthRegion() - 1];
    if (stub->quality() < 6 || innerStub->quality() < 6)
      positionError = positionError * 2;

    uint err =
        positionError + uint(ap_fixed<8, 2>(propErrB[stub->depthRegion() - innerStub->depthRegion() - 1]) * absK);
    ap_fixed<8, 2> propC = ap_fixed<8, 2>(prop[stub->depthRegion() - innerStub->depthRegion() - 1]);
    ap_fixed<BITSCURV - 3, BITSCURV - 3> AK = -propC * Kshifted;
    int delta = diffPhi + diffPhiB + AK;
    uint absDelta = delta < 0 ? -delta : delta;
    chi = chi + absDelta;
    chiErr = chiErr + err;
    if (verbose_) {
      edm::LogInfo("KMTFCore") << "Chi Square stub for track with pattern=" << track.hitPattern()
                               << "   inner stub depth=" << innerStub->depthRegion() << "-> AK=" << int(AK)
                               << " stubDepth=" << stub->depthRegion() << " diff1=" << diffPhi << " diff2=" << diffPhiB
                               << " delta=" << absDelta << " absK=" << uint(absK) << " err=" << err;
    }
  }
  if (verbose_) {
    edm::LogInfo("KMTFCore") << "Chi Square =" << chi << " ChiSquare Error = " << chiErr;
  }

  track.setApproxChi2(chi, chiErr, vertex);
  if (chi > chiErr)
    return false;
  return true;
}

void KMTFCore::setRank(l1t::KMTFTrack& track, bool vertex) {
  uint chi = 0;
  if (vertex)
    chi = track.approxPromptChi2() > 127 ? 127 : track.approxPromptChi2();
  else
    chi = track.approxDispChi2() > 127 ? 127 : track.approxDispChi2();

  uint Q = 0;
  for (const auto& stub : track.stubs()) {
    if (stub->quality() > 2)
      Q = Q + 2;
    else
      Q = Q + 1;
  }
  uint rank = Q * 4 - chi + 125;

  //exception for  track 1100
  if (hitPattern(track) == customBitmask(0, 0, 1, 1)) {
    rank = 1;
  }

  //Exception for one stub tracks.
  if (track.stubs().size() == 1)
    rank = 0;

  if (verbose_)
    edm::LogInfo("KMTFCore") << "Rank Calculated for vertex=" << vertex << "  = " << rank;
  track.setRank(rank, vertex);
}

int KMTFCore::wrapAround(int value, int maximum) {
  if (value > maximum - 1)
    return wrapAround(value - 2 * maximum, maximum);
  if (value < -maximum)
    return wrapAround(value + 2 * maximum, maximum);
  return value;
}

int KMTFCore::encode(bool ownwheel, int sector, int tag) {
  int wheel = ownwheel ? 1 : 0;
  int phi = 0;
  if (sector == 1)
    phi = 1;
  if (sector == -1)
    phi = 2;
  int addr = (wheel << 4) + (phi << 2) + tag;
  return addr;
}

std::pair<bool, uint> KMTFCore::getByCode(const std::vector<l1t::KMTFTrack>& tracks, int mask) {
  for (uint i = 0; i < tracks.size(); ++i) {
    edm::LogInfo("KMTFCore") << "Code=" << tracks[i].hitPattern() << ", track=" << mask;
    if (tracks[i].hitPattern() == mask)
      return std::make_pair(true, i);
  }
  return std::make_pair(false, 0);
}

uint KMTFCore::twosCompToBits(int q) {
  if (q >= 0)
    return q;
  else
    return (~q) + 1;
}

uint KMTFCore::etaStubRank(const l1t::MuonStubRef& stub) {
  if (stub->etaQuality() == 3)
    return 0;
  if (stub->etaQuality() == 0)
    return 0;
  return (stub->etaQuality());
}

void KMTFCore::calculateEta(l1t::KMTFTrack& track) {
  uint pattern = track.hitPattern();
  int wheel = track.stubs()[0]->etaRegion();
  uint awheel = fabs(wheel);
  int sign = 1;
  if (wheel < 0)
    sign = -1;
  uint nstubs = track.stubs().size();
  if (nstubs <= 1) {
    track.setCoarseEta(0);
    return;
  }
  uint mask = 0;

  for (unsigned int i = 0; i < track.stubs().size(); ++i) {
    mask = mask | ((uint(fabs(track.stubs()[i]->etaRegion()) + 1) << (2 * (track.stubs()[i]->depthRegion() - 1))));
  }
  if (verbose_)
    edm::LogInfo("KMTFCore") << "Mask  = " << mask;

  track.setCoarseEta(sign * lutService_->coarseEta(mask));
  if (verbose_)
    edm::LogInfo("KMTFCore") << "Coarse Eta mask=" << mask << " set = " << sign * lutService_->coarseEta(mask);
  track.setFineEta(0);
}

uint KMTFCore::matchAbs(std::map<uint, uint>& info, uint i, uint j) {
  if (info[i] < info[j])
    return i;
  else
    return j;
}

int KMTFCore::ptLUT(int K) {
  int charge = (K >= 0) ? +1 : -1;
  float lsb = 1.25 / float(1 << (BITSCURV - 1));
  float FK = fabs(K);
  int pt = 0;
  if (FK > 8191)
    FK = 8191;
  if (FK < 103)
    FK = 103;
  FK = FK * lsb;
  if (FK == 0) {
    pt = 8191;
  } else {
    float ptF = 1.0 / FK;  //ct
    pt = int(ptF / 0.03125);
  }

  return pt;
}
