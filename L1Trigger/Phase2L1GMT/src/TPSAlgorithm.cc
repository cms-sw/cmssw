#include "L1Trigger/Phase2L1GMT/interface/TPSAlgorithm.h"

using namespace Phase2L1GMT;

TPSAlgorithm::TPSAlgorithm(const edm::ParameterSet& iConfig) : verbose_(iConfig.getParameter<int>("verbose")) {}

std::vector<PreTrackMatchedMuon> TPSAlgorithm::processNonant(const std::vector<ConvertedTTTrack>& convertedTracks,
                                                             const l1t::MuonStubRefVector& stubs) const {
  std::vector<PreTrackMatchedMuon> preMuons;
  for (const auto& track : convertedTracks) {
    PreTrackMatchedMuon mu = processTrack(track, stubs);
    if (mu.valid() && preMuons.size() < 16)
      preMuons.push_back(mu);
  }
  std::vector<PreTrackMatchedMuon> cleanedMuons = clean(preMuons);
  return cleanedMuons;
}

std::vector<PreTrackMatchedMuon> TPSAlgorithm::cleanNeighbor(const std::vector<PreTrackMatchedMuon>& muons,
                                                             const std::vector<PreTrackMatchedMuon>& muonsPrevious,
                                                             const std::vector<PreTrackMatchedMuon>& muonsNext,
                                                             bool equality) const {
  std::vector<PreTrackMatchedMuon> out;

  if (muons.empty())
    return out;

  if (verbose_ == 1) {
    edm::LogInfo("TPSAlgo") << "-----Cleaning Up Muons in the neighbours";
    edm::LogInfo("TPSAlgo") << "Before:";
  }

  for (uint i = 0; i < muons.size(); ++i) {
    if (verbose_ == 1) {
      muons[i].print();
    }
    ap_uint<5> mask = 0x1f;
    for (uint j = 0; j < muonsPrevious.size(); ++j) {
      mask = mask & cleanMuon(muons[i], muonsPrevious[j], equality);
    }
    for (uint j = 0; j < muonsNext.size(); ++j) {
      mask = mask & cleanMuon(muons[i], muonsNext[j], equality);
    }
    if (mask) {
      if (verbose_ == 1)
        edm::LogInfo("TPSAlgo") << "kept";
      out.push_back(muons[i]);
    } else {
      if (verbose_ == 1)
        edm::LogInfo("TPSAlgo") << "discarded";
    }
  }
  return out;
}

std::vector<l1t::TrackerMuon> TPSAlgorithm::convert(const std::vector<PreTrackMatchedMuon>& muons, uint maximum) const {
  std::vector<l1t::TrackerMuon> out;
  for (const auto& mu : muons) {
    if (out.size() == maximum)
      break;
    l1t::TrackerMuon muon(mu.trkPtr(), mu.charge(), mu.pt(), mu.eta(), mu.phi(), mu.z0(), mu.d0(), mu.quality());
    muon.setMuonRef(mu.muonRef());
    for (const auto& stub : mu.stubs())
      muon.addStub(stub);

    uint matches = 0;
    uint mask = mu.matchMask();

    for (uint i = 0; i < 10; i = i + 1) {
      if (mask & (1 << i))
        matches++;
    }
    muon.setNumberOfMatches(matches);
    out.push_back(muon);

    if (verbose_ == 1) {
      edm::LogInfo("TPSAlgo") << "Final Muon:" << std::flush;
      muon.print();
    }
  }
  return out;
}

void TPSAlgorithm::SetQualityBits(std::vector<l1t::TrackerMuon>& muons) const {
  for (auto& mu : muons) {
    // A preliminary suggestion. Need feedback from the menu group
    bool veryloose = mu.numberOfMatches() > 0;
    bool loose = mu.numberOfMatches() > 1;
    bool medium = mu.stubs().size() > 1;
    bool tight = mu.numberOfMatches() > 2;
    int qualbit = 0;
    qualbit = (veryloose << 0) | (loose << 1) | (medium << 2) | (tight << 3);
    mu.setHwQual(qualbit);
  }
}

bool TPSAlgorithm::outputGT(std::vector<l1t::TrackerMuon>& muons) const {
  for (auto& mu : muons) {
    wordtype word1 = 0;
    wordtype word2 = 0;

    int bstart = 0;
    bstart = wordconcat<wordtype>(word1, bstart, mu.hwPt() > 0, 1);
    bstart = wordconcat<wordtype>(word1, bstart, mu.hwPt(), BITSGTPT);
    bstart = wordconcat<wordtype>(word1, bstart, mu.hwPhi(), BITSGTPHI);
    bstart = wordconcat<wordtype>(word1, bstart, mu.hwEta(), BITSGTETA);
    bstart = wordconcat<wordtype>(word1, bstart, mu.hwZ0(), BITSGTZ0);
    wordconcat<wordtype>(word1, bstart, (mu.hwD0() >> 2), BITSGTD0);

    bstart = 0;
    bstart = wordconcat<wordtype>(word2, bstart, mu.hwCharge(), 1);
    bstart = wordconcat<wordtype>(word2, bstart, mu.hwQual(), BITSGTQUAL);
    bstart = wordconcat<wordtype>(word2, bstart, mu.hwIso(), BITSGTISO);
    wordconcat<wordtype>(word2, bstart, mu.hwBeta(), BITSGTBETA);

    std::array<uint64_t, 2> wordout = {{word1, word2}};
    mu.setWord(wordout);
  }
  return true;
}

std::vector<l1t::TrackerMuon> TPSAlgorithm::sort(std::vector<l1t::TrackerMuon>& muons, uint maximum) const {
  if (muons.size() < 2)
    return muons;

  std::sort(muons.begin(), muons.end(), [](l1t::TrackerMuon a, l1t::TrackerMuon b) { return a.hwPt() > b.hwPt(); });
  std::vector<l1t::TrackerMuon> out{muons.begin(), muons.begin() + (maximum < muons.size() ? maximum : muons.size())};

  return out;
}

propagation_t TPSAlgorithm::propagate(const ConvertedTTTrack& track, uint layer) const {
  static const std::array<const ap_uint<BITSPROPCOORD>*, 5> lt_prop1_coord1 = {
      {lt_prop1_coord1_0, lt_prop1_coord1_1, lt_prop1_coord1_2, lt_prop1_coord1_3, lt_prop1_coord1_4}};
  static const std::array<const ap_uint<BITSPROPCOORD>*, 5> lt_prop1_coord2 = {
      {lt_prop1_coord2_0, lt_prop1_coord2_1, lt_prop1_coord2_2, lt_prop1_coord2_3, lt_prop1_coord2_4}};
  static const std::array<const ap_uint<BITSPROPCOORD>*, 5> lt_prop2_coord1 = {
      {lt_prop2_coord1_0, lt_prop2_coord1_1, lt_prop2_coord1_2, lt_prop2_coord1_3, lt_prop2_coord1_4}};
  static const std::array<const ap_uint<BITSPROPCOORD>*, 5> lt_prop2_coord2 = {
      {lt_prop2_coord2_0, lt_prop2_coord2_1, lt_prop2_coord2_2, lt_prop2_coord2_3, lt_prop2_coord2_4}};

  static const std::array<const ap_uint<BITSPROPSIGMACOORD_A>*, 5> lt_res0_coord1 = {
      {lt_res0_coord1_0, lt_res0_coord1_1, lt_res0_coord1_2, lt_res0_coord1_3, lt_res0_coord1_4}};
  static const std::array<const ap_uint<BITSPROPSIGMACOORD_B>*, 5> lt_res1_coord1 = {
      {lt_res1_coord1_0, lt_res1_coord1_1, lt_res1_coord1_2, lt_res1_coord1_3, lt_res1_coord1_4}};
  static const std::array<const ap_uint<BITSPROPSIGMACOORD_A>*, 5> lt_res0_coord2 = {
      {lt_res0_coord2_0, lt_res0_coord2_1, lt_res0_coord2_2, lt_res0_coord2_3, lt_res0_coord2_4}};
  static const std::array<const ap_uint<BITSPROPSIGMACOORD_B>*, 5> lt_res1_coord2 = {
      {lt_res1_coord2_0, lt_res1_coord2_1, lt_res1_coord2_2, lt_res1_coord2_3, lt_res1_coord2_4}};

  static const std::array<const ap_uint<BITSPROPSIGMAETA_A>*, 5> lt_res0_eta1 = {
      {lt_res0_eta1_0, lt_res0_eta1_1, lt_res0_eta1_2, lt_res0_eta1_3, lt_res0_eta1_4}};
  static const std::array<const ap_uint<BITSPROPSIGMAETA_A>*, 5> lt_res1_eta1 = {
      {lt_res1_eta_0, lt_res1_eta_1, lt_res1_eta_2, lt_res1_eta_3, lt_res1_eta_4}};

  static const std::array<const ap_uint<BITSPROPSIGMAETA_A>*, 5> lt_res0_eta2 = {
      {lt_res0_eta2_0, lt_res0_eta2_1, lt_res0_eta2_2, lt_res0_eta2_3, lt_res0_eta2_4}};

  static const uint barrellimit[5] = {barrelLimit0_, barrelLimit1_, barrelLimit2_, barrelLimit3_, barrelLimit4_};

  ap_uint<BITSPROPCOORD> prop1_coord1 = 0;
  ap_uint<BITSPROPCOORD> prop1_coord2 = 0;
  ap_uint<BITSPROPCOORD> prop2_coord1 = 0;
  ap_uint<BITSPROPCOORD> prop2_coord2 = 0;
  ap_uint<BITSPROPSIGMACOORD_A> res0_coord1 = 0;
  ap_uint<BITSPROPSIGMACOORD_B> res1_coord1 = 0;
  ap_uint<BITSPROPSIGMACOORD_A> res0_coord2 = 0;
  ap_uint<BITSPROPSIGMACOORD_B> res1_coord2 = 0;
  ap_uint<BITSPROPSIGMAETA_A> res0_eta1 = 0;
  ap_uint<BITSPROPSIGMAETA_B> res1_eta = 0;
  ap_uint<BITSPROPSIGMAETA_A> res0_eta2 = 0;
  ap_uint<1> is_barrel = 0;

  uint reducedAbsEta = track.abseta() / 8;

  //Propagate to layers
  assert(layer < 5);
  prop1_coord1 = lt_prop1_coord1[layer][reducedAbsEta];
  prop1_coord2 = lt_prop1_coord2[layer][reducedAbsEta];
  prop2_coord1 = lt_prop2_coord1[layer][reducedAbsEta];
  prop2_coord2 = lt_prop2_coord2[layer][reducedAbsEta];
  res0_coord1 = lt_res0_coord1[layer][reducedAbsEta];
  res1_coord1 = lt_res1_coord1[layer][reducedAbsEta];
  res0_coord2 = lt_res0_coord2[layer][reducedAbsEta];
  res1_coord2 = lt_res1_coord2[layer][reducedAbsEta];
  res0_eta1 = lt_res0_eta1[layer][reducedAbsEta];
  res1_eta = lt_res1_eta1[layer][reducedAbsEta];
  res0_eta2 = lt_res0_eta2[layer][reducedAbsEta];
  is_barrel = reducedAbsEta < barrellimit[layer] ? 1 : 0;

  //try inflating res0's
  //res0_coord1 = 2 * res0_coord1;
  //res0_coord2 = 2 * res0_coord2;

  propagation_t out;
  ap_int<BITSTTCURV> curvature = track.curvature();
  ap_int<BITSPHI> phi = track.phi();

  //should be enough bits to hold all of c1k + d1kabsK, so if each are the same number of bits (they should be), that is 1 more bit (12 bits in this case)
  ap_uint<BITSPROPCOORD + BITSTTCURV - 12> absDphiOverflow;
  ap_int<BITSPROP + 1> dphi;

  ap_uint<BITSTTCURV - 1> absK = 0;
  ap_uint<1> negativeCurv;
  if (track.curvature() < 0) {
    absK = ap_uint<BITSTTCURV - 1>(-track.curvature());
    negativeCurv = 1;
  } else {
    absK = ap_uint<BITSTTCURV - 1>(track.curvature());
    negativeCurv = 0;
  }

  ap_uint<BITSPROPCOORD + BITSTTCURV - 1> c1kFull = prop1_coord1 * absK;
  ap_uint<BITSPROPCOORD + BITSTTCURV - 13> c1k = (c1kFull) >> 12;  // 1024;
  //ap_int<BITSPHI> coord1 = phi - c1k;

  ap_uint<BITSPROPCOORD + 2 * BITSTTCURV - 2> d1kabsKFull = prop2_coord1 * absK * absK;
  ap_uint<BITSPROPCOORD + 2 * BITSTTCURV - 28> d1kabsK = (d1kabsKFull) >> 26;  // 16777216;

  absDphiOverflow = c1k + d1kabsK;
  if (absDphiOverflow > PROPMAX)
    dphi = PROPMAX;
  else
    dphi = absDphiOverflow;

  if (negativeCurv == 1)
    dphi = -dphi;

  //ap_int<BITSPHI> coord1 = phi - dphi;
  out.coord1 = (phi - dphi) / PHIDIVIDER;

  ap_uint<BITSPROPCOORD + BITSTTCURV - 1> c2kFull = prop1_coord2 * absK;
  ap_uint<BITSPROPCOORD + BITSTTCURV - 13> c2k = (c2kFull) >> 12;  // 1024;

  ap_uint<BITSPROPCOORD + 2 * BITSTTCURV - 2> d2kabsKFull = prop2_coord2 * absK * absK;
  ap_uint<BITSPROPCOORD + 2 * BITSTTCURV - 28> d2kabsK = (d2kabsKFull) >> 26;  // 16777216;

  absDphiOverflow = c2k + d2kabsK;
  if (absDphiOverflow > PROPMAX)
    dphi = PROPMAX;
  else
    dphi = absDphiOverflow;

  if (negativeCurv == 1)
    dphi = -dphi;

  if (is_barrel)
    out.coord2 = -dphi / PHIDIVIDER;
  else
    out.coord2 = (phi - dphi) / PHIDIVIDER;

  ap_int<BITSETA> eta = track.eta();
  out.eta = eta / ETADIVIDER;

  ap_uint<2 * BITSTTCURV - 2> curvature2All = curvature * curvature;
  ap_uint<BITSTTCURV2> curvature2 = curvature2All / 2;
  /*
  /////New propagation for sigma
  ap_uint<BITSTTCURV - 1> absK = 0;
  if (track.curvature() < 0)
    absK = ap_uint<BITSTTCURV - 1>(-track.curvature());
  else
    absK = ap_uint<BITSTTCURV - 1>(track.curvature());
  */
  //bound the resolution propagation
  //if (absK > 6000)
  //  absK = 6000;

  ap_uint<BITSPROPSIGMAETA_B + BITSTTCURV2> resetak = (res1_eta * curvature2) >> 23;
  ap_ufixed<BITSSIGMAETA, BITSSIGMAETA, AP_TRN_ZERO, AP_SAT_SYM> sigma_eta1 = res0_eta1 + resetak;
  out.sigma_eta1 = ap_uint<BITSSIGMAETA>(sigma_eta1);
  ap_ufixed<BITSSIGMAETA, BITSSIGMAETA, AP_TRN_ZERO, AP_SAT_SYM> sigma_eta2 = res0_eta2 + resetak;
  out.sigma_eta2 = ap_uint<BITSSIGMAETA>(sigma_eta2);

  ap_uint<BITSPROPSIGMACOORD_B + BITSTTCURV - 1> s1kFull = res1_coord1 * absK;
  ap_uint<BITSPROPSIGMACOORD_B + BITSTTCURV - 1 - 10> s1k = res0_coord1 + (s1kFull >> 10);
  if (s1k >= (1 << (BITSSIGMACOORD + BITSPHI - BITSSTUBCOORD)))
    out.sigma_coord1 = ~ap_uint<BITSSIGMACOORD>(0);
  else if (s1k < PHIDIVIDER)
    out.sigma_coord1 = 1;
  else
    out.sigma_coord1 = s1k / PHIDIVIDER;

  ap_uint<BITSPROPSIGMACOORD_B + BITSTTCURV - 1> s2kFull = res1_coord2 * absK;
  ap_uint<BITSPROPSIGMACOORD_B + BITSTTCURV - 1 - 10> s2k = res0_coord2 + (s2kFull >> 10);
  if (s2k >= (1 << (BITSSIGMACOORD + BITSPHI - BITSSTUBCOORD)))
    out.sigma_coord2 = ~ap_uint<BITSSIGMACOORD>(0);
  else if (s2k < PHIDIVIDER)
    out.sigma_coord2 = 1;
  else
    out.sigma_coord2 = s2k / PHIDIVIDER;

  out.valid = 1;
  out.is_barrel = is_barrel;

  if (verbose_ == 1) {
    edm::LogInfo("TPSAlgo") << "Propagating to layer " << int(layer) << ":is barrel=" << out.is_barrel.to_int()
                            << "  coords=" << out.coord1.to_int() << "+-" << out.sigma_coord1.to_int() << " , "
                            << out.coord2.to_int() << " +-" << out.sigma_coord2.to_int()
                            << " etas = " << out.eta.to_int() << " +- " << out.sigma_eta1.to_int() << " +-"
                            << out.sigma_eta2.to_int();

    edm::LogInfo("TPSAlgo") << "----- breakout of sigma 1 : constant=" << res0_coord1.to_int()
                            << " slope=" << res1_coord1.to_int() << " before division=" << s1k.to_int();

    edm::LogInfo("TPSAlgo") << "----- breakout of sigma 2 : constant=" << res0_coord2.to_int()
                            << " slope=" << res1_coord2.to_int() << " before division=" << s2k.to_int();
  }
  return out;
}

ap_uint<BITSSIGMAETA + 1> TPSAlgorithm::deltaEta(const ap_int<BITSSTUBETA>& eta1,
                                                 const ap_int<BITSSTUBETA>& eta2) const {
  ap_fixed<BITSSIGMAETA + 2, BITSSIGMAETA + 2, AP_TRN_ZERO, AP_SAT_SYM> dEta = eta1 - eta2;
  if (dEta < 0)
    return ap_uint<BITSSIGMAETA + 1>(-dEta);
  else
    return ap_uint<BITSSIGMAETA + 1>(dEta);
}

ap_uint<BITSSIGMACOORD + 1> TPSAlgorithm::deltaCoord(const ap_int<BITSSTUBCOORD>& phi1,
                                                     const ap_int<BITSSTUBCOORD>& phi2) const {
  ap_int<BITSSTUBCOORD> dPhiRoll = phi1 - phi2;
  ap_ufixed<BITSSIGMACOORD + 1, BITSSIGMACOORD + 1, AP_TRN_ZERO, AP_SAT_SYM> dPhi;
  if (dPhiRoll < 0)
    dPhi = ap_ufixed<BITSSIGMACOORD + 1, BITSSIGMACOORD + 1, AP_TRN_ZERO, AP_SAT_SYM>(-dPhiRoll);
  else
    dPhi = ap_ufixed<BITSSIGMACOORD + 1, BITSSIGMACOORD + 1, AP_TRN_ZERO, AP_SAT_SYM>(dPhiRoll);

  return ap_uint<BITSSIGMACOORD + 1>(dPhi);
}

match_t TPSAlgorithm::match(const propagation_t prop, const l1t::MuonStubRef& stub, uint trackID) const {
  if (verbose_ == 1) {
    edm::LogInfo("TPSAlgo") << "Matching to coord1=" << stub->coord1() << " coord2=" << stub->coord2()
                            << " eta1=" << stub->eta1() << " eta2=" << stub->eta2();

    stub->print();
  }
  //Matching of Coord1
  ap_uint<1> coord1Matched;
  ap_uint<BITSSIGMACOORD + 1> deltaCoord1 = deltaCoord(prop.coord1, stub->coord1());
  if (deltaCoord1 <= prop.sigma_coord1 && (stub->quality() & 0x1)) {
    coord1Matched = 1;
  } else {
    coord1Matched = 0;
  }
  if (verbose_ == 1)
    edm::LogInfo("TPSAlgo") << "Coord1 matched=" << coord1Matched.to_int() << " delta=" << deltaCoord1.to_int()
                            << " res=" << prop.sigma_coord1.to_int();

  //Matching of Coord2
  ap_uint<1> coord2Matched;
  ap_uint<BITSSIGMACOORD + 1> deltaCoord2 = deltaCoord(prop.coord2, stub->coord2());
  if (deltaCoord2 <= prop.sigma_coord2 && (stub->quality() & 0x2)) {
    coord2Matched = 1;
  } else {
    coord2Matched = 0;
  }
  if (verbose_ == 1)
    edm::LogInfo("TPSAlgo") << "Coord2 matched=" << coord2Matched.to_int() << " delta=" << deltaCoord2.to_int()
                            << " res=" << prop.sigma_coord2.to_int();

  //Matching of Eta1

  ap_uint<1> eta1Matched;

  //if we have really bad quality[Barrel no eta]
  //increase the resolution
  ap_ufixed<BITSSIGMAETA, BITSSIGMAETA, AP_TRN_ZERO, AP_SAT_SYM> prop_sigma_eta1;
  if (stub->etaQuality() == 0)
    prop_sigma_eta1 = prop.sigma_eta1 + 6;
  else
    prop_sigma_eta1 = prop.sigma_eta1;

  ap_uint<BITSSIGMAETA + 1> deltaEta1 = deltaEta(prop.eta, stub->eta1());
  if (deltaEta1 <= prop_sigma_eta1 && (stub->etaQuality() == 0 || (stub->etaQuality() & 0x1)))
    eta1Matched = 1;
  else
    eta1Matched = 0;

  if (verbose_ == 1)
    edm::LogInfo("TPSAlgo") << "eta1 matched=" << eta1Matched.to_int() << " delta=" << deltaEta1.to_int()
                            << " res=" << prop_sigma_eta1.to_int();

  //Matching of Eta2

  ap_uint<1> eta2Matched;

  ap_uint<BITSSIGMAETA + 1> deltaEta2 = deltaEta(prop.eta, stub->eta2());
  if (deltaEta2 <= prop.sigma_eta2 && (stub->etaQuality() & 0x2))
    eta2Matched = 1;
  else
    eta2Matched = 0;
  match_t out;
  out.id = trackID;

  if (verbose_ == 1)
    edm::LogInfo("TPSAlgo") << "eta2 matched=" << eta2Matched.to_int() << " delta=" << deltaEta2.to_int()
                            << " res=" << prop.sigma_eta2.to_int();

  //Note I divided by 4 because of the new coordinate. Make it automatic

  //if barrel, coord1 has to always be matched, coord2 maybe and eta1 is needed if etaQ=0 or then the one that depends on eta quality
  if (prop.is_barrel) {
    out.valid = (coord1Matched == 1 && (eta1Matched == 1 || eta2Matched == 1)) ? 1 : 0;
    if (out.valid == 0) {
      out.quality = 0;
    } else {
      out.quality = 32 - deltaCoord1 / 4;
      if (coord2Matched == 1) {
        out.quality += 32 - deltaCoord2 / 4;
        out.valid = 3;
      }
    }
  }
  //if endcap each coordinate is independent except the case where phiQuality=1 and etaQuality==3
  else {
    bool match1 = (coord1Matched == 1 && eta1Matched == 1);
    bool match2 = (coord2Matched == 1 && eta2Matched == 1);
    bool match3 =
        (coord1Matched == 1 && (eta1Matched || eta2Matched) && stub->etaQuality() == 3 && stub->quality() == 1);
    out.valid = (match1 || match2 || match3) ? 1 : 0;
    if (out.valid == 0)
      out.quality = 0;
    else {
      out.quality = 0;
      if (match1 || match3)
        out.quality += 32 - deltaCoord1 / 4;
      if (match2) {
        out.quality += 32 - deltaCoord2 / 4;
        if (match1 || match3)
          out.valid = 3;
      }
    }
  }
  if (verbose_ == 1)
    edm::LogInfo("TPSAlgo") << "GlobalMatchQuality = " << out.quality.to_int();
  out.stubRef = stub;
  return out;
}

match_t TPSAlgorithm::propagateAndMatch(const ConvertedTTTrack& track,
                                        const l1t::MuonStubRef& stub,
                                        uint trackID) const {
  propagation_t prop = propagate(track, stub->tfLayer());
  return match(prop, stub, trackID);
}

match_t TPSAlgorithm::getBest(const std::vector<match_t>& matches) const {
  match_t best = matches[0];
  for (const auto& m : matches) {
    if (m.quality > best.quality)
      best = m;
  }

  return best;
}

void TPSAlgorithm::matchingInfos(const std::vector<match_t>& matchInfo,
                                 PreTrackMatchedMuon& muon,
                                 ap_uint<BITSMATCHQUALITY>& quality) const {
  if (!matchInfo.empty()) {
    match_t b = getBest(matchInfo);
    if (b.valid != 0) {
      muon.addStub(b.stubRef, b.valid);
      if (b.isGlobal)
        muon.addMuonRef(b.muRef);
      quality += b.quality;
    }
  }
}

PreTrackMatchedMuon TPSAlgorithm::processTrack(const ConvertedTTTrack& track,
                                               const l1t::MuonStubRefVector& stubs) const {
  std::array<std::vector<match_t>, 6> matchInfos;

  if (verbose_ == 1 && !stubs.empty()) {
    edm::LogInfo("TPSAlgo") << "-----------processing new track----------";
    track.print();
  }
  for (const auto& stub : stubs) {
    match_t m = propagateAndMatch(track, stub, 0);
    if (m.valid != 0 && stub->tfLayer() < 6) {
      matchInfos[stub->tfLayer()].push_back(m);
    }
  }

  ap_ufixed<6, 6, AP_TRN_ZERO, AP_SAT_SYM> ptPenalty = ap_ufixed<6, 6, AP_TRN_ZERO, AP_SAT_SYM>(track.pt() / 32);

  ap_uint<BITSMATCHQUALITY> quality = 0;
  PreTrackMatchedMuon muon(track.charge(), track.pt(), track.eta(), track.phi(), track.z0(), track.d0());

  for (auto&& m : matchInfos)
    matchingInfos(m, muon, quality);

  muon.setOfflineQuantities(track.offline_pt(), track.offline_eta(), track.offline_phi());
  muon.setTrkPtr(track.trkPtr());

  ap_uint<8> etaAddr = muon.eta() < 0 ? ap_uint<8>(-muon.eta() / 256) : ap_uint<8>((muon.eta()) / 256);
  ap_uint<8> ptAddr = muon.pt() > 4095 ? ap_uint<8>(15) : ap_uint<8>(muon.pt() / 256);
  ap_uint<8> addr = ptAddr | (etaAddr << 4);
  ap_uint<8> qualityCut = lt_tpsID[addr];

  if (!muon.stubs().empty()) {  //change the ID for now
    muon.setValid(true);
    muon.setQuality(quality + ptPenalty);
  } else {
    muon.setValid(false);
    muon.setQuality(0);
    muon.resetGlobal();
  }
  if (verbose_ == 1)
    muon.print();

  if (verbose_ == 1 && !stubs.empty()) {  //patterns for HLS

    edm::LogInfo("TPSAlgo") << "TPS " << track.trkPtr()->phiSector() << std::flush;
    track.printWord();

    for (uint i = 0; i < 16; ++i) {
      if (stubs.size() > i) {
        edm::LogInfo("TPSAlgo") << "remember to implement printout of muon";
      } else {
        edm::LogInfo("TPSAlgo") << std::hex << std::setw(8) << 0 << std::flush;
        edm::LogInfo("TPSAlgo") << std::hex << std::setw(16) << 0x1ff000000000000 << std::flush;
        edm::LogInfo("TPSAlgo") << std::hex << std::setw(16) << 0x1ff000000000000 << std::flush;
        edm::LogInfo("TPSAlgo") << std::hex << std::setw(16) << 0x1ff000000000000 << std::flush;
        edm::LogInfo("TPSAlgo") << std::hex << std::setw(16) << 0x1ff000000000000 << std::flush;
        edm::LogInfo("TPSAlgo") << std::hex << std::setw(16) << 0x1ff000000000000 << std::flush;
      }
    }
    muon.printWord();
    edm::LogInfo("TPSAlgo") << std::endl;
  }

  return muon;
}

ap_uint<5> TPSAlgorithm::cleanMuon(const PreTrackMatchedMuon& mu, const PreTrackMatchedMuon& other, bool eq) const {
  ap_uint<5> valid = 0;
  ap_uint<5> overlap = 0;
  constexpr int bittest = 0xfff;  // 4095, corresponding to 11bits
  if (mu.stubID0() != bittest) {
    valid = valid | 0x1;
    if (mu.stubID0() == other.stubID0())
      overlap = overlap | 0x1;
  }
  if (mu.stubID1() != bittest) {
    valid = valid | 0x2;
    if (mu.stubID1() == other.stubID1())
      overlap = overlap | 0x2;
  }
  if (mu.stubID2() != bittest) {
    valid = valid | 0x4;
    if (mu.stubID2() == other.stubID2())
      overlap = overlap | 0x4;
  }
  if (mu.stubID3() != bittest) {
    valid = valid | 0x8;
    if (mu.stubID3() == other.stubID3())
      overlap = overlap | 0x8;
  }
  if (mu.stubID4() != bittest) {
    valid = valid | 0x10;
    if (mu.stubID4() == other.stubID4())
      overlap = overlap | 0x10;
  }

  if (((mu.quality() < other.quality()) && (!eq)) || ((mu.quality() <= other.quality()) && (eq)))
    return valid & (~overlap);
  else
    return valid;
}

std::vector<PreTrackMatchedMuon> TPSAlgorithm::clean(const std::vector<PreTrackMatchedMuon>& muons) const {
  std::vector<PreTrackMatchedMuon> out;
  if (muons.empty())
    return out;
  if (verbose_ == 1) {
    edm::LogInfo("TPSAlgo") << "-----Cleaning Up Muons in the same Nonant";
    edm::LogInfo("TPSAlgo") << "Before:";
  }
  for (uint i = 0; i < muons.size(); ++i) {
    if (verbose_ == 1)
      muons[i].print();

    ap_uint<5> mask = 0x1f;
    for (uint j = 0; j < muons.size(); ++j) {
      if (i == j)
        continue;
      mask = mask & cleanMuon(muons[i], muons[j], false);
    }
    if (mask) {
      if (verbose_ == 1)
        edm::LogInfo("TPSAlgo") << "kept";
      out.push_back(muons[i]);
    } else {
      if (verbose_ == 1)
        edm::LogInfo("TPSAlgo") << "discarded";
    }
  }
  return out;
}
