#include <memory>

#include "L1Trigger/CSCTriggerPrimitives/interface/CSCGEMMotherboard.h"

CSCGEMMotherboard::CSCGEMMotherboard(unsigned endcap,
                                     unsigned station,
                                     unsigned sector,
                                     unsigned subsector,
                                     unsigned chamber,
                                     const edm::ParameterSet& conf)
    : CSCUpgradeMotherboard(endcap, station, sector, subsector, chamber, conf),
      maxDeltaBXPad_(tmbParams_.getParameter<int>("maxDeltaBXPad")),
      maxDeltaBXCoPad_(tmbParams_.getParameter<int>("maxDeltaBXCoPad")),
      promoteALCTGEMpattern_(tmbParams_.getParameter<bool>("promoteALCTGEMpattern")),
      promoteALCTGEMquality_(tmbParams_.getParameter<bool>("promoteALCTGEMquality")) {
  // super chamber has layer=0!
  gemId = GEMDetId(theRegion, 1, theStation, 0, theChamber, 0).rawId();

  const edm::ParameterSet coPadParams(station == 1 ? conf.getParameter<edm::ParameterSet>("copadParamGE11")
                                                   : conf.getParameter<edm::ParameterSet>("copadParamGE21"));
  coPadProcessor = std::make_unique<GEMCoPadProcessor>(theRegion, theStation, theChamber, coPadParams);

  maxDeltaPadL1_ = (theParity ? tmbParams_.getParameter<int>("maxDeltaPadL1Even")
                              : tmbParams_.getParameter<int>("maxDeltaPadL1Odd"));
  maxDeltaPadL2_ = (theParity ? tmbParams_.getParameter<int>("maxDeltaPadL2Even")
                              : tmbParams_.getParameter<int>("maxDeltaPadL2Odd"));
}

CSCGEMMotherboard::CSCGEMMotherboard() : CSCUpgradeMotherboard() {}

CSCGEMMotherboard::~CSCGEMMotherboard() {}

void CSCGEMMotherboard::clear() {
  CSCUpgradeMotherboard::clear();
  gemCoPadV.clear();
  coPadProcessor->clear();
  pads_.clear();
  coPads_.clear();
}

void CSCGEMMotherboard::processGEMClusters(const GEMPadDigiClusterCollection* gemClusters) {
  std::unique_ptr<GEMPadDigiCollection> gemPads(new GEMPadDigiCollection());
  coPadProcessor->declusterize(gemClusters, *gemPads);

  gemCoPadV = coPadProcessor->run(gemPads.get());

  processGEMPads(gemPads.get());
  processGEMCoPads();
}

void CSCGEMMotherboard::processGEMPads(const GEMPadDigiCollection* gemPads) {
  pads_.clear();
  const auto& superChamber(gem_g->superChamber(gemId));
  for (const auto& ch : superChamber->chambers()) {
    for (const auto& roll : ch->etaPartitions()) {
      GEMDetId roll_id(roll->id());
      auto pads_in_det = gemPads->get(roll_id);
      for (auto pad = pads_in_det.first; pad != pads_in_det.second; ++pad) {
        // ignore 16-partition GE2/1 pads
        if (roll->isGE21() and pad->nPartitions() == GEMPadDigi::GE21SplitStrip)
          continue;

        // ignore invalid pads
        if (!pad->isValid())
          continue;

        const int bx_shifted(CSCConstants::LCT_CENTRAL_BX + pad->bx());
        // consider matches with BX difference +1/0/-1
        for (int bx = bx_shifted - maxDeltaBXPad_; bx <= bx_shifted + maxDeltaBXPad_; ++bx) {
          pads_[bx].emplace_back(roll_id.rawId(), *pad);
        }
      }
    }
  }
}

void CSCGEMMotherboard::processGEMCoPads() {
  coPads_.clear();
  for (const auto& copad : gemCoPadV) {
    GEMDetId detId(theRegion, 1, theStation, 0, theChamber, 0);

    // ignore 16-partition GE2/1 pads
    if (detId.isGE21() and copad.first().nPartitions() == GEMPadDigi::GE21SplitStrip)
      continue;

    // only consider matches with same BX
    coPads_[CSCConstants::LCT_CENTRAL_BX + copad.bx(1)].emplace_back(detId.rawId(), copad);
  }
}

CSCCorrelatedLCTDigi CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& alct,
                                                         const GEMCoPadDigi& gem,
                                                         int trknmb) const {
  return constructLCTsGEM(alct, CSCCLCTDigi(), GEMPadDigi(), gem, trknmb);
}

CSCCorrelatedLCTDigi CSCGEMMotherboard::constructLCTsGEM(const CSCCLCTDigi& clct,
                                                         const GEMCoPadDigi& gem,
                                                         int trknmb) const {
  return constructLCTsGEM(CSCALCTDigi(), clct, GEMPadDigi(), gem, trknmb);
}

CSCCorrelatedLCTDigi CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& alct,
                                                         const CSCCLCTDigi& clct,
                                                         const GEMCoPadDigi& gem,
                                                         int trknmb) const {
  return constructLCTsGEM(alct, clct, GEMPadDigi(), gem, trknmb);
}

CSCCorrelatedLCTDigi CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& alct,
                                                         const CSCCLCTDigi& clct,
                                                         const GEMPadDigi& gem,
                                                         int trknmb) const {
  return constructLCTsGEM(alct, clct, gem, GEMCoPadDigi(), trknmb);
}

CSCCorrelatedLCTDigi CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& alct,
                                                         const CSCCLCTDigi& clct,
                                                         const GEMPadDigi& gem1,
                                                         const GEMCoPadDigi& gem2,
                                                         int trknmb) const {
  int pattern = 0, quality = 0, bx = 0, keyStrip = 0, keyWG = 0, bend = 0, valid = 0;

  // make a new LCT
  CSCCorrelatedLCTDigi thisLCT;
  if (!alct.isValid() and !clct.isValid()) {
    edm::LogError("CSCGEMCMotherboard") << "Warning!!! neither ALCT nor CLCT valid, return invalid LCT";
    return thisLCT;
  }

  // Determine the case and assign properties depending on the LCT dataformat (old/new)
  if (alct.isValid() and clct.isValid() and gem1.isValid() and not gem2.isValid()) {
    pattern = encodePattern(clct.getPattern());
    if (runCCLUT_) {
      quality = static_cast<unsigned int>(findQualityGEMv2(alct, clct, 1));
    } else {
      quality = static_cast<unsigned int>(findQualityGEMv1(alct, clct, 1));
    }
    bx = alct.getBX();
    keyStrip = clct.getKeyStrip();
    keyWG = alct.getKeyWG();
    bend = clct.getBend();
    thisLCT.setALCT(getBXShiftedALCT(alct));
    thisLCT.setCLCT(getBXShiftedCLCT(clct));
    thisLCT.setGEM1(gem1);
    thisLCT.setType(CSCCorrelatedLCTDigi::ALCTCLCTGEM);
    valid = doesWiregroupCrossStrip(keyWG, keyStrip) ? 1 : 0;
    if (runCCLUT_) {
      thisLCT.setRun3(true);
      // 4-bit slope value derived with the CCLUT algorithm
      thisLCT.setSlope(clct.getSlope());
      thisLCT.setQuartStrip(clct.getQuartStrip());
      thisLCT.setEightStrip(clct.getEightStrip());
      thisLCT.setRun3Pattern(clct.getRun3Pattern());
    }
  } else if (alct.isValid() and clct.isValid() and not gem1.isValid() and gem2.isValid()) {
    pattern = encodePattern(clct.getPattern());
    if (runCCLUT_) {
      quality = static_cast<unsigned int>(findQualityGEMv2(alct, clct, 2));
    } else {
      quality = static_cast<unsigned int>(findQualityGEMv1(alct, clct, 2));
    }
    bx = alct.getBX();
    keyStrip = clct.getKeyStrip();
    keyWG = alct.getKeyWG();
    bend = clct.getBend();
    thisLCT.setALCT(getBXShiftedALCT(alct));
    thisLCT.setCLCT(getBXShiftedCLCT(clct));
    thisLCT.setGEM1(gem2.first());
    thisLCT.setGEM2(gem2.second());
    thisLCT.setType(CSCCorrelatedLCTDigi::ALCTCLCT2GEM);
    valid = doesWiregroupCrossStrip(keyWG, keyStrip) ? 1 : 0;
    if (runCCLUT_) {
      thisLCT.setRun3(true);
      // 4-bit slope value derived with the CCLUT algorithm
      thisLCT.setSlope(clct.getSlope());
      thisLCT.setQuartStrip(clct.getQuartStrip());
      thisLCT.setEightStrip(clct.getEightStrip());
      thisLCT.setRun3Pattern(clct.getRun3Pattern());
    }
  } else if (alct.isValid() and gem2.isValid() and not clct.isValid()) {
    //in ME11
    //ME1b: keyWG >15,
    //ME1a and ME1b overlap:  10<=keyWG<=15
    //ME1a: keyWG < 10
    //in overlap region, firstly try a match in ME1b

    auto p(getCSCPart(-1));  //use -1 as fake halfstrip, it returns ME11 if station==1 && (ring==1 or ring==4)
    if (p == CSCPart::ME11) {
      if (alct.getKeyWG() >= 10)
        p = CSCPart::ME1B;
      else
        p = CSCPart::ME1A;
    }

    // min pad number is always 0
    // max pad number is 191 or 383, depending on the station
    assert(gem2.pad(1) >= 0);
    assert(gem2.pad(2) >= 0);
    assert(gem2.pad(1) < maxPads());
    assert(gem2.pad(2) < maxPads());

    const auto& mymap1 = getLUT()->get_gem_pad_to_csc_hs(theParity, p);
    // GEM pad number is counting from 1
    // keyStrip from mymap:  for ME1b 0-127 and for ME1a 0-95
    // keyStrip for CLCT: for ME1b 0-127 and for ME1a 128-223
    keyStrip = mymap1.at(gem2.pad(2));
    if (p == CSCPart::ME1A and keyStrip <= CSCConstants::MAX_HALF_STRIP_ME1B) {
      keyStrip += CSCConstants::MAX_HALF_STRIP_ME1B + 1;
    }
    keyWG = alct.getKeyWG();

    if ((not doesWiregroupCrossStrip(keyWG, keyStrip)) and p == CSCPart::ME1B and keyWG <= 15) {
      //try ME1A as strip and WG do not cross
      p = CSCPart::ME1A;
      const auto& mymap2 = getLUT()->get_gem_pad_to_csc_hs(theParity, p);
      keyStrip = mymap2.at(gem2.pad(2)) + CSCConstants::MAX_HALF_STRIP_ME1B + 1;
    }

    pattern = promoteALCTGEMpattern_ ? 10 : 0;
    quality = promoteALCTGEMquality_ ? 15 : 11;
    bx = alct.getBX();
    thisLCT.setALCT(getBXShiftedALCT(alct));
    thisLCT.setGEM1(gem2.first());
    thisLCT.setGEM2(gem2.second());
    thisLCT.setType(CSCCorrelatedLCTDigi::ALCT2GEM);
    valid = true;
  } else if (clct.isValid() and gem2.isValid() and not alct.isValid()) {
    // min roll number is always 1
    // max roll number is 8 or 16, depending on the station
    assert(gem2.roll() >= GEMDetId::minRollId);
    assert(gem2.roll() <= maxRolls());

    const auto& mymap2 = getLUT()->get_gem_roll_to_csc_wg(theParity);
    pattern = encodePattern(clct.getPattern());
    quality = promoteCLCTGEMquality_ ? 15 : 11;
    bx = gem2.bx(1) + CSCConstants::LCT_CENTRAL_BX;
    keyStrip = clct.getKeyStrip();
    // choose the corresponding wire-group in the middle of the partition
    keyWG = mymap2.at(gem2.roll() - 1);
    bend = clct.getBend();
    thisLCT.setCLCT(clct);
    thisLCT.setGEM1(gem2.first());
    thisLCT.setGEM2(gem2.second());
    thisLCT.setType(CSCCorrelatedLCTDigi::CLCT2GEM);
    valid = true;
    if (runCCLUT_) {
      thisLCT.setRun3(true);
      // 4-bit slope value derived with the CCLUT algorithm
      thisLCT.setSlope(clct.getSlope());
      thisLCT.setQuartStrip(clct.getQuartStrip());
      thisLCT.setEightStrip(clct.getEightStrip());
      thisLCT.setRun3Pattern(clct.getRun3Pattern());
    }
  }

  if (valid == 0)
    LogTrace("CSCGEMCMotherboard") << "Warning!!! wiregroup and strip pair are not crossing each other"
                                   << " detid " << cscId_ << " with wiregroup " << keyWG << "keyStrip " << keyStrip
                                   << " \n";

  // fill the rest of the properties
  thisLCT.setTrknmb(trknmb);
  thisLCT.setValid(valid);
  thisLCT.setQuality(quality);
  thisLCT.setWireGroup(keyWG);
  thisLCT.setStrip(keyStrip);
  thisLCT.setPattern(pattern);
  thisLCT.setBend(bend);
  thisLCT.setBX(bx);
  thisLCT.setMPCLink(0);
  thisLCT.setBX0(0);
  // Not used in Run-2. Will not be assigned in Run-3
  thisLCT.setSyncErr(0);
  thisLCT.setCSCID(theTrigChamber);
  // in Run-3 we plan to denote the presence of exotic signatures in the chamber
  if (useHighMultiplicityBits_)
    thisLCT.setHMT(highMultiplicityBits_);

  // future work: add a section that produces LCTs according
  // to the new LCT dataformat (not yet defined)

  // return new LCT
  return thisLCT;
}

bool CSCGEMMotherboard::isPadInOverlap(int roll) const {
  // this only works for ME1A!
  const auto& mymap = (getLUT()->get_csc_wg_to_gem_roll(theParity));
  for (unsigned i = 0; i < mymap.size(); i++) {
    // overlap region are WGs 10-15
    if ((i < 10) or (i > 15))
      continue;
    if ((mymap.at(i).first <= roll) and (roll <= mymap.at(i).second))
      return true;
  }
  return false;
}

bool CSCGEMMotherboard::isGEMDetId(unsigned int p) const {
  return (DetId(p).subdetId() == MuonSubdetId::GEM and DetId(p).det() == DetId::Muon);
}

int CSCGEMMotherboard::getBX(const GEMPadDigi& p) const { return p.bx(); }

int CSCGEMMotherboard::getBX(const GEMCoPadDigi& p) const { return p.bx(1); }

int CSCGEMMotherboard::getRoll(const GEMPadDigiId& p) const { return GEMDetId(p.first).roll(); }

int CSCGEMMotherboard::getRoll(const GEMCoPadDigiId& p) const { return p.second.roll(); }

std::pair<int, int> CSCGEMMotherboard::getRolls(const CSCALCTDigi& alct) const {
  const auto& mymap(getLUT()->get_csc_wg_to_gem_roll(theParity));
  return mymap.at(alct.getKeyWG());
}

float CSCGEMMotherboard::getPad(const GEMPadDigi& p) const { return p.pad(); }

float CSCGEMMotherboard::getPad(const GEMCoPadDigi& p) const {
  // average pad number for a GEMCoPad
  return 0.5 * (p.pad(1) + p.pad(2));
}

float CSCGEMMotherboard::getPad(const CSCCLCTDigi& clct, enum CSCPart part) const {
  const auto& mymap = (getLUT()->get_csc_hs_to_gem_pad(theParity, part));
  int keyStrip = clct.getKeyStrip();
  //ME1A part, convert halfstrip from 128-223 to 0-95
  if (part == CSCPart::ME1A and keyStrip > CSCConstants::MAX_HALF_STRIP_ME1B)
    keyStrip = keyStrip - CSCConstants::MAX_HALF_STRIP_ME1B - 1;
  return 0.5 * (mymap.at(keyStrip).first + mymap.at(keyStrip).second);
}

int CSCGEMMotherboard::maxPads() const { return gem_g->superChamber(gemId)->chamber(1)->etaPartition(1)->npads(); }

int CSCGEMMotherboard::maxRolls() const { return gem_g->superChamber(gemId)->chamber(1)->nEtaPartitions(); }

void CSCGEMMotherboard::printGEMTriggerPads(int bx_start, int bx_stop, enum CSCPart part) {
  LogTrace("CSCGEMMotherboard") << "------------------------------------------------------------------------"
                                << std::endl;
  LogTrace("CSCGEMMotherboard") << "* GEM trigger pads: " << std::endl;

  for (int bx = bx_start; bx <= bx_stop; bx++) {
    const auto& in_pads = pads_[bx];
    LogTrace("CSCGEMMotherboard") << "N(pads) BX " << bx << " : " << in_pads.size() << std::endl;

    for (const auto& pad : in_pads) {
      LogTrace("CSCGEMMotherboard") << "\tdetId " << GEMDetId(pad.first) << ", pad = " << pad.second;
      const auto& roll_id(GEMDetId(pad.first));

      if (part == CSCPart::ME11 and isPadInOverlap(GEMDetId(roll_id).roll()))
        LogTrace("CSCGEMMotherboard") << " (in overlap)" << std::endl;
      else
        LogTrace("CSCGEMMotherboard") << std::endl;
    }
  }
}

void CSCGEMMotherboard::printGEMTriggerCoPads(int bx_start, int bx_stop, enum CSCPart part) {
  LogTrace("CSCGEMMotherboard") << "------------------------------------------------------------------------"
                                << std::endl;
  LogTrace("CSCGEMMotherboard") << "* GEM trigger coincidence pads: " << std::endl;

  for (int bx = bx_start; bx <= bx_stop; bx++) {
    const auto& in_pads = coPads_[bx];
    LogTrace("CSCGEMMotherboard") << "N(copads) BX " << bx << " : " << in_pads.size() << std::endl;

    for (const auto& pad : in_pads) {
      LogTrace("CSCGEMMotherboard") << "\tdetId " << GEMDetId(pad.first) << ", pad = " << pad.second;
      const auto& roll_id(GEMDetId(pad.first));

      if (part == CSCPart::ME11 and isPadInOverlap(GEMDetId(roll_id).roll()))
        LogTrace("CSCGEMMotherboard") << " (in overlap)" << std::endl;
      else
        LogTrace("CSCGEMMotherboard") << std::endl;
    }
  }
}

CSCMotherboard::LCT_Quality CSCGEMMotherboard::findQualityGEMv1(const CSCALCTDigi& aLCT,
                                                                const CSCCLCTDigi& cLCT,
                                                                int gemlayers) const {
  // Either ALCT or CLCT is invalid
  if (!(aLCT.isValid()) || !(cLCT.isValid())) {
    // No CLCT
    if (aLCT.isValid() && !(cLCT.isValid()))
      return LCT_Quality::NO_CLCT;

    // No ALCT
    else if (!(aLCT.isValid()) && cLCT.isValid())
      return LCT_Quality::NO_ALCT;

    // No ALCT and no CLCT
    else
      return LCT_Quality::INVALID;
  }
  // Both ALCT and CLCT are valid
  else {
    const int pattern(cLCT.getPattern());

    // Layer-trigger in CLCT
    if (pattern == 1)
      return LCT_Quality::CLCT_LAYER_TRIGGER;

    // Multi-layer pattern in CLCT
    else {
      // ALCT quality is the number of layers hit minus 3.
      bool a4 = false;

      // Case of ME11 with GEMs: require 4 layers for ALCT
      if (theStation == 1)
        a4 = aLCT.getQuality() >= 1;

      // Case of ME21 with GEMs: require 4 layers for ALCT+GEM
      if (theStation == 2)
        a4 = aLCT.getQuality() + gemlayers >= 1;

      // CLCT quality is the number of layers hit.
      const bool c4((cLCT.getQuality() >= 4) or (cLCT.getQuality() >= 3 and gemlayers >= 1));

      // quality = 4; "reserved for low-quality muons in future"

      // marginal anode and cathode
      if (!a4 && !c4)
        return LCT_Quality::MARGINAL_ANODE_CATHODE;

      // HQ anode, but marginal cathode
      else if (a4 && !c4)
        return LCT_Quality::HQ_ANODE_MARGINAL_CATHODE;

      // HQ cathode, but marginal anode
      else if (!a4 && c4)
        return LCT_Quality::HQ_CATHODE_MARGINAL_ANODE;

      // HQ muon, but accelerator ALCT
      else if (a4 && c4) {
        if (aLCT.getAccelerator())
          return LCT_Quality::HQ_ACCEL_ALCT;

        else {
          // quality =  9; "reserved for HQ muons with future patterns
          // quality = 10; "reserved for HQ muons with future patterns

          // High quality muons are determined by their CLCT pattern
          if (pattern == 2 || pattern == 3)
            return LCT_Quality::HQ_PATTERN_2_3;

          else if (pattern == 4 || pattern == 5)
            return LCT_Quality::HQ_PATTERN_4_5;

          else if (pattern == 6 || pattern == 7)
            return LCT_Quality::HQ_PATTERN_6_7;

          else if (pattern == 8 || pattern == 9)
            return LCT_Quality::HQ_PATTERN_8_9;

          else if (pattern == 10)
            return LCT_Quality::HQ_PATTERN_10;

          else {
            edm::LogWarning("CSCGEMMotherboard")
                << "findQualityGEMv1: Unexpected CLCT pattern id = " << pattern << " in " << theCSCName_;
            return LCT_Quality::INVALID;
          }
        }
      }
    }
  }
  return LCT_Quality::INVALID;
}

CSCGEMMotherboard::LCT_QualityRun3 CSCGEMMotherboard::findQualityGEMv2(const CSCALCTDigi& aLCT,
                                                                       const CSCCLCTDigi& cLCT,
                                                                       int gemlayers) const {
  // ALCT and CLCT invalid
  if (!(aLCT.isValid()) and !(cLCT.isValid())) {
    return LCT_QualityRun3::INVALID;
  } else if (!aLCT.isValid() && cLCT.isValid() and gemlayers == 2) {
    return LCT_QualityRun3::CLCT_2GEM;
  } else if (aLCT.isValid() && !cLCT.isValid() and gemlayers == 2) {
    return LCT_QualityRun3::ALCT_2GEM;
  } else if (aLCT.isValid() && cLCT.isValid()) {
    if (gemlayers == 0)
      return LCT_QualityRun3::ALCTCLCT;
    else if (gemlayers == 1)
      return LCT_QualityRun3::ALCTCLCT_1GEM;
    else if (gemlayers == 2)
      return LCT_QualityRun3::ALCTCLCT_2GEM;
  }
  return LCT_QualityRun3::INVALID;
}

template <>
const matchesBX<GEMPadDigi>& CSCGEMMotherboard::getPads<GEMPadDigi>() const {
  return pads_;
}

template <>
const matchesBX<GEMCoPadDigi>& CSCGEMMotherboard::getPads<GEMCoPadDigi>() const {
  return coPads_;
}

template <>
int CSCGEMMotherboard::getMaxDeltaBX<GEMPadDigi>() const {
  return maxDeltaBXPad_;
}

template <>
int CSCGEMMotherboard::getMaxDeltaBX<GEMCoPadDigi>() const {
  return maxDeltaBXCoPad_;
}

template <>
int CSCGEMMotherboard::getLctTrigEnable<CSCALCTDigi>() const {
  return alct_trig_enable;
}

template <>
int CSCGEMMotherboard::getLctTrigEnable<CSCCLCTDigi>() const {
  return clct_trig_enable;
}
