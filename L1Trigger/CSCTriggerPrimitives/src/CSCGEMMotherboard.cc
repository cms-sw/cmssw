#include "L1Trigger/CSCTriggerPrimitives/src/CSCGEMMotherboard.h"

CSCGEMMotherboard::CSCGEMMotherboard(unsigned endcap, unsigned station,
                                     unsigned sector, unsigned subsector,
                                     unsigned chamber,
                                     const edm::ParameterSet& conf) :
  CSCUpgradeMotherboard(endcap, station, sector, subsector, chamber, conf)
  , maxDeltaBXPad_(tmbParams_.getParameter<int>("maxDeltaBXPad"))
  , maxDeltaBXCoPad_(tmbParams_.getParameter<int>("maxDeltaBXCoPad"))
  , useOldLCTDataFormat_(tmbParams_.getParameter<bool>("useOldLCTDataFormat"))
  , promoteALCTGEMpattern_(tmbParams_.getParameter<bool>("promoteALCTGEMpattern"))
  , promoteALCTGEMquality_(tmbParams_.getParameter<bool>("promoteALCTGEMquality"))
  , doLCTGhostBustingWithGEMs_(tmbParams_.getParameter<bool>("doLCTGhostBustingWithGEMs"))
{
  // super chamber has layer=0!
  gemId = GEMDetId(theRegion, 1, theStation, 0, theChamber, 0).rawId();

  const edm::ParameterSet coPadParams(station==1 ?
				      conf.getParameter<edm::ParameterSet>("copadParamGE11") :
				      conf.getParameter<edm::ParameterSet>("copadParamGE21"));
  coPadProcessor.reset( new GEMCoPadProcessor(endcap, station, chamber, coPadParams) );

  maxDeltaPadL1_ = (par ? tmbParams_.getParameter<int>("maxDeltaPadL1Even") :
		    tmbParams_.getParameter<int>("maxDeltaPadL1Odd") );
  maxDeltaPadL2_ = (par ? tmbParams_.getParameter<int>("maxDeltaPadL2Even") :
		    tmbParams_.getParameter<int>("maxDeltaPadL2Odd") );
}

CSCGEMMotherboard::CSCGEMMotherboard()
  : CSCUpgradeMotherboard()
{
}

CSCGEMMotherboard::~CSCGEMMotherboard()
{
}

void CSCGEMMotherboard::clear()
{
  CSCUpgradeMotherboard::clear();
  pads_.clear();
  coPads_.clear();
}


void CSCGEMMotherboard::run(const CSCWireDigiCollection* wiredc,
                            const CSCComparatorDigiCollection* compdc,
                            const GEMPadDigiClusterCollection* gemClusters)
{
  std::unique_ptr<GEMPadDigiCollection> gemPads(new GEMPadDigiCollection());
  coPadProcessor->declusterize(gemClusters, *gemPads);
  run(wiredc, compdc, gemPads.get());
}


void CSCGEMMotherboard::retrieveGEMPads(const GEMPadDigiCollection* gemPads, unsigned id)
{
  pads_.clear();
  auto superChamber(gem_g->superChamber(id));
  for (const auto& ch : superChamber->chambers()) {
    for (const auto& roll : ch->etaPartitions()) {
      GEMDetId roll_id(roll->id());
      auto pads_in_det = gemPads->get(roll_id);
      for (auto pad = pads_in_det.first; pad != pads_in_det.second; ++pad) {
        const int bx_shifted(CSCConstants::LCT_CENTRAL_BX + pad->bx());
        // consider matches with BX difference +1/0/-1
        for (int bx = bx_shifted - maxDeltaBXPad_;bx <= bx_shifted + maxDeltaBXPad_; ++bx) {
          pads_[bx].emplace_back(roll_id.rawId(), *pad);
        }
      }
    }
  }
}

void CSCGEMMotherboard::retrieveGEMCoPads()
{
  coPads_.clear();
  for (const auto& copad: gemCoPadV){
    GEMDetId detId(theRegion, 1, theStation, 0, theChamber, 0);
    // only consider matches with same BX
    coPads_[CSCConstants::LCT_CENTRAL_BX + copad.bx(1)].emplace_back(detId.rawId(), copad);
  }
}

CSCCorrelatedLCTDigi CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& alct,
                                                         const GEMCoPadDigi& gem,
                                                         int trknmb) const
{
  return constructLCTsGEM(alct, CSCCLCTDigi(), GEMPadDigi(), gem, trknmb);
}


CSCCorrelatedLCTDigi CSCGEMMotherboard::constructLCTsGEM(const CSCCLCTDigi& clct,
                                                         const GEMCoPadDigi& gem,
                                                         int trknmb) const
{
  return constructLCTsGEM(CSCALCTDigi(), clct, GEMPadDigi(), gem, trknmb);
}

CSCCorrelatedLCTDigi CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& alct,
                                                         const CSCCLCTDigi& clct,
                                                         const GEMCoPadDigi& gem,
                                                         int trknmb) const
{
  return constructLCTsGEM(alct, clct, GEMPadDigi(), gem, trknmb);
}


CSCCorrelatedLCTDigi CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& alct,
                                                         const CSCCLCTDigi& clct,
                                                         const GEMPadDigi& gem,
                                                         int trknmb) const
{
  return constructLCTsGEM(alct, clct, gem, GEMCoPadDigi(), trknmb);
}

CSCCorrelatedLCTDigi CSCGEMMotherboard::constructLCTsGEM(const CSCALCTDigi& alct,
                                                         const CSCCLCTDigi& clct,
                                                         const GEMPadDigi& gem1,
                                                         const GEMCoPadDigi& gem2,
                                                         int trknmb) const
{
  int pattern = 0, quality = 0, bx = 0, keyStrip = 0, keyWG = 0, bend = 0;

  // make a new LCT
  CSCCorrelatedLCTDigi thisLCT;
  if (not alct.isValid() and not clct.isValid()) {
    LogTrace("CSCGEMCMotherboard") << "Warning!!! either ALCT or CLCT not valid, return invalid LCT \n";
    return thisLCT;
  }

  // Determine the case and assign properties depending on the LCT dataformat (old/new)
  if (alct.isValid() and clct.isValid() and gem1.isValid() and not gem2.isValid()) {
    pattern = encodePattern(clct.getPattern());
    quality = findQualityGEM(alct, clct, 1);
    bx = alct.getBX();
    keyStrip = clct.getKeyStrip();
    keyWG = alct.getKeyWG();
    bend = clct.getBend();
    thisLCT.setALCT(getBXShiftedALCT(alct));
    thisLCT.setCLCT(clct);
    thisLCT.setGEM1(gem1);
    thisLCT.setType(CSCCorrelatedLCTDigi::ALCTCLCTGEM);
  }
  else if (alct.isValid() and clct.isValid() and not gem1.isValid() and gem2.isValid()) {
    pattern = encodePattern(clct.getPattern());
    quality = findQualityGEM(alct, clct, 2);
    bx = alct.getBX();
    keyStrip = clct.getKeyStrip();
    keyWG = alct.getKeyWG();
    bend = clct.getBend();
    thisLCT.setALCT(getBXShiftedALCT(alct));
    thisLCT.setCLCT(clct);
    thisLCT.setGEM1(gem2.first());
    thisLCT.setGEM2(gem2.second());
    thisLCT.setType(CSCCorrelatedLCTDigi::ALCTCLCT2GEM);
  }
  else if (alct.isValid() and gem2.isValid() and not clct.isValid()) {
    //in ME11
    //ME1b: keyWG >15,
    //ME1a and ME1b overlap:  10<=keyWG<=15
    //ME1a: keyWG < 10
    //in overlap region, firstly try a match in ME1b

    auto p(getCSCPart(-1));//use -1 as fake halfstrip, it returns ME11 if station==1 && (ring==1 or ring==4)
    if (p == CSCPart::ME11 and alct.getKeyWG() <= 15)
      p = CSCPart::ME1B;
    const auto& mymap1 = getLUT()->get_gem_pad_to_csc_hs(par, p);
    pattern = promoteALCTGEMpattern_ ? 10 : 0;
    quality = promoteALCTGEMquality_ ? 15 : 11;
    bx = alct.getBX();
    // GEM pad number is counting from 1
    keyStrip = mymap1[gem2.pad(2) - 1];
    keyWG = alct.getKeyWG();
    thisLCT.setALCT(getBXShiftedALCT(alct));
    thisLCT.setGEM1(gem2.first());
    thisLCT.setGEM2(gem2.second());
    thisLCT.setType(CSCCorrelatedLCTDigi::ALCT2GEM);
  }
  else if (clct.isValid() and gem2.isValid() and not alct.isValid()) {
    auto p(getCSCPart(clct.getKeyStrip()));
    const auto& mymap2 = getLUT()->get_gem_roll_to_csc_wg(par, p);
    pattern = encodePattern(clct.getPattern());
    quality = promoteCLCTGEMquality_ ? 15 : 11;
    bx = gem2.bx(1) + CSCConstants::LCT_CENTRAL_BX;
    keyStrip = clct.getKeyStrip();
    // choose the corresponding wire-group in the middle of the partition
    keyWG = mymap2[gem2.roll()];
    bend = clct.getBend();
    thisLCT.setCLCT(clct);
    thisLCT.setGEM1(gem2.first());
    thisLCT.setGEM2(gem2.second());
    thisLCT.setType(CSCCorrelatedLCTDigi::CLCT2GEM);
  }

  // fill the rest of the properties
  thisLCT.setTrknmb(trknmb);
  thisLCT.setValid(1);
  thisLCT.setQuality(quality);
  thisLCT.setWireGroup(keyWG);
  thisLCT.setStrip(keyStrip);
  thisLCT.setPattern(pattern);
  thisLCT.setBend(bend);
  thisLCT.setBX(bx);
  thisLCT.setMPCLink(0);
  thisLCT.setBX0(0);
  thisLCT.setSyncErr(0);
  thisLCT.setCSCID(theTrigChamber);

  // future work: add a section that produces LCTs according
  // to the new LCT dataformat (not yet defined)

  // return new LCT
  return thisLCT;
}


bool CSCGEMMotherboard::isPadInOverlap(int roll) const
{
  // this only works for ME1A!
  const auto& mymap = (getLUT()->get_csc_wg_to_gem_roll(par));
  for (unsigned i=0; i<mymap.size(); i++) {
    // overlap region are WGs 10-15
    if ((i < 10) or (i > 15)) continue;
    if ((mymap[i].first <= roll) and (roll <= mymap[i].second)) return true;
  }
  return false;
}

bool CSCGEMMotherboard::isGEMDetId(unsigned int p) const
{
  return (DetId(p).subdetId() == MuonSubdetId::GEM and
          DetId(p).det() == DetId::Muon);
}

int CSCGEMMotherboard::getBX(const GEMPadDigi& p) const
{
  return p.bx();
}

int CSCGEMMotherboard::getBX(const GEMCoPadDigi& p) const
{
  return p.bx(1);
}

int CSCGEMMotherboard::getRoll(const GEMPadDigiId& p) const
{
  return GEMDetId(p.first).roll();
}

int CSCGEMMotherboard::getRoll(const GEMCoPadDigiId& p) const
{
  return p.second.roll();
}

int CSCGEMMotherboard::getRoll(const CSCALCTDigi& alct) const
{
  return (getLUT()->get_csc_wg_to_gem_roll(par))[alct.getKeyWG()].first;
}

float CSCGEMMotherboard::getPad(const GEMPadDigi& p) const
{
  return p.pad();
}

float CSCGEMMotherboard::getPad(const GEMCoPadDigi& p) const
{
  // average pad number for a GEMCoPad
  return 0.5*(p.pad(1) + p.pad(2));
}

float CSCGEMMotherboard::getPad(const CSCCLCTDigi& clct, enum CSCPart part) const
{
  const auto& mymap = (getLUT()->get_csc_hs_to_gem_pad(par, part));
  return 0.5*(mymap[clct.getKeyStrip()].first + mymap[clct.getKeyStrip()].second);
}

void CSCGEMMotherboard::setupGeometry()
{
  CSCUpgradeMotherboard::setupGeometry();
  generator_->setGEMGeometry(gem_g);
}

void CSCGEMMotherboard::printGEMTriggerPads(int bx_start, int bx_stop, enum CSCPart part)
{
  LogTrace("CSCGEMMotherboard") << "------------------------------------------------------------------------" << std::endl;
  LogTrace("CSCGEMMotherboard") << "* GEM trigger pads: " << std::endl;

  for (int bx = bx_start; bx <= bx_stop; bx++) {
    const auto& in_pads = pads_[bx];
    LogTrace("CSCGEMMotherboard") << "N(pads) BX " << bx << " : " << in_pads.size() << std::endl;

    for (const auto& pad : in_pads){
      LogTrace("CSCGEMMotherboard") << "\tdetId " << GEMDetId(pad.first) << ", pad = " << pad.second;
      const auto& roll_id(GEMDetId(pad.first));

      if (part==CSCPart::ME11 and isPadInOverlap(GEMDetId(roll_id).roll()))
        LogTrace("CSCGEMMotherboard") << " (in overlap)" << std::endl;
      else
        LogTrace("CSCGEMMotherboard") << std::endl;
    }
  }
}


void CSCGEMMotherboard::printGEMTriggerCoPads(int bx_start, int bx_stop, enum CSCPart part)
{
  LogTrace("CSCGEMMotherboard") << "------------------------------------------------------------------------" << std::endl;
  LogTrace("CSCGEMMotherboard") << "* GEM trigger coincidence pads: " << std::endl;

  for (int bx = bx_start; bx <= bx_stop; bx++) {
    const auto& in_pads = coPads_[bx];
    LogTrace("CSCGEMMotherboard") << "N(copads) BX " << bx << " : " << in_pads.size() << std::endl;

    for (const auto& pad : in_pads){
      LogTrace("CSCGEMMotherboard") << "\tdetId " << GEMDetId(pad.first) << ", pad = " << pad.second;
      const auto& roll_id(GEMDetId(pad.first));

      if (part==CSCPart::ME11 and isPadInOverlap(GEMDetId(roll_id).roll())) LogTrace("CSCGEMMotherboard") << " (in overlap)" << std::endl;
      else LogTrace("CSCGEMMotherboard") << std::endl;
    }
  }
}


unsigned int CSCGEMMotherboard::findQualityGEM(const CSCALCTDigi& aLCT,
                                               const CSCCLCTDigi& cLCT, int gemlayers) const
{
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
      if (theStation==1) a4 = aLCT.getQuality() >= 1;

      // Case of ME21 with GEMs: require 4 layers for ALCT+GEM
      if (theStation==2) a4 = aLCT.getQuality() + gemlayers >=1;

      // CLCT quality is the number of layers hit.
      const bool c4((cLCT.getQuality() >= 4) or (cLCT.getQuality() >= 3 and gemlayers>=1));

      // quality = 4; "reserved for low-quality muons in future"

      // marginal anode and cathode
      if (!a4 && !c4)
        return LCT_Quality::MARGINAL_ANODE_CATHODE;

      // HQ anode, but marginal cathode
      else if ( a4 && !c4)
        return LCT_Quality::HQ_ANODE_MARGINAL_CATHODE;

      // HQ cathode, but marginal anode
      else if (!a4 &&  c4)
        return LCT_Quality::HQ_CATHODE_MARGINAL_ANODE;

      // HQ muon, but accelerator ALCT
      else if ( a4 &&  c4) {

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
            if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongValues")
                              << "+++ findQuality: Unexpected CLCT pattern id = "
                              << pattern << "+++\n";
            return LCT_Quality::INVALID;
          }
        }
      }
    }
  }
  return LCT_Quality::INVALID;
}


template <> const matchesBX<GEMPadDigi>&
CSCGEMMotherboard::getPads<GEMPadDigi>() const
{
  return pads_;
}

template <> const matchesBX<GEMCoPadDigi>&
CSCGEMMotherboard::getPads<GEMCoPadDigi>() const
{
  return coPads_;
}

template <>
int CSCGEMMotherboard::getMaxDeltaBX<GEMPadDigi>() const
{
  return maxDeltaBXPad_;
}

template <>
int CSCGEMMotherboard::getMaxDeltaBX<GEMCoPadDigi>() const
{
  return maxDeltaBXCoPad_;
}

template <>
int CSCGEMMotherboard::getLctTrigEnable<CSCALCTDigi>() const
{
  return alct_trig_enable;
}

template <>
int CSCGEMMotherboard::getLctTrigEnable<CSCCLCTDigi>() const
{
  return clct_trig_enable;
}
