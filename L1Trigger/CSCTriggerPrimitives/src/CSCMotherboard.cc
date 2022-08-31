#include "L1Trigger/CSCTriggerPrimitives/interface/CSCMotherboard.h"
#include <iostream>
#include <memory>

// Default values of configuration parameters.
const unsigned int CSCMotherboard::def_mpc_block_me1a = 1;
const unsigned int CSCMotherboard::def_alct_trig_enable = 0;
const unsigned int CSCMotherboard::def_clct_trig_enable = 0;
const unsigned int CSCMotherboard::def_match_trig_enable = 1;
const unsigned int CSCMotherboard::def_match_trig_window_size = 7;
const unsigned int CSCMotherboard::def_tmb_l1a_window_size = 7;

CSCMotherboard::CSCMotherboard(unsigned endcap,
                               unsigned station,
                               unsigned sector,
                               unsigned subsector,
                               unsigned chamber,
                               const edm::ParameterSet& conf)
    : CSCBaseboard(endcap, station, sector, subsector, chamber, conf) {
  // Normal constructor.  -JM
  // Pass ALCT, CLCT, and common parameters on to ALCT and CLCT processors.
  static std::atomic<bool> config_dumped{false};

  mpc_block_me1a = tmbParams_.getParameter<unsigned int>("mpcBlockMe1a");
  alct_trig_enable = tmbParams_.getParameter<unsigned int>("alctTrigEnable");
  clct_trig_enable = tmbParams_.getParameter<unsigned int>("clctTrigEnable");
  match_trig_enable = tmbParams_.getParameter<unsigned int>("matchTrigEnable");
  match_trig_window_size = tmbParams_.getParameter<unsigned int>("matchTrigWindowSize");
  tmb_l1a_window_size =  // Common to CLCT and TMB
      tmbParams_.getParameter<unsigned int>("tmbL1aWindowSize");

  // configuration handle for number of early time bins
  early_tbins = tmbParams_.getParameter<int>("tmbEarlyTbins");

  // whether to not reuse CLCTs that were used by previous matching ALCTs
  drop_used_clcts = tmbParams_.getParameter<bool>("tmbDropUsedClcts");

  // whether to readout only the earliest two LCTs in readout window
  readout_earliest_2 = tmbParams_.getParameter<bool>("tmbReadoutEarliest2");

  match_earliest_clct_only_ = tmbParams_.getParameter<bool>("matchEarliestClctOnly");

  infoV = tmbParams_.getParameter<int>("verbosity");

  alctProc = std::make_unique<CSCAnodeLCTProcessor>(endcap, station, sector, subsector, chamber, conf);
  clctProc = std::make_unique<CSCCathodeLCTProcessor>(endcap, station, sector, subsector, chamber, conf);

  // Check and print configuration parameters.
  checkConfigParameters();
  if (infoV > 0 && !config_dumped) {
    dumpConfigParams();
    config_dumped = true;
  }

  allLCTs_.setMatchTrigWindowSize(match_trig_window_size);

  // get the preferred CLCT BX match array
  preferred_bx_match_ = tmbParams_.getParameter<std::vector<int>>("preferredBxMatch");

  // quality assignment
  qualityAssignment_ = std::make_unique<LCTQualityAssignment>(endcap, station, sector, subsector, chamber, conf);

  // quality control of stubs
  qualityControl_ = std::make_unique<LCTQualityControl>(endcap, station, sector, subsector, chamber, conf);

  // shower-trigger source
  showerSource_ = showerParams_.getParameter<std::vector<unsigned>>("source");

  unsigned csc_idx = CSCDetId::iChamberType(theStation, theRing) - 2;
  thisShowerSource_ = showerSource_[csc_idx];

  // enable the upgrade processors for ring 1 stations
  if (runPhase2_ and theRing == 1) {
    clctProc = std::make_unique<CSCUpgradeCathodeLCTProcessor>(endcap, station, sector, subsector, chamber, conf);
    if (enableAlctPhase2_) {
      alctProc = std::make_unique<CSCUpgradeAnodeLCTProcessor>(endcap, station, sector, subsector, chamber, conf);
    }
  }

  // set up helper class to check if ALCT and CLCT cross
  ignoreAlctCrossClct_ = tmbParams_.getParameter<bool>("ignoreAlctCrossClct");
  if (!ignoreAlctCrossClct_) {
    cscOverlap_ = std::make_unique<CSCALCTCrossCLCT>(endcap, station, theRing, ignoreAlctCrossClct_, conf);
  }
}

void CSCMotherboard::clear() {
  // clear the processors
  if (alctProc)
    alctProc->clear();
  if (clctProc)
    clctProc->clear();

  // clear the ALCT and CLCT containers
  alctV.clear();
  clctV.clear();
  lctV.clear();

  allLCTs_.clear();

  for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++) {
    showers_[bx].clear();
  }
}

// Set configuration parameters obtained via EventSetup mechanism.
void CSCMotherboard::setConfigParameters(const CSCDBL1TPParameters* conf) {
  static std::atomic<bool> config_dumped{false};

  // Config. parameters for the TMB itself.
  mpc_block_me1a = conf->tmbMpcBlockMe1a();
  alct_trig_enable = conf->tmbAlctTrigEnable();
  clct_trig_enable = conf->tmbClctTrigEnable();
  match_trig_enable = conf->tmbMatchTrigEnable();
  match_trig_window_size = conf->tmbMatchTrigWindowSize();
  tmb_l1a_window_size = conf->tmbTmbL1aWindowSize();

  // Config. paramteres for ALCT and CLCT processors.
  alctProc->setConfigParameters(conf);
  clctProc->setConfigParameters(conf);

  // Check and print configuration parameters.
  checkConfigParameters();
  if (!config_dumped) {
    dumpConfigParams();
    config_dumped = true;
  }
}

void CSCMotherboard::setESLookupTables(const CSCL1TPLookupTableCCLUT* conf) { lookupTableCCLUT_ = conf; }

void CSCMotherboard::setESLookupTables(const CSCL1TPLookupTableME11ILT* conf) { lookupTableME11ILT_ = conf; }

void CSCMotherboard::setESLookupTables(const CSCL1TPLookupTableME21ILT* conf) { lookupTableME21ILT_ = conf; }

void CSCMotherboard::run(const CSCWireDigiCollection* wiredc, const CSCComparatorDigiCollection* compdc) {
  // Step 1: Setup
  clear();

  // Check for existing processors
  if (!(alctProc && clctProc)) {
    edm::LogError("CSCMotherboard|SetupError") << "+++ run() called for non-existing ALCT/CLCT processor! +++ \n";
    return;
  }

  // set geometry
  alctProc->setCSCGeometry(cscGeometry_);
  clctProc->setCSCGeometry(cscGeometry_);

  // set CCLUT parameters if necessary
  if (runCCLUT_) {
    clctProc->setESLookupTables(lookupTableCCLUT_);
  }

  // Step 2: Run the processors
  alctV = alctProc->run(wiredc);  // run anodeLCT
  clctV = clctProc->run(compdc);  // run cathodeLCT

  // Step 2b: encode high multiplicity bits (independent of LCT construction)
  encodeHighMultiplicityBits();

  // if there are no ALCTs and no CLCTs, it does not make sense to run this TMB
  if (alctV.empty() and clctV.empty())
    return;

  // step 3: match the ALCTs to the CLCTs
  matchALCTCLCT();

  // Step 4: Select at most 2 LCTs per BX
  selectLCTs();
}

void CSCMotherboard::matchALCTCLCT() {
  // array to mask CLCTs
  bool used_clct_mask[CSCConstants::MAX_CLCT_TBINS] = {false};

  // Step 3: ALCT-centric ALCT-to-CLCT matching
  int bx_clct_matched = 0;  // bx of last matched CLCT
  for (int bx_alct = 0; bx_alct < CSCConstants::MAX_ALCT_TBINS; bx_alct++) {
    // There should be at least one valid CLCT or ALCT for a
    // correlated LCT to be formed.  Decision on whether to reject
    // non-complete LCTs (and if yes of which type) is made further
    // upstream.
    if (alctProc->getBestALCT(bx_alct).isValid()) {
      // Look for CLCTs within the match-time window.  The window is
      // centered at the ALCT bx; therefore, we make an assumption
      // that anode and cathode hits are perfectly synchronized.  This
      // is always true for MC, but only an approximation when the
      // data is analyzed (which works fairly good as long as wide
      // windows are used).  To get rid of this assumption, one would
      // need to access "full BX" words, which are not readily
      // available.
      bool is_matched = false;
      // loop on the preferred "delta BX" array
      for (unsigned mbx = 0; mbx < match_trig_window_size; mbx++) {
        // evaluate the preffered CLCT BX, taking into account that there is an offset in the simulation
        int bx_clct = bx_alct + preferred_bx_match_[mbx] - CSCConstants::ALCT_CLCT_OFFSET;
        // check that the CLCT BX is valid
        if (bx_clct >= CSCConstants::MAX_CLCT_TBINS or bx_clct < 0)
          continue;
        // do not consider previously matched CLCTs
        if (drop_used_clcts && used_clct_mask[bx_clct])
          continue;
        // only consider >=4 layer CLCTs for ALCT-CLCT type LCTs
        // this condition is lowered to >=3 layers for CLCTs in the
        // matchALCTCLCTGEM function
        if (clctProc->getBestCLCT(bx_clct).getQuality() <= 3)
          continue;
        // a valid CLCT with sufficient layers!
        if (clctProc->getBestCLCT(bx_clct).isValid()) {
          if (infoV > 1)
            LogTrace("CSCMotherboard") << "Successful ALCT-CLCT match: bx_alct = " << bx_alct
                                       << "; bx_clct = " << bx_clct << "; mbx = " << mbx;
          // now correlate the ALCT and CLCT into LCT.
          // smaller mbx means more preferred!
          correlateLCTs(alctProc->getBestALCT(bx_alct),
                        alctProc->getSecondALCT(bx_alct),
                        clctProc->getBestCLCT(bx_clct),
                        clctProc->getSecondCLCT(bx_clct),
                        allLCTs_(bx_alct, mbx, 0),
                        allLCTs_(bx_alct, mbx, 1),
                        CSCCorrelatedLCTDigi::ALCTCLCT);
          // when the first LCT is valid, you can mask the matched CLCT and/or
          // move on to the next ALCT if match_earliest_clct_only_ is set to true
          if (allLCTs_(bx_alct, mbx, 0).isValid()) {
            is_matched = true;
            used_clct_mask[bx_clct] = true;
            bx_clct_matched = bx_clct;
            if (match_earliest_clct_only_)
              break;
          }
        }
      }
      // No CLCT within the match time interval found: report ALCT-only LCT
      // (use dummy CLCTs).
      if (!is_matched) {
        if (infoV > 1)
          LogTrace("CSCMotherboard") << "Unsuccessful ALCT-CLCT match (ALCT only): bx_alct = " << bx_alct
                                     << " first ALCT " << alctProc->getBestALCT(bx_alct);
        if (alct_trig_enable)
          correlateLCTs(alctProc->getBestALCT(bx_alct),
                        alctProc->getSecondALCT(bx_alct),
                        clctProc->getBestCLCT(bx_alct),
                        clctProc->getSecondCLCT(bx_alct),
                        allLCTs_(bx_alct, 0, 0),
                        allLCTs_(bx_alct, 0, 1),
                        CSCCorrelatedLCTDigi::ALCTONLY);
      }
    }
    // No valid ALCTs; attempt to make CLCT-only LCT.  Use only CLCTs
    // which have zeroth chance to be matched at later cathode times.
    // (I am not entirely sure this perfectly matches the firmware logic.)
    // Use dummy ALCTs.
    else {
      int bx_clct = bx_alct - match_trig_window_size / 2;
      if (bx_clct >= 0 && bx_clct > bx_clct_matched) {
        if (clctProc->getBestCLCT(bx_clct).isValid() and clct_trig_enable) {
          if (infoV > 1)
            LogTrace("CSCMotherboard") << "Unsuccessful ALCT-CLCT match (CLCT only): bx_clct = " << bx_clct;
          correlateLCTs(alctProc->getBestALCT(bx_alct),
                        alctProc->getSecondALCT(bx_alct),
                        clctProc->getBestCLCT(bx_clct),
                        clctProc->getSecondCLCT(bx_clct),
                        allLCTs_(bx_clct, 0, 0),
                        allLCTs_(bx_clct, 0, 1),
                        CSCCorrelatedLCTDigi::CLCTONLY);
        }
      }
    }
  }
}

// Returns vector of read-out correlated LCTs, if any.  Starts with
// the vector of all found LCTs and selects the ones in the read-out
// time window.
std::vector<CSCCorrelatedLCTDigi> CSCMotherboard::readoutLCTs() const {
  // temporary container for further selection
  std::vector<CSCCorrelatedLCTDigi> tmpV;

  /*
    LCTs in the BX window [early_tbin,...,late_tbin] are considered good for physics
    The central LCT BX is time bin 8.
    For tmb_l1a_window_size set to 7 (Run-1, Run-2), the window is [5, 6, 7, 8, 9, 10, 11]
    For tmb_l1a_window_size set to 5 (Run-3), the window is [6, 7, 8, 9, 10]
    For tmb_l1a_window_size set to 3 (Run-4?), the window is [ 7, 8, 9]
  */
  const unsigned delta_tbin = tmb_l1a_window_size / 2;
  int early_tbin = CSCConstants::LCT_CENTRAL_BX - delta_tbin;
  int late_tbin = CSCConstants::LCT_CENTRAL_BX + delta_tbin;
  /*
     Special case for an even-numbered time-window,
     For instance tmb_l1a_window_size set to 6: [5, 6, 7, 8, 9, 10]
  */
  if (tmb_l1a_window_size % 2 == 0)
    late_tbin = CSCConstants::LCT_CENTRAL_BX + delta_tbin - 1;
  const int max_late_tbin = CSCConstants::MAX_LCT_TBINS - 1;

  // debugging messages when early_tbin or late_tbin has a suspicious value
  if (early_tbin < 0) {
    edm::LogWarning("CSCMotherboard|SuspiciousParameters")
        << "Early time bin (early_tbin) smaller than minimum allowed, which is 0. set early_tbin to 0.";
    early_tbin = 0;
  }
  if (late_tbin > max_late_tbin) {
    edm::LogWarning("CSCMotherboard|SuspiciousParameters")
        << "Late time bin (late_tbin) larger than maximum allowed, which is " << max_late_tbin
        << ". set early_tbin to max allowed";
    late_tbin = CSCConstants::MAX_LCT_TBINS - 1;
  }

  // Start from the vector of all found correlated LCTs and select
  // those within the LCT*L1A coincidence window.
  int bx_readout = -1;
  for (const auto& lct : lctV) {
    // extra check on invalid LCTs
    if (!lct.isValid()) {
      continue;
    }

    const int bx = lct.getBX();
    // Skip LCTs found too early relative to L1Accept.
    if (bx < early_tbin) {
      if (infoV > 1)
        LogDebug("CSCMotherboard") << " Do not report correlated LCT on key halfstrip " << lct.getStrip()
                                   << " and key wire " << lct.getKeyWG() << ": found at bx " << bx
                                   << ", whereas the earliest allowed bx is " << early_tbin;
      continue;
    }

    // Skip LCTs found too late relative to L1Accept.
    if (bx > late_tbin) {
      if (infoV > 1)
        LogDebug("CSCMotherboard") << " Do not report correlated LCT on key halfstrip " << lct.getStrip()
                                   << " and key wire " << lct.getKeyWG() << ": found at bx " << bx
                                   << ", whereas the latest allowed bx is " << late_tbin;
      continue;
    }

    // Do not report LCTs found in ME1/A if mpc_block_me1a is set.
    if (mpc_block_me1a and isME11_ and lct.getStrip() > CSCConstants::MAX_HALF_STRIP_ME1B) {
      continue;
    }

    // If (readout_earliest_2) take only LCTs in the earliest bx in the read-out window:
    // in digi->raw step, LCTs have to be packed into the TMB header, and
    // currently there is room just for two.
    if (readout_earliest_2) {
      if (bx_readout == -1 || bx == bx_readout) {
        tmpV.push_back(lct);
        if (bx_readout == -1)
          bx_readout = bx;
      }
    }
    // if readout_earliest_2 == false, save all LCTs
    else {
      tmpV.push_back(lct);
    }
  }

  // do a final check on the LCTs in readout
  qualityControl_->checkMultiplicityBX(tmpV);
  for (const auto& lct : tmpV) {
    qualityControl_->checkValid(lct);
    /*std::cout << "\n########################################## Emu LCT ##########################################\n" << std::endl;
    std::cout << "Emu LCT: " << lct << std::endl;
    std::cout << "\n########################################## THE END ##########################################\n" << std::endl;*/
  }

  return tmpV;
}

std::vector<CSCShowerDigi> CSCMotherboard::readoutShower() const {
  unsigned minbx_readout = CSCConstants::LCT_CENTRAL_BX - tmb_l1a_window_size / 2;
  unsigned maxbx_readout = CSCConstants::LCT_CENTRAL_BX + tmb_l1a_window_size / 2;
  unsigned minBXdiff = 2 * tmb_l1a_window_size;  //impossible value
  unsigned minBX = 0;
  std::vector<CSCShowerDigi> showerOut;
  for (unsigned bx = minbx_readout; bx < maxbx_readout; bx++) {
    unsigned bx_diff = (bx > bx - CSCConstants::LCT_CENTRAL_BX) ? bx - CSCConstants::LCT_CENTRAL_BX
                                                                : CSCConstants::LCT_CENTRAL_BX - bx;
    if (showers_[bx].isValid() and bx_diff < minBXdiff) {
      minBXdiff = bx_diff;
      minBX = bx;
    }
  }

  for (unsigned bx = minbx_readout; bx < maxbx_readout; bx++)
    if (bx == minBX)
      showerOut.push_back(showers_[bx]);
  return showerOut;
}

void CSCMotherboard::correlateLCTs(const CSCALCTDigi& bALCT,
                                   const CSCALCTDigi& sALCT,
                                   const CSCCLCTDigi& bCLCT,
                                   const CSCCLCTDigi& sCLCT,
                                   CSCCorrelatedLCTDigi& bLCT,
                                   CSCCorrelatedLCTDigi& sLCT,
                                   int type) const {
  CSCALCTDigi bestALCT = bALCT;
  CSCALCTDigi secondALCT = sALCT;
  CSCCLCTDigi bestCLCT = bCLCT;
  CSCCLCTDigi secondCLCT = sCLCT;

  // extra check to make sure that both CLCTs have at least 4 layers
  // for regular ALCT-CLCT type LCTs. A check was already done on the
  // best CLCT, but not yet on the second best CLCT. The check on best
  // CLCT is repeated for completeness
  if (bestCLCT.getQuality() <= 3)
    bestCLCT.clear();
  if (secondCLCT.getQuality() <= 3)
    secondCLCT.clear();

  // if the best ALCT/CLCT is valid, but the second ALCT/CLCT is not,
  // the information is copied over
  copyValidToInValidALCT(bestALCT, secondALCT);
  copyValidToInValidCLCT(bestCLCT, secondCLCT);

  // ALCT-only LCTs
  const bool bestCase1(alct_trig_enable and bestALCT.isValid());
  // CLCT-only LCTs
  const bool bestCase2(clct_trig_enable and bestCLCT.isValid());
  /*
    Normal case: ALCT-CLCT matched LCTs. We require ALCT and CLCT to be valid.
    Optionally, we can check if the ALCT cross the CLCT. This check will always return true
    for a valid ALCT-CLCT pair in non-ME1/1 chambers. For ME1/1 chambers, it returns true,
    only when the ALCT-CLCT pair crosses and when the parameter "checkAlctCrossClct" is set to True.
    It is recommended to keep "checkAlctCrossClct" set to False, so that the EMTF receives
    all information, even if it's unphysical.
  */
  const bool bestCase3(match_trig_enable and bestALCT.isValid() and bestCLCT.isValid() and
                       doesALCTCrossCLCT(bestALCT, bestCLCT));

  // at least one of the cases must be valid
  if (bestCase1 or bestCase2 or bestCase3) {
    constructLCTs(bestALCT, bestCLCT, type, 1, bLCT);
  }

  // ALCT-only LCTs
  const bool secondCase1(alct_trig_enable and secondALCT.isValid());
  // CLCT-only LCTs
  const bool secondCase2(clct_trig_enable and secondCLCT.isValid());
  /*
    Normal case: ALCT-CLCT matched LCTs. We require ALCT and CLCT to be valid.
    Optionally, we can check if the ALCT cross the CLCT. This check will always return true
    for a valid ALCT-CLCT pair in non-ME1/1 chambers. For ME1/1 chambers, it returns true,
    only when the ALCT-CLCT pair crosses and when the parameter "checkAlctCrossClct" is set to True.
    It is recommended to keep "checkAlctCrossClct" set to False, so that the EMTF receives
    all information, even if it's unphysical.
  */
  const bool secondCase3(match_trig_enable and secondALCT.isValid() and secondCLCT.isValid() and
                         doesALCTCrossCLCT(secondALCT, secondCLCT));

  // at least one component must be different in order to consider the secondLCT
  if ((secondALCT != bestALCT) or (secondCLCT != bestCLCT)) {
    // at least one of the cases must be valid
    if (secondCase1 or secondCase2 or secondCase3)
      constructLCTs(secondALCT, secondCLCT, type, 2, sLCT);
  }
}

// copy the valid ALCT/CLCT information to the valid ALCT
void CSCMotherboard::copyValidToInValidALCT(CSCALCTDigi& bestALCT, CSCALCTDigi& secondALCT) const {
  if (bestALCT.isValid() and !secondALCT.isValid())
    secondALCT = bestALCT;
}

// copy the valid CLCT information to the valid CLCT
void CSCMotherboard::copyValidToInValidCLCT(CSCCLCTDigi& bestCLCT, CSCCLCTDigi& secondCLCT) const {
  if (bestCLCT.isValid() and !secondCLCT.isValid())
    secondCLCT = bestCLCT;
}

bool CSCMotherboard::doesALCTCrossCLCT(const CSCALCTDigi& alct, const CSCCLCTDigi& clct) const {
  if (ignoreAlctCrossClct_)
    return true;
  else
    return cscOverlap_->doesALCTCrossCLCT(alct, clct);
}

// This method calculates all the TMB words and then passes them to the
// constructor of correlated LCTs.
void CSCMotherboard::constructLCTs(
    const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT, int type, int trknmb, CSCCorrelatedLCTDigi& thisLCT) const {
  thisLCT.setValid(true);
  thisLCT.setType(type);
  // make sure to shift the ALCT BX from 8 to 3 and the CLCT BX from 8 to 7!
  thisLCT.setALCT(getBXShiftedALCT(aLCT));
  thisLCT.setCLCT(getBXShiftedCLCT(cLCT));
  thisLCT.setPattern(encodePattern(cLCT.getPattern()));
  thisLCT.setMPCLink(0);
  thisLCT.setBX0(0);
  thisLCT.setSyncErr(0);
  thisLCT.setCSCID(theTrigChamber);
  thisLCT.setTrknmb(trknmb);
  thisLCT.setWireGroup(aLCT.getKeyWG());
  thisLCT.setStrip(cLCT.getKeyStrip());
  thisLCT.setBend(cLCT.getBend());
  // Bunch crossing: get it from cathode LCT if anode LCT is not there.
  int bx = aLCT.isValid() ? aLCT.getBX() : cLCT.getBX();
  thisLCT.setBX(bx);
  thisLCT.setQuality(qualityAssignment_->findQuality(aLCT, cLCT));
  if (runCCLUT_) {
    thisLCT.setRun3(true);
    // 4-bit slope value derived with the CCLUT algorithm
    thisLCT.setSlope(cLCT.getSlope());
    thisLCT.setQuartStripBit(cLCT.getQuartStripBit());
    thisLCT.setEighthStripBit(cLCT.getEighthStripBit());
    thisLCT.setRun3Pattern(cLCT.getRun3Pattern());
  }
}

// CLCT pattern number: encodes the pattern number itself
unsigned int CSCMotherboard::encodePattern(const int ptn) const {
  const int kPatternBitWidth = 4;

  // In the TMB07 firmware, LCT pattern is just a 4-bit CLCT pattern.
  unsigned int pattern = (abs(ptn) & ((1 << kPatternBitWidth) - 1));

  return pattern;
}

void CSCMotherboard::selectLCTs() {
  // in each of the LCT time bins
  for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++) {
    unsigned nLCTs = 0;

    std::vector<CSCCorrelatedLCTDigi> tempV;
    // check each of the preferred combinations
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++) {
      // select at most 2
      for (int i = 0; i < CSCConstants::MAX_LCTS_PER_CSC; i++) {
        if (allLCTs_(bx, mbx, i).isValid() and nLCTs < 2) {
          tempV.push_back(allLCTs_(bx, mbx, i));
          ++nLCTs;
        }
      }
    }
    // store the best 2
    for (const auto& lct : tempV) {
      lctV.push_back(lct);
    }
  }

  // Show the pre-selected LCTs. They're not final yet. Some selection is done in the readoutLCTs function
  if (infoV > 0) {
    for (const auto& lct : lctV) {
      LogDebug("CSCMotherboard") << "Selected LCT" << lct;
    }
  }
}

void CSCMotherboard::checkConfigParameters() {
  // Make sure that the parameter values are within the allowed range.

  // Max expected values.
  static const unsigned int max_mpc_block_me1a = 1 << 1;
  static const unsigned int max_alct_trig_enable = 1 << 1;
  static const unsigned int max_clct_trig_enable = 1 << 1;
  static const unsigned int max_match_trig_enable = 1 << 1;
  static const unsigned int max_match_trig_window_size = 1 << 4;
  static const unsigned int max_tmb_l1a_window_size = 1 << 4;

  // Checks.
  CSCBaseboard::checkConfigParameters(mpc_block_me1a, max_mpc_block_me1a, def_mpc_block_me1a, "mpc_block_me1a");
  CSCBaseboard::checkConfigParameters(alct_trig_enable, max_alct_trig_enable, def_alct_trig_enable, "alct_trig_enable");
  CSCBaseboard::checkConfigParameters(clct_trig_enable, max_clct_trig_enable, def_clct_trig_enable, "clct_trig_enable");
  CSCBaseboard::checkConfigParameters(
      match_trig_enable, max_match_trig_enable, def_match_trig_enable, "match_trig_enable");
  CSCBaseboard::checkConfigParameters(
      match_trig_window_size, max_match_trig_window_size, def_match_trig_window_size, "match_trig_window_size");
  CSCBaseboard::checkConfigParameters(
      tmb_l1a_window_size, max_tmb_l1a_window_size, def_tmb_l1a_window_size, "tmb_l1a_window_size");
}

void CSCMotherboard::dumpConfigParams() const {
  std::ostringstream strm;
  strm << "\n";
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  strm << "+                   TMB configuration parameters:                  +\n";
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  strm << " mpc_block_me1a [block/not block triggers which come from ME1/A] = " << mpc_block_me1a << "\n";
  strm << " alct_trig_enable [allow ALCT-only triggers] = " << alct_trig_enable << "\n";
  strm << " clct_trig_enable [allow CLCT-only triggers] = " << clct_trig_enable << "\n";
  strm << " match_trig_enable [allow matched ALCT-CLCT triggers] = " << match_trig_enable << "\n";
  strm << " match_trig_window_size [ALCT-CLCT match window width, in 25 ns] = " << match_trig_window_size << "\n";
  strm << " tmb_l1a_window_size [L1Accept window width, in 25 ns bins] = " << tmb_l1a_window_size << "\n";
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  LogDebug("CSCMotherboard") << strm.str();
}

CSCALCTDigi CSCMotherboard::getBXShiftedALCT(const CSCALCTDigi& aLCT) const {
  CSCALCTDigi aLCT_shifted = aLCT;
  aLCT_shifted.setBX(aLCT_shifted.getBX() - (CSCConstants::LCT_CENTRAL_BX - CSCConstants::ALCT_CENTRAL_BX));
  return aLCT_shifted;
}

CSCCLCTDigi CSCMotherboard::getBXShiftedCLCT(const CSCCLCTDigi& cLCT) const {
  CSCCLCTDigi cLCT_shifted = cLCT;
  cLCT_shifted.setBX(cLCT_shifted.getBX() - CSCConstants::ALCT_CLCT_OFFSET);
  return cLCT_shifted;
}

void CSCMotherboard::matchShowers(CSCShowerDigi* anode_showers, CSCShowerDigi* cathode_showers, bool andlogic) {
  CSCShowerDigi ashower, cshower;
  bool used_cshower_mask[CSCConstants::MAX_CLCT_TBINS] = {false};
  for (unsigned bx = 0; bx < CSCConstants::MAX_ALCT_TBINS; bx++) {
    ashower = anode_showers[bx];
    cshower = CSCShowerDigi();  //use empty shower digi to initialize cshower
    if (ashower.isValid()) {
      for (unsigned mbx = 0; mbx < match_trig_window_size; mbx++) {
        int cbx = bx + preferred_bx_match_[mbx] - CSCConstants::ALCT_CLCT_OFFSET;
        //check bx range [0, CSCConstants::MAX_LCT_TBINS]
        if (cbx < 0 || cbx >= CSCConstants::MAX_CLCT_TBINS)
          continue;
        if (cathode_showers[cbx].isValid() and not used_cshower_mask[cbx]) {
          cshower = cathode_showers[cbx];
          used_cshower_mask[cbx] = true;
          break;
        }
      }
    } else
      cshower = cathode_showers[bx];  //if anode shower is not valid, use the cshower from this bx

    //matched HMT, with and/or logic
    unsigned matchHMT = 0;
    if (andlogic) {
      if (ashower.isTightInTime() and cshower.isTightInTime())
        matchHMT = 3;
      else if (ashower.isNominalInTime() and cshower.isNominalInTime())
        matchHMT = 2;
      else if (ashower.isLooseInTime() and cshower.isLooseInTime())
        matchHMT = 1;
    } else {
      if (ashower.isTightInTime() or cshower.isTightInTime())
        matchHMT = 3;
      else if (ashower.isNominalInTime() or cshower.isNominalInTime())
        matchHMT = 2;
      else if (ashower.isLooseInTime() or cshower.isLooseInTime())
        matchHMT = 1;
    }
    //LCTShower with showerType = 3
    showers_[bx] = CSCShowerDigi(matchHMT & 3,
                                 false,
                                 ashower.getCSCID(),
                                 bx,
                                 CSCShowerDigi::ShowerType::kLCTShower,
                                 ashower.getWireNHits(),
                                 cshower.getComparatorNHits());
  }
}

void CSCMotherboard::encodeHighMultiplicityBits() {
  // get the high multiplicity
  // for anode this reflects what is already in the anode CSCShowerDigi object
  CSCShowerDigi cathode_showers[CSCConstants::MAX_CLCT_TBINS];
  CSCShowerDigi anode_showers[CSCConstants::MAX_ALCT_TBINS];
  auto cshowers_v = clctProc->getAllShower();
  auto ashowers_v = alctProc->getAllShower();

  std::copy(cshowers_v.begin(), cshowers_v.end(), cathode_showers);
  std::copy(ashowers_v.begin(), ashowers_v.end(), anode_showers);

  // set the value according to source
  switch (thisShowerSource_) {
    case 0:
      std::copy(std::begin(cathode_showers), std::end(cathode_showers), std::begin(showers_));
      break;
    case 1:
      std::copy(std::begin(anode_showers), std::end(anode_showers), std::begin(showers_));
      break;
    case 2:
      matchShowers(anode_showers, cathode_showers, false);
      break;
    case 3:
      matchShowers(anode_showers, cathode_showers, true);
      break;
    default:
      std::copy(std::begin(anode_showers), std::end(anode_showers), std::begin(showers_));
      break;
  };
}
