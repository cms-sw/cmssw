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
  preferred_bx_match_ = tmbParams_.getParameter<std::vector<int> >("preferredBxMatch");

  // quality assignment
  qualityAssignment_ = std::make_unique<LCTQualityAssignment>(station);

  // quality control of stubs
  qualityControl_ = std::make_unique<LCTQualityControl>(endcap, station, sector, subsector, chamber, conf);

  // shower-trigger source
  showerSource_ = showerParams_.getParameter<unsigned>("source");

  // enable the upgrade processors for ring 1 stations
  if (runPhase2_ and theRing == 1) {
    clctProc = std::make_unique<CSCUpgradeCathodeLCTProcessor>(endcap, station, sector, subsector, chamber, conf);
    if (enableAlctPhase2_) {
      alctProc = std::make_unique<CSCUpgradeAnodeLCTProcessor>(endcap, station, sector, subsector, chamber, conf);
    }
  }

  // set up helper class to check if ALCT and CLCT cross
  const bool ignoreAlctCrossClct = tmbParams_.getParameter<bool>("ignoreAlctCrossClct");
  cscOverlap_ = std::make_unique<CSCALCTCrossCLCT>(endcap, station, theRing, ignoreAlctCrossClct, conf);
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

  // reset the shower trigger
  shower_.clear();
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

  // Step 2: Run the processors
  alctV = alctProc->run(wiredc);  // run anodeLCT
  clctV = clctProc->run(compdc);  // run cathodeLCT

  // Step 2b: encode high multiplicity bits (independent of LCT construction)
  encodeHighMultiplicityBits();

  // if there are no ALCTs and no CLCTs, it does not make sense to run this TMB
  if (alctV.empty() and clctV.empty())
    return;

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
        unsigned bx_clct = bx_alct + preferred_bx_match_[mbx] - CSCConstants::ALCT_CLCT_OFFSET;
        // check that the CLCT BX is valid
        if (bx_clct >= CSCConstants::MAX_CLCT_TBINS)
          continue;
        // do not consider previously matched CLCTs
        if (drop_used_clcts && used_clct_mask[bx_clct])
          continue;
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

  // Step 4: Select at most 2 LCTs per BX
  selectLCTs();
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
  bool debugTimeBins = true;
  if (debugTimeBins) {
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
    debugTimeBins = false;
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
  }

  return tmpV;
}

CSCShowerDigi CSCMotherboard::readoutShower() const { return shower_; }

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

  // check which ALCTs and CLCTs are valid
  copyValidToInValid(bestALCT, secondALCT, bestCLCT, secondCLCT);

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
                       cscOverlap_->doesALCTCrossCLCT(bestALCT, bestCLCT));

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
                         cscOverlap_->doesALCTCrossCLCT(secondALCT, secondCLCT));

  // at least one component must be different in order to consider the secondLCT
  if ((secondALCT != bestALCT) or (secondCLCT != bestCLCT)) {
    // at least one of the cases must be valid
    if (secondCase1 or secondCase2 or secondCase3) {
      constructLCTs(secondALCT, secondCLCT, type, 2, sLCT);
    }
  }
}

void CSCMotherboard::copyValidToInValid(CSCALCTDigi& bestALCT,
                                        CSCALCTDigi& secondALCT,
                                        CSCCLCTDigi& bestCLCT,
                                        CSCCLCTDigi& secondCLCT) const {
  // check which ALCTs and CLCTs are valid
  const bool anodeBestValid = bestALCT.isValid();
  const bool anodeSecondValid = secondALCT.isValid();
  const bool cathodeBestValid = bestCLCT.isValid();
  const bool cathodeSecondValid = secondCLCT.isValid();

  // copy the valid ALCT/CLCT information to the valid ALCT/CLCT
  if (anodeBestValid && !anodeSecondValid)
    secondALCT = bestALCT;
  if (!anodeBestValid && anodeSecondValid)
    bestALCT = secondALCT;
  if (cathodeBestValid && !cathodeSecondValid)
    secondCLCT = bestCLCT;
  if (!cathodeBestValid && cathodeSecondValid)
    bestCLCT = secondCLCT;
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
  thisLCT.setQuality(qualityAssignment_->findQuality(aLCT, cLCT, runCCLUT_));

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
  aLCT_shifted.setBX(aLCT_shifted.getBX() - (CSCConstants::LCT_CENTRAL_BX - tmb_l1a_window_size / 2));
  return aLCT_shifted;
}

CSCCLCTDigi CSCMotherboard::getBXShiftedCLCT(const CSCCLCTDigi& cLCT) const {
  CSCCLCTDigi cLCT_shifted = cLCT;
  cLCT_shifted.setBX(cLCT_shifted.getBX() - CSCConstants::ALCT_CLCT_OFFSET);
  return cLCT_shifted;
}

void CSCMotherboard::encodeHighMultiplicityBits() {
  // get the high multiplicity
  // for anode this reflects what is already in the anode CSCShowerDigi object
  unsigned cathodeInTime = clctProc->getInTimeHMT();
  unsigned cathodeOutTime = clctProc->getOutTimeHMT();
  unsigned anodeInTime = alctProc->getInTimeHMT();
  unsigned anodeOutTime = alctProc->getOutTimeHMT();

  // assign the bits
  unsigned inTimeHMT_;
  unsigned outTimeHMT_;

  // set the value according to source
  switch (showerSource_) {
    case 0:
      inTimeHMT_ = cathodeInTime;
      outTimeHMT_ = cathodeOutTime;
      break;
    case 1:
      inTimeHMT_ = anodeInTime;
      outTimeHMT_ = anodeOutTime;
      break;
    case 2:
      inTimeHMT_ = anodeInTime | cathodeInTime;
      outTimeHMT_ = anodeOutTime | cathodeOutTime;
      break;
    default:
      inTimeHMT_ = cathodeInTime;
      outTimeHMT_ = cathodeOutTime;
      break;
  };

  // create a new object
  shower_ = CSCShowerDigi(inTimeHMT_, outTimeHMT_, theTrigChamber);
}
