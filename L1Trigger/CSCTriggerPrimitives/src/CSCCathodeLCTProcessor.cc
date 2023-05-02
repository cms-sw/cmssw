#include "L1Trigger/CSCTriggerPrimitives/interface/CSCCathodeLCTProcessor.h"

#include <iomanip>
#include <memory>

// Default values of configuration parameters.
const unsigned int CSCCathodeLCTProcessor::def_fifo_tbins = 12;
const unsigned int CSCCathodeLCTProcessor::def_fifo_pretrig = 7;
const unsigned int CSCCathodeLCTProcessor::def_hit_persist = 6;
const unsigned int CSCCathodeLCTProcessor::def_drift_delay = 2;
const unsigned int CSCCathodeLCTProcessor::def_nplanes_hit_pretrig = 2;
const unsigned int CSCCathodeLCTProcessor::def_nplanes_hit_pattern = 4;
const unsigned int CSCCathodeLCTProcessor::def_pid_thresh_pretrig = 2;
const unsigned int CSCCathodeLCTProcessor::def_min_separation = 10;
const unsigned int CSCCathodeLCTProcessor::def_tmb_l1a_window_size = 7;

//----------------
// Constructors --
//----------------

CSCCathodeLCTProcessor::CSCCathodeLCTProcessor(unsigned endcap,
                                               unsigned station,
                                               unsigned sector,
                                               unsigned subsector,
                                               unsigned chamber,
                                               CSCBaseboard::Parameters& conf)
    : CSCBaseboard(endcap, station, sector, subsector, chamber, conf) {
  static std::atomic<bool> config_dumped{false};

  // CLCT configuration parameters.
  fifo_tbins = conf.clctParams().getParameter<unsigned int>("clctFifoTbins");
  hit_persist = conf.clctParams().getParameter<unsigned int>("clctHitPersist");
  drift_delay = conf.clctParams().getParameter<unsigned int>("clctDriftDelay");
  nplanes_hit_pretrig = conf.clctParams().getParameter<unsigned int>("clctNplanesHitPretrig");
  nplanes_hit_pattern = conf.clctParams().getParameter<unsigned int>("clctNplanesHitPattern");

  // Not used yet.
  fifo_pretrig = conf.clctParams().getParameter<unsigned int>("clctFifoPretrig");

  pid_thresh_pretrig = conf.clctParams().getParameter<unsigned int>("clctPidThreshPretrig");
  min_separation = conf.clctParams().getParameter<unsigned int>("clctMinSeparation");

  start_bx_shift = conf.clctParams().getParameter<int>("clctStartBxShift");

  localShowerZone = conf.clctParams().getParameter<int>("clctLocalShowerZone");

  localShowerThresh = conf.clctParams().getParameter<int>("clctLocalShowerThresh");

  // Motherboard parameters: common for all configurations.
  tmb_l1a_window_size =  // Common to CLCT and TMB
      conf.tmbParams().getParameter<unsigned int>("tmbL1aWindowSize");

  /*
    In Summer 2021 the CLCT readout function was updated so that the
    window is based on a number of time bins around the central CLCT
    time BX7. In the past the window was based on early_tbins and late_tbins.
    The parameter is kept, but is not used.
  */
  early_tbins = conf.tmbParams().getParameter<int>("tmbEarlyTbins");
  if (early_tbins < 0)
    early_tbins = fifo_pretrig - CSCConstants::CLCT_EMUL_TIME_OFFSET;

  // wether to readout only the earliest two LCTs in readout window
  readout_earliest_2 = conf.tmbParams().getParameter<bool>("tmbReadoutEarliest2");

  // Verbosity level, set to 0 (no print) by default.
  infoV = conf.clctParams().getParameter<int>("verbosity");

  // Do not exclude pattern 0 and 1 when the Run-3 patterns are enabled!!
  // Valid Run-3 patterns are 0,1,2,3,4
  if (runCCLUT_) {
    pid_thresh_pretrig = 0;
  }

  // Check and print configuration parameters.
  checkConfigParameters();
  if ((infoV > 0) && !config_dumped) {
    dumpConfigParams();
    config_dumped = true;
  }

  numStrips_ = 0;  // Will be set later.
  // Provisional, but should be OK for all stations except ME1.
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    if ((i_layer + 1) % 2 == 0)
      stagger[i_layer] = 0;
    else
      stagger[i_layer] = 1;
  }

  for (int i = 0; i < CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER; ++i) {
    ispretrig_[i] = false;
  }

  // which patterns should we use?
  if (runCCLUT_) {
    clct_pattern_ = CSCPatternBank::clct_pattern_run3_;
    // comparator code lookup table algorithm for Phase-2
    cclut_ = std::make_unique<ComparatorCodeLUT>(conf.conf());
  } else {
    clct_pattern_ = CSCPatternBank::clct_pattern_legacy_;
  }

  const auto& shower = conf.showerParams().getParameterSet("cathodeShower");
  thresholds_ = shower.getParameter<std::vector<unsigned>>("showerThresholds");
  showerNumTBins_ = shower.getParameter<unsigned>("showerNumTBins");
  minLayersCentralTBin_ = shower.getParameter<unsigned>("minLayersCentralTBin");
  peakCheck_ = shower.getParameter<bool>("peakCheck");
  minbx_readout_ = CSCConstants::LCT_CENTRAL_BX - tmb_l1a_window_size / 2;
  maxbx_readout_ = CSCConstants::LCT_CENTRAL_BX + tmb_l1a_window_size / 2;
  assert(tmb_l1a_window_size / 2 <= CSCConstants::LCT_CENTRAL_BX);

  thePreTriggerDigis.clear();

  // quality control of stubs
  qualityControl_ = std::make_unique<LCTQualityControl>(endcap, station, sector, subsector, chamber, conf);
}

void CSCCathodeLCTProcessor::setDefaultConfigParameters() {
  // Set default values for configuration parameters.
  fifo_tbins = def_fifo_tbins;
  fifo_pretrig = def_fifo_pretrig;
  hit_persist = def_hit_persist;
  drift_delay = def_drift_delay;
  nplanes_hit_pretrig = def_nplanes_hit_pretrig;
  nplanes_hit_pattern = def_nplanes_hit_pattern;
  pid_thresh_pretrig = def_pid_thresh_pretrig;
  min_separation = def_min_separation;
  tmb_l1a_window_size = def_tmb_l1a_window_size;
  minbx_readout_ = CSCConstants::LCT_CENTRAL_BX - tmb_l1a_window_size / 2;
  maxbx_readout_ = CSCConstants::LCT_CENTRAL_BX + tmb_l1a_window_size / 2;
}

// Set configuration parameters obtained via EventSetup mechanism.
void CSCCathodeLCTProcessor::setConfigParameters(const CSCDBL1TPParameters* conf) {
  static std::atomic<bool> config_dumped{false};

  fifo_tbins = conf->clctFifoTbins();
  fifo_pretrig = conf->clctFifoPretrig();
  hit_persist = conf->clctHitPersist();
  drift_delay = conf->clctDriftDelay();
  nplanes_hit_pretrig = conf->clctNplanesHitPretrig();
  nplanes_hit_pattern = conf->clctNplanesHitPattern();
  pid_thresh_pretrig = conf->clctPidThreshPretrig();
  min_separation = conf->clctMinSeparation();

  // Check and print configuration parameters.
  checkConfigParameters();
  if (!config_dumped) {
    dumpConfigParams();
    config_dumped = true;
  }
  minbx_readout_ = CSCConstants::LCT_CENTRAL_BX - tmb_l1a_window_size / 2;
  maxbx_readout_ = CSCConstants::LCT_CENTRAL_BX + tmb_l1a_window_size / 2;
}

void CSCCathodeLCTProcessor::setESLookupTables(const CSCL1TPLookupTableCCLUT* conf) { cclut_->setESLookupTables(conf); }

void CSCCathodeLCTProcessor::checkConfigParameters() {
  // Make sure that the parameter values are within the allowed range.

  // Max expected values.
  static const unsigned int max_fifo_tbins = 1 << 5;
  static const unsigned int max_fifo_pretrig = 1 << 5;
  static const unsigned int max_hit_persist = 1 << 4;
  static const unsigned int max_drift_delay = 1 << 2;
  static const unsigned int max_nplanes_hit_pretrig = 1 << 3;
  static const unsigned int max_nplanes_hit_pattern = 1 << 3;
  static const unsigned int max_pid_thresh_pretrig = 1 << 4;
  static const unsigned int max_min_separation = CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER;
  static const unsigned int max_tmb_l1a_window_size = 1 << 4;

  // Checks.
  CSCBaseboard::checkConfigParameters(fifo_tbins, max_fifo_tbins, def_fifo_tbins, "fifo_tbins");
  CSCBaseboard::checkConfigParameters(fifo_pretrig, max_fifo_pretrig, def_fifo_pretrig, "fifo_pretrig");
  CSCBaseboard::checkConfigParameters(hit_persist, max_hit_persist, def_hit_persist, "hit_persist");
  CSCBaseboard::checkConfigParameters(drift_delay, max_drift_delay, def_drift_delay, "drift_delay");
  CSCBaseboard::checkConfigParameters(
      nplanes_hit_pretrig, max_nplanes_hit_pretrig, def_nplanes_hit_pretrig, "nplanes_hit_pretrig");
  CSCBaseboard::checkConfigParameters(
      nplanes_hit_pattern, max_nplanes_hit_pattern, def_nplanes_hit_pattern, "nplanes_hit_pattern");
  CSCBaseboard::checkConfigParameters(
      pid_thresh_pretrig, max_pid_thresh_pretrig, def_pid_thresh_pretrig, "pid_thresh_pretrig");
  CSCBaseboard::checkConfigParameters(min_separation, max_min_separation, def_min_separation, "min_separation");
  CSCBaseboard::checkConfigParameters(
      tmb_l1a_window_size, max_tmb_l1a_window_size, def_tmb_l1a_window_size, "tmb_l1a_window_size");
  assert(tmb_l1a_window_size / 2 <= CSCConstants::LCT_CENTRAL_BX);
}

void CSCCathodeLCTProcessor::clear() {
  thePreTriggerDigis.clear();
  thePreTriggerBXs.clear();
  for (int bx = 0; bx < CSCConstants::MAX_CLCT_TBINS; bx++) {
    bestCLCT[bx].clear();
    secondCLCT[bx].clear();
    cathode_showers_[bx].clear();
    localShowerFlag[bx] = false;  //init with no shower around CLCT
  }
}

std::vector<CSCCLCTDigi> CSCCathodeLCTProcessor::run(const CSCComparatorDigiCollection* compdc) {
  // This is the version of the run() function that is called when running
  // over the entire detector.  It gets the comparator & timing info from the
  // comparator digis and then passes them on to another run() function.

  static std::atomic<bool> config_dumped{false};
  if ((infoV > 0) && !config_dumped) {
    dumpConfigParams();
    config_dumped = true;
  }

  // Get the number of strips and stagger of layers for the given chamber.
  // Do it only once per chamber.
  if (numStrips_ <= 0 or numStrips_ > CSCConstants::MAX_NUM_STRIPS_RUN2) {
    if (cscChamber_) {
      numStrips_ = cscChamber_->layer(1)->geometry()->numberOfStrips();

      // ME1/a is known to the readout hardware as strips 65-80 of ME1/1.
      // Still need to decide whether we do any special adjustments to
      // reconstruct LCTs in this region (3:1 ganged strips); for now, we
      // simply allow for hits in ME1/a and apply standard reconstruction
      // to them.
      // For Phase2 ME1/1 is set to have 4 CFEBs in ME1/b and 3 CFEBs in ME1/a
      if (isME11_) {
        if (theRing == 4) {
          edm::LogError("CSCCathodeLCTProcessor|SetupError")
              << "+++ Invalid ring number for this processor " << theRing << " was set in the config."
              << " +++\n"
              << "+++ CSC geometry looks garbled; no emulation possible +++\n";
        }
        if (!disableME1a_ && theRing == 1 && !gangedME1a_)
          numStrips_ = CSCConstants::MAX_NUM_STRIPS_RUN2;
        if (!disableME1a_ && theRing == 1 && gangedME1a_)
          numStrips_ = CSCConstants::MAX_NUM_STRIPS_RUN1;
        if (disableME1a_ && theRing == 1)
          numStrips_ = CSCConstants::NUM_STRIPS_ME1B;
      }

      numHalfStrips_ = 2 * numStrips_ + 1;
      numCFEBs_ = numStrips_ / CSCConstants::NUM_STRIPS_PER_CFEB;

      if (numStrips_ > CSCConstants::MAX_NUM_STRIPS_RUN2) {
        edm::LogError("CSCCathodeLCTProcessor|SetupError")
            << "+++ Number of strips, " << numStrips_ << " found in " << theCSCName_ << " (sector " << theSector
            << " subsector " << theSubsector << " trig id. " << theTrigChamber << ")"
            << " exceeds max expected, " << CSCConstants::MAX_NUM_STRIPS_RUN2 << " +++\n"
            << "+++ CSC geometry looks garbled; no emulation possible +++\n";
        numStrips_ = -1;
        numHalfStrips_ = -1;
        numCFEBs_ = -1;
      }
      // The strips for a given layer may be offset from the adjacent layers.
      // This was done in order to improve resolution.  We need to find the
      // 'staggering' for each layer and make necessary conversions in our
      // arrays.  -JM
      // In the TMB-07 firmware, half-strips in odd layers (layers are
      // counted as ly0-ly5) are shifted by -1 half-strip, whereas in
      // the previous firmware versions half-strips in even layers
      // were shifted by +1 half-strip.  This difference is due to a
      // change from ly3 to ly2 in the choice of the key layer, and
      // the intention to keep half-strips in the key layer unchanged.
      // In the emulator, we use the old way for both cases, to avoid
      // negative half-strip numbers.  This will necessitate a
      // subtraction of 1 half-strip for TMB-07 later on. -SV.
      for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
        stagger[i_layer] = (cscChamber_->layer(i_layer + 1)->geometry()->stagger() + 1) / 2;
      }
    } else {
      edm::LogError("CSCCathodeLCTProcessor|ConfigError")
          << " " << theCSCName_ << " (sector " << theSector << " subsector " << theSubsector << " trig id. "
          << theTrigChamber << ")"
          << " is not defined in current geometry! +++\n"
          << "+++ CSC geometry looks garbled; no emulation possible +++\n";
      numStrips_ = -1;
      numHalfStrips_ = -1;
      numCFEBs_ = -1;
    }
  }

  if (numStrips_ <= 0 or 2 * (unsigned)numStrips_ > qualityControl_->get_csc_max_halfstrip(theStation, theRing)) {
    edm::LogError("CSCCathodeLCTProcessor|ConfigError")
        << " " << theCSCName_ << " (sector " << theSector << " subsector " << theSubsector << " trig id. "
        << theTrigChamber << "):"
        << " numStrips_ = " << numStrips_ << "; CLCT emulation skipped! +++";
    std::vector<CSCCLCTDigi> emptyV;
    return emptyV;
  }

  // Get comparator digis in this chamber.
  bool hasDigis = getDigis(compdc);

  if (hasDigis) {
    // Get halfstrip times from comparator digis.
    std::vector<int> halfStripTimes[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER];
    readComparatorDigis(halfStripTimes);

    // Pass arrays of halfstrips on to another run() doing the
    // LCT search.
    // If the number of layers containing digis is smaller than that
    // required to trigger, quit right away.  (If LCT-based digi suppression
    // is implemented one day, this condition will have to be changed
    // to the number of planes required to pre-trigger.)
    unsigned int layersHit = 0;
    for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
      for (int i_hstrip = 0; i_hstrip < CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER; i_hstrip++) {
        if (!halfStripTimes[i_layer][i_hstrip].empty()) {
          layersHit++;
          break;
        }
      }
    }
    // Run the algorithm only if the probability for the pre-trigger
    // to fire is not null.  (Pre-trigger decisions are used for the
    // strip read-out conditions in DigiToRaw.)
    if (layersHit >= nplanes_hit_pretrig)
      run(halfStripTimes);

    // Get the high multiplicity bits in this chamber
    encodeHighMultiplicityBits();
  }

  // Return vector of CLCTs.
  std::vector<CSCCLCTDigi> tmpV = getCLCTs();

  // shift the BX from 7 to 8
  // the unpacked real data CLCTs have central BX at bin 7
  // however in simulation the central BX  is bin 8
  // to make a proper comparison with ALCTs we need
  // CLCT and ALCT to have the central BX in the same bin
  // this shift does not affect the readout of the CLCTs
  // emulated CLCTs put in the event should be centered at bin 7 (as in data)
  for (auto& p : tmpV) {
    p.setBX(p.getBX() + CSCConstants::ALCT_CLCT_OFFSET);
  }

  return tmpV;
}

void CSCCathodeLCTProcessor::run(
    const std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER]) {
  // This version of the run() function can either be called in a standalone
  // test, being passed the halfstrip times, or called by the
  // run() function above.  It uses the findLCTs() method to find vectors
  // of LCT candidates. These candidates are already sorted and the best two per bx
  // are returned.

  // initialize the pulse array.
  // add 1 for possible stagger
  pulse_.initialize(numHalfStrips_ + 1);

  std::vector<CSCCLCTDigi> CLCTlist = findLCTs(halfstrip);

  for (const auto& p : CLCTlist) {
    const int bx = p.getBX();
    if (bx >= CSCConstants::MAX_CLCT_TBINS) {
      if (infoV > 0)
        edm::LogWarning("L1CSCTPEmulatorOutOfTimeCLCT")
            << "+++ Bx of CLCT candidate, " << bx << ", exceeds max allowed, " << CSCConstants::MAX_CLCT_TBINS - 1
            << "; skipping it... +++\n";
      continue;
    }

    if (!bestCLCT[bx].isValid()) {
      bestCLCT[bx] = p;
    } else if (!secondCLCT[bx].isValid()) {
      secondCLCT[bx] = p;
    }
  }

  for (int bx = 0; bx < CSCConstants::MAX_CLCT_TBINS; bx++) {
    if (bestCLCT[bx].isValid()) {
      bestCLCT[bx].setTrknmb(1);

      // check if the LCT is valid
      qualityControl_->checkValid(bestCLCT[bx]);

      if (infoV > 0)
        LogDebug("CSCCathodeLCTProcessor")
            << bestCLCT[bx] << " found in " << CSCDetId::chamberName(theEndcap, theStation, theRing, theChamber)
            << " (sector " << theSector << " subsector " << theSubsector << " trig id. " << theTrigChamber << ")"
            << "\n";
    }
    if (secondCLCT[bx].isValid()) {
      secondCLCT[bx].setTrknmb(2);

      // check if the LCT is valid
      qualityControl_->checkValid(secondCLCT[bx]);

      if (infoV > 0)
        LogDebug("CSCCathodeLCTProcessor")
            << secondCLCT[bx] << " found in " << CSCDetId::chamberName(theEndcap, theStation, theRing, theChamber)
            << " (sector " << theSector << " subsector " << theSubsector << " trig id. " << theTrigChamber << ")"
            << "\n";
    }
  }
  checkLocalShower(localShowerZone, halfstrip);
  // Now that we have our best CLCTs, they get correlated with the best
  // ALCTs and then get sent to the MotherBoard.  -JM
}

void CSCCathodeLCTProcessor::checkLocalShower(
    int zone,
    const std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER]) {
  // Fire half-strip one-shots for hit_persist bx's (4 bx's by default).
  //check local shower after pulse extension
  pulseExtension(halfstrip);

  for (int bx = 0; bx < CSCConstants::MAX_CLCT_TBINS; bx++) {
    if (not bestCLCT[bx].isValid())
      continue;

    //only check the region around best CLCT
    int keyHS = bestCLCT[bx].getKeyStrip();
    int minHS = (keyHS - zone) >= stagger[CSCConstants::KEY_CLCT_LAYER - 1] ? keyHS - zone
                                                                            : stagger[CSCConstants::KEY_CLCT_LAYER - 1];
    int maxHS = (keyHS + zone) >= numHalfStrips_ ? numHalfStrips_ : keyHS + zone;
    int totalHits = 0;
    for (int hstrip = minHS; hstrip < maxHS; hstrip++) {
      for (int this_layer = 0; this_layer < CSCConstants::NUM_LAYERS; this_layer++)
        if (pulse_.isOneShotHighAtBX(this_layer, hstrip, bx + drift_delay))
          totalHits++;
    }

    localShowerFlag[bx] = totalHits >= localShowerThresh;
    if (infoV > 1)
      LogDebug("CSCCathodeLCTProcessor") << " bx " << bx << " bestCLCT key HS " << keyHS
                                         << " localshower zone: " << minHS << ", " << maxHS << " totalHits "
                                         << totalHits
                                         << (localShowerFlag[bx] ? " Validlocalshower " : " NolocalShower ");
  }
}

bool CSCCathodeLCTProcessor::getDigis(const CSCComparatorDigiCollection* compdc) {
  bool hasDigis = false;

  // Loop over layers and save comparator digis on each one into digiV[layer].
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    digiV[i_layer].clear();

    CSCDetId detid(theEndcap, theStation, theRing, theChamber, i_layer + 1);
    getDigis(compdc, detid);

    if (isME11_ && !disableME1a_) {
      CSCDetId detid_me1a(theEndcap, theStation, 4, theChamber, i_layer + 1);
      getDigis(compdc, detid_me1a);
    }

    if (!digiV[i_layer].empty()) {
      hasDigis = true;
      if (infoV > 1) {
        LogTrace("CSCCathodeLCTProcessor") << "found " << digiV[i_layer].size() << " comparator digi(s) in layer "
                                           << i_layer << " of " << detid.chamberName() << " (trig. sector " << theSector
                                           << " subsector " << theSubsector << " id " << theTrigChamber << ")";
      }
    }
  }

  return hasDigis;
}

void CSCCathodeLCTProcessor::getDigis(const CSCComparatorDigiCollection* compdc, const CSCDetId& id) {
  const bool me1a = (id.station() == 1) && (id.ring() == 4);
  const CSCComparatorDigiCollection::Range rcompd = compdc->get(id);
  for (CSCComparatorDigiCollection::const_iterator digiIt = rcompd.first; digiIt != rcompd.second; ++digiIt) {
    const unsigned int origStrip = digiIt->getStrip();
    const unsigned int maxStripsME1a =
        gangedME1a_ ? CSCConstants::NUM_STRIPS_ME1A_GANGED : CSCConstants::NUM_STRIPS_ME1A_UNGANGED;
    // this special case can only be reached in MC
    // in real data, the comparator digis have always ring==1
    if (me1a && origStrip <= maxStripsME1a && !disableME1a_) {
      // Move ME1/A comparators from CFEB=0 to CFEB=4 if this has not
      // been done already.
      CSCComparatorDigi digi_corr(
          origStrip + CSCConstants::NUM_STRIPS_ME1B, digiIt->getComparator(), digiIt->getTimeBinWord());
      digiV[id.layer() - 1].push_back(digi_corr);
    } else {
      digiV[id.layer() - 1].push_back(*digiIt);
    }
  }
}

void CSCCathodeLCTProcessor::readComparatorDigis(
    std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER]) {
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    int i_digi = 0;  // digi counter, for dumps.
    for (std::vector<CSCComparatorDigi>::iterator pld = digiV[i_layer].begin(); pld != digiV[i_layer].end();
         pld++, i_digi++) {
      // Dump raw digi info.
      if (infoV > 1) {
        std::ostringstream strstrm;
        strstrm << "Comparator digi: comparator = " << pld->getComparator() << " strip #" << pld->getStrip()
                << " time bins on:";
        std::vector<int> bx_times = pld->getTimeBinsOn();
        for (unsigned int tbin = 0; tbin < bx_times.size(); tbin++)
          strstrm << " " << bx_times[tbin];
        LogTrace("CSCCathodeLCTProcessor") << strstrm.str();
      }

      // Get comparator: 0/1 for left/right halfstrip for each comparator
      // that fired.
      int thisComparator = pld->getComparator();
      if (thisComparator != 0 && thisComparator != 1) {
        if (infoV >= 0)
          edm::LogWarning("L1CSCTPEmulatorWrongInput")
              << "+++ station " << theStation << " ring " << theRing << " chamber " << theChamber
              << " Found comparator digi with wrong comparator value = " << thisComparator << "; skipping it... +++\n";
        continue;
      }

      // Get strip number.
      int thisStrip = pld->getStrip() - 1;  // count from 0
      if (thisStrip < 0 || thisStrip >= numStrips_) {
        if (infoV >= 0)
          edm::LogWarning("L1CSCTPEmulatorWrongInput")
              << "+++ station " << theStation << " ring " << theRing << " chamber " << theChamber
              << " Found comparator digi with wrong strip number = " << thisStrip << " (max strips = " << numStrips_
              << "); skipping it... +++\n";
        continue;
      }
      // 2*strip: convert strip to 1/2 strip
      // comp   : comparator output
      // stagger: stagger for this layer
      int thisHalfstrip = 2 * thisStrip + thisComparator + stagger[i_layer];
      if (thisHalfstrip >= numHalfStrips_) {
        if (infoV >= 0)
          edm::LogWarning("L1CSCTPEmulatorWrongInput")
              << "+++ station " << theStation << " ring " << theRing << " chamber " << theChamber
              << " Found wrong halfstrip number = " << thisHalfstrip << "; skipping this digi... +++\n";
        continue;
      }

      // Get bx times on this digi and check that they are within the bounds.
      std::vector<int> bx_times = pld->getTimeBinsOn();
      for (unsigned int i = 0; i < bx_times.size(); i++) {
        // Total number of time bins in DAQ readout is given by fifo_tbins,
        // which thus determines the maximum length of time interval.
        //
        // In data, only the CLCT in the time bin that was matched with L1A are read out
        // while comparator digi is read out by 12 time bin, which includes 12 time bin info
        // in other word, CLCTs emulated from comparator digis usually showed the OTMB behavior in 12 time bin
        // while CLCT from data only showed 1 time bin OTMB behavior
        // the CLCT emulated from comparator digis usually is centering at time bin 7 (BX7) and
        // it is definitly safe to ignore any CLCTs in bx 0 or 1 and those CLCTs will never impacts on any triggers
        if (bx_times[i] > 1 && bx_times[i] < static_cast<int>(fifo_tbins)) {
          if (i == 0 || (i > 0 && bx_times[i] - bx_times[i - 1] >= static_cast<int>(hit_persist))) {
            // A later hit on the same strip is ignored during the
            // number of clocks defined by the "hit_persist" parameter
            // (i.e., 6 bx's by default).
            if (infoV > 1)
              LogTrace("CSCCathodeLCTProcessor")
                  << "Comp digi: layer " << i_layer + 1 << " digi #" << i_digi + 1 << " strip " << thisStrip
                  << " halfstrip " << thisHalfstrip << " time " << bx_times[i] << " comparator " << thisComparator
                  << " stagger " << stagger[i_layer];
            halfstrip[i_layer][thisHalfstrip].push_back(bx_times[i]);
          } else if (i > 0) {
            if (infoV > 1)
              LogTrace("CSCCathodeLCTProcessor")
                  << "+++ station " << theStation << " ring " << theRing << " chamber " << theChamber
                  << " Skipping comparator digi: strip = " << thisStrip << ", layer = " << i_layer + 1
                  << ", bx = " << bx_times[i] << ", bx of previous hit = " << bx_times[i - 1];
          }
        } else {
          if (infoV > 1)
            LogTrace("CSCCathodeLCTProcessor") << "+++ station " << theStation << " ring " << theRing << " chamber "
                                               << theChamber << "+++ Skipping comparator digi: strip = " << thisStrip
                                               << ", layer = " << i_layer + 1 << ", bx = " << bx_times[i] << " +++";
        }
      }
    }
  }
}

// TMB-07 version.
std::vector<CSCCLCTDigi> CSCCathodeLCTProcessor::findLCTs(
    const std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER]) {
  std::vector<CSCCLCTDigi> lctList;

  if (infoV > 1)
    dumpDigis(halfstrip);

  // Fire half-strip one-shots for hit_persist bx's (4 bx's by default).
  pulseExtension(halfstrip);

  unsigned int start_bx = start_bx_shift;
  // Stop drift_delay bx's short of fifo_tbins since at later bx's we will
  // not have a full set of hits to start pattern search anyway.
  unsigned int stop_bx = fifo_tbins - drift_delay;
  // Allow for more than one pass over the hits in the time window.
  while (start_bx < stop_bx) {
    // temp CLCT objects
    CSCCLCTDigi tempBestCLCT;
    CSCCLCTDigi tempSecondCLCT;

    // All half-strip pattern envelopes are evaluated simultaneously, on every
    // clock cycle.
    int first_bx = 999;
    bool pre_trig = preTrigger(start_bx, first_bx);

    // If any of half-strip envelopes has enough layers hit in it, TMB
    // will pre-trigger.
    if (pre_trig) {
      thePreTriggerBXs.push_back(first_bx);
      if (infoV > 1)
        LogTrace("CSCCathodeLCTProcessor") << "..... pretrigger at bx = " << first_bx << "; waiting drift delay .....";

      // TMB latches LCTs drift_delay clocks after pretrigger.
      // in the configuration the drift_delay is set to 2bx by default
      // this is the time that is required for the electrons to drift to the
      // cathode strips. 15ns drift time --> 45 ns is 3 sigma for the delay
      // this corresponds to 2bx
      int latch_bx = first_bx + drift_delay;

      // define a new pattern map
      // for each key half strip, and for each pattern, store the 2D collection of fired comparator digis
      std::map<int, std::map<int, CSCCLCTDigi::ComparatorContainer>> hits_in_patterns;
      hits_in_patterns.clear();

      // We check if there is at least one key half strip for which at least
      // one pattern id has at least the minimum number of hits
      bool hits_in_time = patternFinding(latch_bx, hits_in_patterns);
      if (infoV > 1) {
        if (hits_in_time) {
          for (int hstrip = stagger[CSCConstants::KEY_CLCT_LAYER - 1]; hstrip < numHalfStrips_; hstrip++) {
            if (nhits[hstrip] > 0) {
              LogTrace("CSCCathodeLCTProcessor")
                  << " bx = " << std::setw(2) << latch_bx << " --->"
                  << " halfstrip = " << std::setw(3) << hstrip << " best pid = " << std::setw(2) << best_pid[hstrip]
                  << " nhits = " << nhits[hstrip];
            }
          }
        }
      }
      // This trigger emulator does not have an active CFEB flag for DAQ (csc trigger hardware: AFF)
      // This is a fundamental difference with the firmware where the TMB prepares the DAQ to
      // read out the chamber

      // The pattern finder runs continuously, so another pre-trigger
      // could occur already at the next bx.

      // Quality for sorting.
      int quality[CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER];
      int best_halfstrip[CSCConstants::MAX_CLCTS_PER_PROCESSOR];
      int best_quality[CSCConstants::MAX_CLCTS_PER_PROCESSOR];

      for (int ilct = 0; ilct < CSCConstants::MAX_CLCTS_PER_PROCESSOR; ilct++) {
        best_halfstrip[ilct] = -1;
        best_quality[ilct] = 0;
      }

      // Calculate quality from pattern id and number of hits, and
      // simultaneously select best-quality LCT.
      if (hits_in_time) {
        for (int hstrip = stagger[CSCConstants::KEY_CLCT_LAYER - 1]; hstrip < numHalfStrips_; hstrip++) {
          // The bend-direction bit pid[0] is ignored (left and right
          // bends have equal quality).
          quality[hstrip] = (best_pid[hstrip] & 14) | (nhits[hstrip] << 5);
          if (quality[hstrip] > best_quality[0]) {
            best_halfstrip[0] = hstrip;
            best_quality[0] = quality[hstrip];
          }
          // temporary alias
          const int best_hs(best_halfstrip[0]);
          // construct a CLCT if the trigger condition has been met
          if (best_hs >= 0 && nhits[best_hs] >= nplanes_hit_pattern) {
            // overwrite the current best CLCT
            tempBestCLCT = constructCLCT(first_bx, best_hs, hits_in_patterns[best_hs][best_pid[best_hs]]);
          }
        }
      }

      // If 1st best CLCT is found, look for the 2nd best.
      if (best_halfstrip[0] >= 0) {
        // Get the half-strip of the best CLCT in this BX that was put into the list.
        // You do need to re-add the any stagger, because the busy keys are based on
        // the pulse array which takes into account strip stagger!!!
        const unsigned halfStripBestCLCT(tempBestCLCT.getKeyStrip() + stagger[CSCConstants::KEY_CLCT_LAYER - 1]);

        // Mark keys near best CLCT as busy by setting their quality to
        // zero, and repeat the search.
        markBusyKeys(halfStripBestCLCT, best_pid[halfStripBestCLCT], quality);

        for (int hstrip = stagger[CSCConstants::KEY_CLCT_LAYER - 1]; hstrip < numHalfStrips_; hstrip++) {
          if (quality[hstrip] > best_quality[1]) {
            best_halfstrip[1] = hstrip;
            best_quality[1] = quality[hstrip];
          }
          // temporary alias
          const int best_hs(best_halfstrip[1]);
          // construct a CLCT if the trigger condition has been met
          if (best_hs >= 0 && nhits[best_hs] >= nplanes_hit_pattern) {
            // overwrite the current second best CLCT
            tempSecondCLCT = constructCLCT(first_bx, best_hs, hits_in_patterns[best_hs][best_pid[best_hs]]);
          }
        }

        // Sort bestCLCT and secondALCT by quality
        // if qualities are the same, sort by run-2 or run-3 pattern
        // if qualities and patterns are the same, sort by half strip number
        bool changeOrder = false;

        unsigned qualityBest = 0, qualitySecond = 0;
        unsigned patternBest = 0, patternSecond = 0;
        unsigned halfStripBest = 0, halfStripSecond = 0;

        if (tempBestCLCT.isValid() and tempSecondCLCT.isValid()) {
          qualityBest = tempBestCLCT.getQuality();
          qualitySecond = tempSecondCLCT.getQuality();
          if (!runCCLUT_) {
            patternBest = tempBestCLCT.getPattern();
            patternSecond = tempSecondCLCT.getPattern();
          } else {
            patternBest = tempBestCLCT.getRun3Pattern();
            patternSecond = tempSecondCLCT.getRun3Pattern();
          }
          halfStripBest = tempBestCLCT.getKeyStrip();
          halfStripSecond = tempSecondCLCT.getKeyStrip();

          if (qualitySecond > qualityBest)
            changeOrder = true;
          else if ((qualitySecond == qualityBest) and (int(patternSecond / 2) > int(patternBest / 2)))
            changeOrder = true;
          else if ((qualitySecond == qualityBest) and (int(patternSecond / 2) == int(patternBest / 2)) and
                   (halfStripSecond < halfStripBest))
            changeOrder = true;
        }

        CSCCLCTDigi tempCLCT;
        if (changeOrder) {
          tempCLCT = tempBestCLCT;
          tempBestCLCT = tempSecondCLCT;
          tempSecondCLCT = tempCLCT;
        }

        // add the CLCTs to the collection
        if (tempBestCLCT.isValid()) {
          lctList.push_back(tempBestCLCT);
        }
        if (tempSecondCLCT.isValid()) {
          lctList.push_back(tempSecondCLCT);
        }
      }  //find CLCT, end of best_halfstrip[0] >= 0

      // If there is a trigger, CLCT pre-trigger state machine
      // checks the number of hits that lie within a pattern template
      // at every bx, and waits for it to drop below threshold.
      // The search for CLCTs resumes only when the number of hits
      // drops below threshold.
      start_bx = fifo_tbins;
      // Stop checking drift_delay bx's short of fifo_tbins since
      // at later bx's we won't have a full set of hits for a
      // pattern search anyway.
      unsigned int stop_time = fifo_tbins - drift_delay;
      for (unsigned int bx = latch_bx + 1; bx < stop_time; bx++) {
        bool return_to_idle = true;
        bool hits_in_time = patternFinding(bx, hits_in_patterns);
        if (hits_in_time) {
          for (int hstrip = stagger[CSCConstants::KEY_CLCT_LAYER - 1]; hstrip < numHalfStrips_; hstrip++) {
            // the dead-time is done at the pre-trigger, not at the trigger
            if (nhits[hstrip] >= nplanes_hit_pretrig) {
              if (infoV > 1)
                LogTrace("CSCCathodeLCTProcessor") << " State machine busy at bx = " << bx;
              return_to_idle = false;
              break;
            }
          }
        }
        if (return_to_idle) {
          if (infoV > 1)
            LogTrace("CSCCathodeLCTProcessor") << " State machine returns to idle state at bx = " << bx;
          start_bx = bx;
          break;
        }
      }
    }  //pre_trig
    else {
      start_bx = first_bx + 1;  // no dead time
    }
  }

  return lctList;
}  // findLCTs -- TMB-07 version.

// Common to all versions.
void CSCCathodeLCTProcessor::pulseExtension(
    const std::vector<int> time[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER]) {
  const unsigned bits_in_pulse = pulse_.bitsInPulse();

  // Clear pulse array.  This array will be used as a bit representation of
  // hit times.  For example: if strip[1][2] has a value of 3, then 1 shifted
  // left 3 will be bit pattern of pulse[1][2].  This would make the pattern
  // look like 0000000000001000.  Then add on additional bits to signify
  // the duration of a signal (hit_persist, formerly bx_width) to simulate
  // the TMB's drift delay.  So for the same pulse[1][2] with a hit_persist
  // of 3 would look like 0000000000111000.  This is similating the digital
  // one-shot in the TMB.
  pulse_.clear();

  // Loop over all layers and halfstrips.
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    for (int i_strip = 0; i_strip < numHalfStrips_; i_strip++) {
      // If there is a hit, simulate digital one-shot persistence starting
      // in the bx of the initial hit.  Fill this into pulse[][].
      if (!time[i_layer][i_strip].empty()) {
        std::vector<int> bx_times = time[i_layer][i_strip];
        for (unsigned int i = 0; i < bx_times.size(); i++) {
          // Check that min and max times are within the allowed range.
          if (bx_times[i] < 0 || bx_times[i] + hit_persist >= bits_in_pulse) {
            if (infoV > 0)
              edm::LogWarning("L1CSCTPEmulatorOutOfTimeDigi")
                  << "+++ BX time of comparator digi (halfstrip = " << i_strip << " layer = " << i_layer
                  << ") bx = " << bx_times[i] << " is not within the range (0-" << bits_in_pulse
                  << "] allowed for pulse extension.  Skip this digi! +++\n";
            continue;
          }
          if (bx_times[i] >= start_bx_shift) {
            pulse_.extend(i_layer, i_strip, bx_times[i], hit_persist);
          }
        }
      }
    }
  }
}  // pulseExtension.

// TMB-07 version.
bool CSCCathodeLCTProcessor::preTrigger(const int start_bx, int& first_bx) {
  if (infoV > 1)
    LogTrace("CSCCathodeLCTProcessor") << "....................PreTrigger...........................";

  int nPreTriggers = 0;

  bool pre_trig = false;
  // Now do a loop over bx times to see (if/when) track goes over threshold
  for (unsigned int bx_time = start_bx; bx_time < fifo_tbins; bx_time++) {
    // For any given bunch-crossing, start at the lowest keystrip and look for
    // the number of separate layers in the pattern for that keystrip that have
    // pulses at that bunch-crossing time.  Do the same for the next keystrip,
    // etc.  Then do the entire process again for the next bunch-crossing, etc
    // until you find a pre-trigger.

    std::map<int, std::map<int, CSCCLCTDigi::ComparatorContainer>> hits_in_patterns;
    hits_in_patterns.clear();

    bool hits_in_time = patternFinding(bx_time, hits_in_patterns);
    if (hits_in_time) {
      // clear the pretriggers
      clearPreTriggers();

      for (int hstrip = stagger[CSCConstants::KEY_CLCT_LAYER - 1]; hstrip < numHalfStrips_; hstrip++) {
        // check the properties of the pattern on this halfstrip
        if (infoV > 1) {
          if (nhits[hstrip] > 0) {
            LogTrace("CSCCathodeLCTProcessor")
                << " bx = " << std::setw(2) << bx_time << " --->"
                << " halfstrip = " << std::setw(3) << hstrip << " best pid = " << std::setw(2) << best_pid[hstrip]
                << " nhits = " << nhits[hstrip];
          }
        }
        // a pretrigger was found
        if (nhits[hstrip] >= nplanes_hit_pretrig && best_pid[hstrip] >= pid_thresh_pretrig) {
          pre_trig = true;
          ispretrig_[hstrip] = true;

          // write each pre-trigger to output
          nPreTriggers++;
          thePreTriggerDigis.push_back(constructPreCLCT(bx_time, hstrip, nPreTriggers));
        }
      }

      // upon the first pretrigger, we save first BX and exit
      if (pre_trig) {
        first_bx = bx_time;  // bx at time of pretrigger
        return true;
      }
    }
  }  // end loop over bx times

  if (infoV > 1)
    LogTrace("CSCCathodeLCTProcessor") << "no pretrigger, returning \n";
  first_bx = fifo_tbins;
  return false;
}  // preTrigger -- TMB-07 version.

// TMB-07 version.
bool CSCCathodeLCTProcessor::patternFinding(
    const unsigned int bx_time, std::map<int, std::map<int, CSCCLCTDigi::ComparatorContainer>>& hits_in_patterns) {
  if (bx_time >= fifo_tbins)
    return false;

  unsigned layers_hit = pulse_.numberOfLayersAtBX(bx_time);
  if (layers_hit < nplanes_hit_pretrig)
    return false;

  for (int key_hstrip = 0; key_hstrip < numHalfStrips_; key_hstrip++) {
    best_pid[key_hstrip] = 0;
    nhits[key_hstrip] = 0;
  }

  bool hit_layer[CSCConstants::NUM_LAYERS];

  // Loop over candidate key strips.
  for (int key_hstrip = stagger[CSCConstants::KEY_CLCT_LAYER - 1]; key_hstrip < numHalfStrips_; key_hstrip++) {
    // Loop over patterns and look for hits matching each pattern.
    for (unsigned int pid = clct_pattern_.size() - 1; pid >= pid_thresh_pretrig and pid < clct_pattern_.size(); pid--) {
      layers_hit = 0;
      // clear all layers
      for (int ilayer = 0; ilayer < CSCConstants::NUM_LAYERS; ilayer++) {
        hit_layer[ilayer] = false;
      }

      // clear a single pattern!
      CSCCLCTDigi::ComparatorContainer hits_single_pattern;
      hits_single_pattern.resize(6);
      for (auto& p : hits_single_pattern) {
        p.resize(CSCConstants::CLCT_PATTERN_WIDTH, CSCConstants::INVALID_HALF_STRIP);
      }

      // clear all medians
      double num_pattern_hits = 0., times_sum = 0.;
      std::multiset<int> mset_for_median;
      mset_for_median.clear();

      // Loop over halfstrips in trigger pattern mask and calculate the
      // "absolute" halfstrip number for each.
      for (int this_layer = 0; this_layer < CSCConstants::NUM_LAYERS; this_layer++) {
        for (int strip_num = 0; strip_num < CSCConstants::CLCT_PATTERN_WIDTH; strip_num++) {
          // ignore "0" half-strips in the pattern
          if (clct_pattern_[pid][this_layer][strip_num] == 0)
            continue;

          // the current strip is the key half-strip plus the offset (can be negative or positive)
          int this_strip = CSCPatternBank::clct_pattern_offset_[strip_num] + key_hstrip;

          // current strip should be valid of course
          if (this_strip >= 0 && this_strip < numHalfStrips_) {
            if (infoV > 3) {
              LogTrace("CSCCathodeLCTProcessor") << " In patternFinding: key_strip = " << key_hstrip << " pid = " << pid
                                                 << " layer = " << this_layer << " strip = " << this_strip << std::endl;
            }
            // Determine if "one shot" is high at this bx_time
            if (pulse_.isOneShotHighAtBX(this_layer, this_strip, bx_time)) {
              if (hit_layer[this_layer] == false) {
                hit_layer[this_layer] = true;
                layers_hit++;  // determines number of layers hit
                // add this strip in this layer to the pattern we are currently considering
                hits_single_pattern[this_layer][strip_num] = this_strip - stagger[this_layer];
              }

              // find at what bx did pulse on this halsfstrip & layer have started
              // use hit_persist constraint on how far back we can go
              int first_bx_layer = bx_time;
              for (unsigned int dbx = 0; dbx < hit_persist; dbx++) {
                if (pulse_.isOneShotHighAtBX(this_layer, this_strip, first_bx_layer - 1))
                  first_bx_layer--;
                else
                  break;
              }
              times_sum += (double)first_bx_layer;
              num_pattern_hits += 1.;
              mset_for_median.insert(first_bx_layer);
              if (infoV > 2)
                LogTrace("CSCCathodeLCTProcessor") << " 1st bx in layer: " << first_bx_layer << " sum bx: " << times_sum
                                                   << " #pat. hits: " << num_pattern_hits;
            }
          }
        }  // end loop over strips in pretrigger pattern
      }    // end loop over layers

      // save the pattern information when a trigger was formed!
      if (layers_hit >= nplanes_hit_pattern) {
        hits_in_patterns[key_hstrip][pid] = hits_single_pattern;
      }

      // determine the current best pattern!
      if (layers_hit > nhits[key_hstrip]) {
        best_pid[key_hstrip] = pid;
        nhits[key_hstrip] = layers_hit;
        // Do not loop over the other (worse) patterns if max. numbers of
        // hits is found.
        if (nhits[key_hstrip] == CSCConstants::NUM_LAYERS)
          break;
      }
    }  // end loop over pid
  }    // end loop over candidate key strips

  // At this point there exists at least one halfstrip for which at least one pattern
  // has at least 3 layers --> definition of a pre-trigger
  return true;
}  // patternFinding -- TMB-07 version.

// TMB-07 version.
void CSCCathodeLCTProcessor::markBusyKeys(const int best_hstrip,
                                          const int best_patid,
                                          int quality[CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER]) {
  int nspan = min_separation;
  int pspan = min_separation;

  for (int hstrip = best_hstrip - nspan; hstrip <= best_hstrip + pspan; hstrip++) {
    if (hstrip >= 0 && hstrip < CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER) {
      quality[hstrip] = 0;
    }
  }
}  // markBusyKeys -- TMB-07 version.

CSCCLCTDigi CSCCathodeLCTProcessor::constructCLCT(const int bx,
                                                  const unsigned halfstrip_withstagger,
                                                  const CSCCLCTDigi::ComparatorContainer& hits) {
  // Assign the CLCT properties
  const unsigned quality = nhits[halfstrip_withstagger];
  const unsigned pattern = best_pid[halfstrip_withstagger];
  const unsigned bend = CSCPatternBank::getPatternBend(clct_pattern_[pattern]);
  const unsigned keyhalfstrip = halfstrip_withstagger - stagger[CSCConstants::KEY_CLCT_LAYER - 1];
  const unsigned cfeb = keyhalfstrip / CSCConstants::NUM_HALF_STRIPS_PER_CFEB;
  const unsigned halfstrip = keyhalfstrip % CSCConstants::NUM_HALF_STRIPS_PER_CFEB;

  // set the Run-2 properties
  CSCCLCTDigi clct(1,
                   quality,
                   pattern,
                   // CLCTs are always of type halfstrip (not strip or distrip)
                   1,
                   bend,
                   halfstrip,
                   cfeb,
                   bx,
                   0,
                   0,
                   -1,
                   CSCCLCTDigi::Version::Legacy);

  // set the hit collection
  clct.setHits(hits);

  // do the CCLUT procedures for Run-3
  if (runCCLUT_) {
    cclut_->run(clct, numCFEBs_);
  }

  // purge the comparator digi collection from the obsolete "65535" entries...
  cleanComparatorContainer(clct);

  if (infoV > 1) {
    LogTrace("CSCCathodeLCTProcessor") << "Produce CLCT " << clct << std::endl;
  }

  return clct;
}

CSCCLCTPreTriggerDigi CSCCathodeLCTProcessor::constructPreCLCT(const int bx_time,
                                                               const unsigned hstrip,
                                                               const unsigned nPreTriggers) const {
  const int bend = clct_pattern_[best_pid[hstrip]][CSCConstants::NUM_LAYERS - 1][CSCConstants::CLCT_PATTERN_WIDTH];
  const int halfstrip = hstrip % CSCConstants::NUM_HALF_STRIPS_PER_CFEB;
  const int cfeb = hstrip / CSCConstants::NUM_HALF_STRIPS_PER_CFEB;
  return CSCCLCTPreTriggerDigi(1, nhits[hstrip], best_pid[hstrip], 1, bend, halfstrip, cfeb, bx_time, nPreTriggers, 0);
}

void CSCCathodeLCTProcessor::clearPreTriggers() {
  for (int hstrip = stagger[CSCConstants::KEY_CLCT_LAYER - 1]; hstrip < numHalfStrips_; hstrip++) {
    ispretrig_[hstrip] = false;
  }
}

void CSCCathodeLCTProcessor::cleanComparatorContainer(CSCCLCTDigi& clct) const {
  CSCCLCTDigi::ComparatorContainer newHits = clct.getHits();
  for (auto& p : newHits) {
    p.erase(
        std::remove_if(p.begin(), p.end(), [](unsigned i) -> bool { return i == CSCConstants::INVALID_HALF_STRIP; }),
        p.end());
  }
  clct.setHits(newHits);
}

// --------------------------------------------------------------------------
// Auxiliary code.
// --------------------------------------------------------------------------
// Dump of configuration parameters.
void CSCCathodeLCTProcessor::dumpConfigParams() const {
  std::ostringstream strm;
  strm << "\n";
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  strm << "+                  CLCT configuration parameters:                  +\n";
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  strm << " fifo_tbins   [total number of time bins in DAQ readout] = " << fifo_tbins << "\n";
  strm << " fifo_pretrig [start time of cathode raw hits in DAQ readout] = " << fifo_pretrig << "\n";
  strm << " hit_persist  [duration of signal pulse, in 25 ns bins] = " << hit_persist << "\n";
  strm << " drift_delay  [time after pre-trigger before TMB latches LCTs] = " << drift_delay << "\n";
  strm << " nplanes_hit_pretrig [min. number of layers hit for pre-trigger] = " << nplanes_hit_pretrig << "\n";
  strm << " nplanes_hit_pattern [min. number of layers hit for trigger] = " << nplanes_hit_pattern << "\n";
  strm << " pid_thresh_pretrig [lower threshold on pattern id] = " << pid_thresh_pretrig << "\n";
  strm << " min_separation     [region of busy key strips] = " << min_separation << "\n";
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  LogDebug("CSCCathodeLCTProcessor") << strm.str();
}

// Reasonably nice dump of digis on half-strips.
void CSCCathodeLCTProcessor::dumpDigis(
    const std::vector<int> strip[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER]) const {
  LogDebug("CSCCathodeLCTProcessor") << theCSCName_ << " strip type: half-strip,  numHalfStrips " << numHalfStrips_;

  std::ostringstream strstrm;
  for (int i_strip = 0; i_strip < numHalfStrips_; i_strip++) {
    if (i_strip % 10 == 0) {
      if (i_strip < 100)
        strstrm << i_strip / 10;
      else
        strstrm << (i_strip - 100) / 10;
    } else
      strstrm << " ";
    if ((i_strip + 1) % CSCConstants::NUM_HALF_STRIPS_PER_CFEB == 0)
      strstrm << " ";
  }
  strstrm << "\n";
  for (int i_strip = 0; i_strip < numHalfStrips_; i_strip++) {
    strstrm << i_strip % 10;
    if ((i_strip + 1) % CSCConstants::NUM_HALF_STRIPS_PER_CFEB == 0)
      strstrm << " ";
  }
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    strstrm << "\n";
    for (int i_strip = 0; i_strip < numHalfStrips_; i_strip++) {
      if (!strip[i_layer][i_strip].empty()) {
        std::vector<int> bx_times = strip[i_layer][i_strip];
        // Dump only the first in time.
        strstrm << std::hex << bx_times[0] << std::dec;
      } else {
        strstrm << "-";
      }
      if ((i_strip + 1) % CSCConstants::NUM_HALF_STRIPS_PER_CFEB == 0)
        strstrm << " ";
    }
  }
  LogTrace("CSCCathodeLCTProcessor") << strstrm.str();
}

// Returns vector of read-out CLCTs, if any.  Starts with the vector
// of all found CLCTs and selects the ones in the read-out time window.
std::vector<CSCCLCTDigi> CSCCathodeLCTProcessor::readoutCLCTs() const {
  // temporary container for further selection
  std::vector<CSCCLCTDigi> tmpV;

  /*
    CLCTs in the BX window [early_tbin,...,late_tbin] are considered good for physics
    The central CLCT BX is time bin 7.
    For tmb_l1a_window_size set to 7 (Run-1, Run-2), the window is [4, 5, 6, 7, 8, 9, 10]
    For tmb_l1a_window_size set to 5 (Run-3), the window is [5, 6, 7, 8, 9]
    For tmb_l1a_window_size set to 3 (Run-4?), the window is [6, 7, 8]
  */
  const unsigned delta_tbin = tmb_l1a_window_size / 2;
  int early_tbin = CSCConstants::CLCT_CENTRAL_BX - delta_tbin;
  int late_tbin = CSCConstants::CLCT_CENTRAL_BX + delta_tbin;
  /*
     Special case for an even-numbered time-window,
     For instance tmb_l1a_window_size set to 6: [4, 5, 6, 7, 8, 9]
  */
  if (tmb_l1a_window_size % 2 == 0)
    late_tbin = CSCConstants::CLCT_CENTRAL_BX + delta_tbin - 1;
  const int max_late_tbin = CSCConstants::MAX_CLCT_TBINS - 1;

  // debugging messages when early_tbin or late_tbin has a suspicious value
  if (early_tbin < 0) {
    edm::LogWarning("CSCCathodeLCTProcessor|SuspiciousParameters")
        << "Early time bin (early_tbin) smaller than minimum allowed, which is 0. set early_tbin to 0.";
    early_tbin = 0;
  }
  if (late_tbin > max_late_tbin) {
    edm::LogWarning("CSCCathodeLCTProcessor|SuspiciousParameters")
        << "Late time bin (late_tbin) larger than maximum allowed, which is " << max_late_tbin
        << ". set early_tbin to max allowed";
    late_tbin = CSCConstants::MAX_CLCT_TBINS - 1;
  }

  // get the valid LCTs. No BX selection is done here
  const auto& all_clcts = getCLCTs();

  // Start from the vector of all found CLCTs and select those within
  // the CLCT*L1A coincidence window.
  int bx_readout = -1;
  for (const auto& clct : all_clcts) {
    // only consider valid CLCTs
    if (!clct.isValid())
      continue;

    const int bx = clct.getBX();
    // Skip CLCTs found too early relative to L1Accept.
    if (bx < early_tbin) {
      if (infoV > 1)
        LogDebug("CSCCathodeLCTProcessor")
            << " Do not report correlated CLCT on key halfstrip " << clct.getStrip() << ": found at bx " << bx
            << ", whereas the earliest allowed bx is " << early_tbin;
      continue;
    }

    // Skip CLCTs found too late relative to L1Accept.
    if (bx > late_tbin) {
      if (infoV > 1)
        LogDebug("CSCCathodeLCTProcessor")
            << " Do not report correlated CLCT on key halfstrip " << clct.getStrip() << ": found at bx " << bx
            << ", whereas the latest allowed bx is " << late_tbin;
      continue;
    }

    // If (readout_earliest_2) take only CLCTs in the earliest bx in the read-out window:
    if (readout_earliest_2) {
      // the first CLCT passes
      // the second CLCT passes if the BX matches to the first
      if (bx_readout == -1 || bx == bx_readout) {
        tmpV.push_back(clct);
        if (bx_readout == -1)
          bx_readout = bx;
      }
    } else
      tmpV.push_back(clct);
  }

  // do a final check on the CLCTs in readout
  qualityControl_->checkMultiplicityBX(tmpV);
  for (const auto& clct : tmpV) {
    qualityControl_->checkValid(clct);
  }

  return tmpV;
}

// Returns vector of all found CLCTs, if any.  Used for ALCT-CLCT matching.
std::vector<CSCCLCTDigi> CSCCathodeLCTProcessor::getCLCTs() const {
  std::vector<CSCCLCTDigi> tmpV;
  for (int bx = 0; bx < CSCConstants::MAX_CLCT_TBINS; bx++) {
    if (bestCLCT[bx].isValid())
      tmpV.push_back(bestCLCT[bx]);
    if (secondCLCT[bx].isValid())
      tmpV.push_back(secondCLCT[bx]);
  }
  return tmpV;
}

// shift the BX from 7 to 8
// the unpacked real data CLCTs have central BX at bin 7
// however in simulation the central BX  is bin 8
// to make a proper comparison with ALCTs we need
// CLCT and ALCT to have the central BX in the same bin
CSCCLCTDigi CSCCathodeLCTProcessor::getBestCLCT(int bx) const {
  if (bx >= CSCConstants::MAX_CLCT_TBINS or bx < 0)
    return CSCCLCTDigi();
  CSCCLCTDigi lct = bestCLCT[bx];
  lct.setBX(lct.getBX() + CSCConstants::ALCT_CLCT_OFFSET);
  return lct;
}

CSCCLCTDigi CSCCathodeLCTProcessor::getSecondCLCT(int bx) const {
  if (bx >= CSCConstants::MAX_CLCT_TBINS or bx < 0)
    return CSCCLCTDigi();
  CSCCLCTDigi lct = secondCLCT[bx];
  lct.setBX(lct.getBX() + CSCConstants::ALCT_CLCT_OFFSET);
  return lct;
}

bool CSCCathodeLCTProcessor::getLocalShowerFlag(int bx) const {
  if (bx >= CSCConstants::MAX_CLCT_TBINS or bx < 0)
    return false;
  return localShowerFlag[bx];
}

/** return vector of CSCShower digi **/
std::vector<CSCShowerDigi> CSCCathodeLCTProcessor::getAllShower() const {
  std::vector<CSCShowerDigi> vshowers(cathode_showers_, cathode_showers_ + CSCConstants::MAX_CLCT_TBINS);
  return vshowers;
};

/** Returns shower bits */
std::vector<CSCShowerDigi> CSCCathodeLCTProcessor::readoutShower() const {
  std::vector<CSCShowerDigi> showerOut;
  for (unsigned bx = minbx_readout_; bx < maxbx_readout_; bx++)
    if (cathode_showers_[bx].isValid())
      showerOut.push_back(cathode_showers_[bx]);
  return showerOut;
}

void CSCCathodeLCTProcessor::encodeHighMultiplicityBits() {
  //inTimeHMT_ = 0;

  //numer of layer with hits and number of hits for 0-15 BXs
  std::set<unsigned> layersWithHits[CSCConstants::MAX_CLCT_TBINS];
  unsigned hitsInTime[CSCConstants::MAX_CLCT_TBINS];
  // Calculate layers with hits
  for (unsigned bx = 0; bx < CSCConstants::MAX_CLCT_TBINS; bx++) {
    hitsInTime[bx] = 0;
    for (unsigned i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
      bool atLeastOneCompHit = false;
      for (const auto& compdigi : digiV[i_layer]) {
        std::vector<int> bx_times = compdigi.getTimeBinsOn();
        // there is at least one comparator digi in this bx
        if (std::find(bx_times.begin(), bx_times.end(), bx) != bx_times.end()) {
          hitsInTime[bx] += 1;
          atLeastOneCompHit = true;
        }
      }
      // add this layer to the number of layers hit
      if (atLeastOneCompHit) {
        layersWithHits[bx].insert(i_layer);
      }
    }
  }  //end of full bx loop

  // convert station and ring number to index
  // index runs from 2 to 10, subtract 2
  unsigned csc_idx = CSCDetId::iChamberType(theStation, theRing) - 2;

  // loose, nominal and tight
  std::vector<unsigned> station_thresholds = {
      thresholds_[csc_idx * 3], thresholds_[csc_idx * 3 + 1], thresholds_[csc_idx * 3 + 2]};

  //hard coded dead time as 2Bx, since showerNumTBins = 3, like firmware
  // for example, nhits = 0 at bx7; = 100 at bx8; = 0 at bx9
  //cathode HMT must be triggered at bx8, not bx7 and bx9
  //meanwhile we forced 2BX dead time after active shower trigger
  unsigned int deadtime =
      showerNumTBins_ - 1;  // firmware hard coded dead time as 2Bx, since showerNumTBins = 3 in firmware
  unsigned int dead_count = 0;

  for (unsigned bx = 0; bx < CSCConstants::MAX_CLCT_TBINS; bx++) {
    unsigned minbx = bx >= showerNumTBins_ / 2 ? bx - showerNumTBins_ / 2 : bx;
    unsigned maxbx = bx < CSCConstants::MAX_CLCT_TBINS - showerNumTBins_ / 2 ? bx + showerNumTBins_ / 2
                                                                             : CSCConstants::MAX_CLCT_TBINS - 1;
    unsigned this_hitsInTime = 0;
    bool isPeak = true;  //check whether total hits in bx is peak of nhits over time bins
    /*following is to count number of hits over [minbx, maxbx], showerNumTBins=3 =>[n-1, n+1]*/
    for (unsigned mbx = minbx; mbx <= maxbx; mbx++) {
      this_hitsInTime += hitsInTime[mbx];
    }

    if (peakCheck_ and bx < CSCConstants::MAX_CLCT_TBINS - showerNumTBins_ / 2 - 1) {
      if (hitsInTime[minbx] < hitsInTime[maxbx + 1] or
          (hitsInTime[minbx] == hitsInTime[maxbx + 1] and hitsInTime[bx] < hitsInTime[bx + 1]))
        isPeak = false;  //next bx would have more hits or in the center
    }
    bool dead_status = dead_count > 0;
    if (dead_status)
      dead_count--;

    unsigned this_inTimeHMT = 0;
    // require at least nLayersWithHits for the central time bin
    // do nothing if there are not enough layers with hits
    if (layersWithHits[bx].size() >= minLayersCentralTBin_ and !dead_status and isPeak) {
      // assign the bits
      if (!station_thresholds.empty()) {
        for (int i = station_thresholds.size() - 1; i >= 0; i--) {
          if (this_hitsInTime >= station_thresholds[i]) {
            this_inTimeHMT = i + 1;
            dead_count = deadtime;
            break;
          }
        }
      }
    }
    //CLCTshower constructor with showerType_ = 2, wirehits = 0;
    cathode_showers_[bx] = CSCShowerDigi(
        this_inTimeHMT, false, theTrigChamber, bx, CSCShowerDigi::ShowerType::kCLCTShower, 0, this_hitsInTime);
  }
}
