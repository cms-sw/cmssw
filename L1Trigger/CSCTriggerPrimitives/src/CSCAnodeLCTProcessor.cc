#include "L1Trigger/CSCTriggerPrimitives/interface/CSCAnodeLCTProcessor.h"
#include <set>

// Default values of configuration parameters.
const unsigned int CSCAnodeLCTProcessor::def_fifo_tbins = 16;
const unsigned int CSCAnodeLCTProcessor::def_fifo_pretrig = 10;
const unsigned int CSCAnodeLCTProcessor::def_drift_delay = 2;
const unsigned int CSCAnodeLCTProcessor::def_nplanes_hit_pretrig = 2;
const unsigned int CSCAnodeLCTProcessor::def_nplanes_hit_pattern = 4;
const unsigned int CSCAnodeLCTProcessor::def_nplanes_hit_accel_pretrig = 2;
const unsigned int CSCAnodeLCTProcessor::def_nplanes_hit_accel_pattern = 4;
const unsigned int CSCAnodeLCTProcessor::def_trig_mode = 2;         // 3?
const unsigned int CSCAnodeLCTProcessor::def_accel_mode = 0;        // 1?
const unsigned int CSCAnodeLCTProcessor::def_l1a_window_width = 7;  // 5?

//----------------
// Constructors --
//----------------

CSCAnodeLCTProcessor::CSCAnodeLCTProcessor(unsigned endcap,
                                           unsigned station,
                                           unsigned sector,
                                           unsigned subsector,
                                           unsigned chamber,
                                           const edm::ParameterSet& conf)
    : CSCBaseboard(endcap, station, sector, subsector, chamber, conf) {
  static std::atomic<bool> config_dumped{false};

  // ALCT configuration parameters.
  fifo_tbins = alctParams_.getParameter<unsigned int>("alctFifoTbins");
  fifo_pretrig = alctParams_.getParameter<unsigned int>("alctFifoPretrig");
  drift_delay = alctParams_.getParameter<unsigned int>("alctDriftDelay");
  nplanes_hit_pretrig = alctParams_.getParameter<unsigned int>("alctNplanesHitPretrig");
  nplanes_hit_pattern = alctParams_.getParameter<unsigned int>("alctNplanesHitPattern");
  nplanes_hit_accel_pretrig = alctParams_.getParameter<unsigned int>("alctNplanesHitAccelPretrig");
  nplanes_hit_accel_pattern = alctParams_.getParameter<unsigned int>("alctNplanesHitAccelPattern");
  trig_mode = alctParams_.getParameter<unsigned int>("alctTrigMode");
  accel_mode = alctParams_.getParameter<unsigned int>("alctAccelMode");
  l1a_window_width = alctParams_.getParameter<unsigned int>("alctL1aWindowWidth");

  hit_persist = alctParams_.getParameter<unsigned int>("alctHitPersist");

  // Verbosity level, set to 0 (no print) by default.
  infoV = alctParams_.getParameter<int>("verbosity");

  // separate handle for early time bins
  early_tbins = alctParams_.getParameter<int>("alctEarlyTbins");
  if (early_tbins < 0)
    early_tbins = fifo_pretrig - CSCConstants::ALCT_EMUL_TIME_OFFSET;

  // delta BX time depth for ghostCancellationLogic
  ghost_cancellation_bx_depth = alctParams_.getParameter<int>("alctGhostCancellationBxDepth");

  // whether to consider ALCT candidates' qualities while doing ghostCancellationLogic on +-1 wire groups
  ghost_cancellation_side_quality = alctParams_.getParameter<bool>("alctGhostCancellationSideQuality");

  // deadtime clocks after pretrigger (extra in addition to drift_delay)
  pretrig_extra_deadtime = alctParams_.getParameter<unsigned int>("alctPretrigDeadtime");

  // whether to use narrow pattern mask for the rings close to the beam
  narrow_mask_r1 = alctParams_.getParameter<bool>("alctNarrowMaskForR1");

  // Check and print configuration parameters.
  checkConfigParameters();
  if ((infoV > 0 || (runPhase2_)) && !config_dumped) {
    dumpConfigParams();
    config_dumped = true;
  }

  numWireGroups = 0;  // Will be set later.
  MESelection = (theStation < 3) ? 0 : 1;

  // whether to calculate bx as corrected_bx instead of pretrigger one
  use_corrected_bx = false;
  if (runPhase2_) {
    use_corrected_bx = alctParams_.getParameter<bool>("alctUseCorrectedBx");
  }

  // Load appropriate pattern mask.
  loadPatternMask();

  // quality control of stubs
  qualityControl_ = std::make_unique<LCTQualityControl>(endcap, station, sector, subsector, chamber, conf);

  const auto& shower = showerParams_.getParameterSet("anodeShower");
  thresholds_ = shower.getParameter<std::vector<unsigned>>("showerThresholds");
  showerMinInTBin_ = shower.getParameter<unsigned>("showerMinInTBin");
  showerMaxInTBin_ = shower.getParameter<unsigned>("showerMaxInTBin");
  showerMinOutTBin_ = shower.getParameter<unsigned>("showerMinOutTBin");
  showerMaxOutTBin_ = shower.getParameter<unsigned>("showerMaxOutTBin");
}

void CSCAnodeLCTProcessor::loadPatternMask() {
  // Load appropriate pattern mask.
  if (narrow_mask_r1 && (theRing == 1 || theRing == 4)) {
    alct_pattern_ = CSCPatternBank::alct_pattern_r1_;
  } else {
    alct_pattern_ = CSCPatternBank::alct_pattern_legacy_;
  }
}

void CSCAnodeLCTProcessor::setDefaultConfigParameters() {
  // Set default values for configuration parameters.
  fifo_tbins = def_fifo_tbins;
  fifo_pretrig = def_fifo_pretrig;
  drift_delay = def_drift_delay;
  nplanes_hit_pretrig = def_nplanes_hit_pretrig;
  nplanes_hit_pattern = def_nplanes_hit_pattern;
  nplanes_hit_accel_pretrig = def_nplanes_hit_accel_pretrig;
  nplanes_hit_accel_pattern = def_nplanes_hit_accel_pattern;
  trig_mode = def_trig_mode;
  accel_mode = def_accel_mode;
  l1a_window_width = def_l1a_window_width;
}

// Set configuration parameters obtained via EventSetup mechanism.
void CSCAnodeLCTProcessor::setConfigParameters(const CSCDBL1TPParameters* conf) {
  static std::atomic<bool> config_dumped{false};

  fifo_tbins = conf->alctFifoTbins();
  fifo_pretrig = conf->alctFifoPretrig();
  drift_delay = conf->alctDriftDelay();
  nplanes_hit_pretrig = conf->alctNplanesHitPretrig();
  nplanes_hit_pattern = conf->alctNplanesHitPattern();
  nplanes_hit_accel_pretrig = conf->alctNplanesHitAccelPretrig();
  nplanes_hit_accel_pattern = conf->alctNplanesHitAccelPattern();
  trig_mode = conf->alctTrigMode();
  accel_mode = conf->alctAccelMode();
  l1a_window_width = conf->alctL1aWindowWidth();

  // Check and print configuration parameters.
  checkConfigParameters();
  if (!config_dumped) {
    dumpConfigParams();
    config_dumped = true;
  }
}

void CSCAnodeLCTProcessor::checkConfigParameters() {
  // Make sure that the parameter values are within the allowed range.

  // Max expected values.
  static const unsigned int max_fifo_tbins = 1 << 5;
  static const unsigned int max_fifo_pretrig = 1 << 5;
  static const unsigned int max_drift_delay = 1 << 2;
  static const unsigned int max_nplanes_hit_pretrig = 1 << 3;
  static const unsigned int max_nplanes_hit_pattern = 1 << 3;
  static const unsigned int max_nplanes_hit_accel_pretrig = 1 << 3;
  static const unsigned int max_nplanes_hit_accel_pattern = 1 << 3;
  static const unsigned int max_trig_mode = 1 << 2;
  static const unsigned int max_accel_mode = 1 << 2;
  static const unsigned int max_l1a_window_width = CSCConstants::MAX_ALCT_TBINS;  // 4 bits

  // Checks.
  CSCBaseboard::checkConfigParameters(fifo_tbins, max_fifo_tbins, def_fifo_tbins, "fifo_tbins");
  CSCBaseboard::checkConfigParameters(fifo_pretrig, max_fifo_pretrig, def_fifo_pretrig, "fifo_pretrig");
  CSCBaseboard::checkConfigParameters(drift_delay, max_drift_delay, def_drift_delay, "drift_delay");
  CSCBaseboard::checkConfigParameters(
      nplanes_hit_pretrig, max_nplanes_hit_pretrig, def_nplanes_hit_pretrig, "nplanes_hit_pretrig");
  CSCBaseboard::checkConfigParameters(
      nplanes_hit_pattern, max_nplanes_hit_pattern, def_nplanes_hit_pattern, "nplanes_hit_pattern");
  CSCBaseboard::checkConfigParameters(nplanes_hit_accel_pretrig,
                                      max_nplanes_hit_accel_pretrig,
                                      def_nplanes_hit_accel_pretrig,
                                      "nplanes_hit_accel_pretrig");
  CSCBaseboard::checkConfigParameters(nplanes_hit_accel_pattern,
                                      max_nplanes_hit_accel_pattern,
                                      def_nplanes_hit_accel_pattern,
                                      "nplanes_hit_accel_pattern");
  CSCBaseboard::checkConfigParameters(trig_mode, max_trig_mode, def_trig_mode, "trig_mode");
  CSCBaseboard::checkConfigParameters(accel_mode, max_accel_mode, def_accel_mode, "accel_mode");
  CSCBaseboard::checkConfigParameters(l1a_window_width, max_l1a_window_width, def_l1a_window_width, "l1a_window_width");
}

void CSCAnodeLCTProcessor::clear() {
  for (int bx = 0; bx < CSCConstants::MAX_ALCT_TBINS; bx++) {
    bestALCT[bx].clear();
    secondALCT[bx].clear();
  }
  lct_list.clear();
  inTimeHMT_ = 0;
  outTimeHMT_ = 0;
}

void CSCAnodeLCTProcessor::clear(const int wire, const int pattern) {
  /* Clear the data off of selected pattern */
  if (pattern == CSCConstants::ALCT_ACCELERATOR_PATTERN)
    quality[wire][CSCConstants::ALCT_ACCELERATOR_PATTERN] = -999;
  else {
    quality[wire][CSCConstants::ALCT_COLLISIONA_PATTERN] = -999;
    quality[wire][CSCConstants::ALCT_COLLISIONB_PATTERN] = -999;
  }
}

std::vector<CSCALCTDigi> CSCAnodeLCTProcessor::run(const CSCWireDigiCollection* wiredc) {
  static std::atomic<bool> config_dumped{false};
  if ((infoV > 0 || (runPhase2_)) && !config_dumped) {
    dumpConfigParams();
    config_dumped = true;
  }

  // Get the number of wire groups for the given chamber.  Do it only once
  // per chamber.
  if (numWireGroups <= 0 or numWireGroups > CSCConstants::MAX_NUM_WIREGROUPS) {
    if (cscChamber_) {
      numWireGroups = cscChamber_->layer(1)->geometry()->numberOfWireGroups();
      if (numWireGroups > CSCConstants::MAX_NUM_WIREGROUPS) {
        edm::LogError("CSCAnodeLCTProcessor|SetupError")
            << "+++ Number of wire groups, " << numWireGroups << " found in " << theCSCName_ << " (sector " << theSector
            << " subsector " << theSubsector << " trig id. " << theTrigChamber << ")"
            << " exceeds max expected, " << CSCConstants::MAX_NUM_WIREGROUPS << " +++\n"
            << "+++ CSC geometry looks garbled; no emulation possible +++\n";
        numWireGroups = -1;
      }
    } else {
      edm::LogError("CSCAnodeLCTProcessor|SetupError")
          << "+++ " << theCSCName_ << " (sector " << theSector << " subsector " << theSubsector << " trig id. "
          << theTrigChamber << ")"
          << " is not defined in current geometry! +++\n"
          << "+++ CSC geometry looks garbled; no emulation possible +++\n";
      numWireGroups = -1;
    }
  }

  if (numWireGroups <= 0 or (unsigned) numWireGroups > qualityControl_->get_csc_max_wiregroup(theStation, theRing)) {
    edm::LogError("CSCAnodeLCTProcessor|SetupError")
        << "+++ " << theCSCName_ << " (sector " << theSector << " subsector " << theSubsector << " trig id. "
        << theTrigChamber << "):"
        << " numWireGroups = " << numWireGroups << "; ALCT emulation skipped! +++";
    std::vector<CSCALCTDigi> emptyV;
    return emptyV;
  }

  // Get wire digis in this chamber from wire digi collection.
  bool hasDigis = getDigis(wiredc);

  if (hasDigis) {
    // First get wiregroup times from the wire digis.
    std::vector<int> wireGroupTimes[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIREGROUPS];
    readWireDigis(wireGroupTimes);

    // Pass an array of wire times on to another run() doing the LCT search.
    // If the number of layers containing digis is smaller than that
    // required to trigger, quit right away.
    const unsigned int min_layers =
        (nplanes_hit_accel_pattern == 0)
            ? nplanes_hit_pattern
            : ((nplanes_hit_pattern <= nplanes_hit_accel_pattern) ? nplanes_hit_pattern : nplanes_hit_accel_pattern);

    unsigned int layersHit = 0;
    for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
      for (int i_wire = 0; i_wire < numWireGroups; i_wire++) {
        if (!wireGroupTimes[i_layer][i_wire].empty()) {
          layersHit++;
          break;
        }
      }
    }
    if (layersHit >= min_layers)
      run(wireGroupTimes);
    // Get the high multiplicity bits in this chamber
    encodeHighMultiplicityBits(wireGroupTimes);
  }

  // Return vector of all found ALCTs.
  return getALCTs();
}

void CSCAnodeLCTProcessor::run(const std::vector<int> wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIREGROUPS]) {
  bool trigger = false;

  // initialize the pulse array.
  pulse_.initialize(numWireGroups);

  // Check if there are any in-time hits and do the pulse extension.
  bool chamber_empty = pulseExtension(wire);

  // define a new pattern map
  // for each key wiregroup, and for each pattern, store the 2D collection of fired wiregroup digis
  std::map<int, std::map<int, CSCALCTDigi::WireContainer>> hits_in_patterns;
  hits_in_patterns.clear();

  // Only do the rest of the processing if chamber is not empty.
  // Stop drift_delay bx's short of fifo_tbins since at later bx's we will
  // not have a full set of hits to start pattern search anyway.
  unsigned int stop_bx = fifo_tbins - drift_delay;
  if (!chamber_empty) {
    for (int i_wire = 0; i_wire < numWireGroups; i_wire++) {
      // extra check to make sure only valid wires are processed
      const unsigned max_wire = qualityControl_->get_csc_max_wiregroup(theStation, theRing);
      if (unsigned(i_wire) >= max_wire)
        continue;

      unsigned int start_bx = 0;
      // Allow for more than one pass over the hits in the time window.
      while (start_bx < stop_bx) {
        if (preTrigger(i_wire, start_bx)) {
          if (infoV > 2)
            showPatterns(i_wire);
          if (patternDetection(i_wire, hits_in_patterns)) {
            trigger = true;
            int ghost_cleared[2] = {0, 0};
            /*
              In older versions of the ALCT emulation, the ghost cancellation was performed after
              the ALCTs were found. In December 2018, it became clear that during the study of data
              and emulation comparison on 2018 data, a small disagreement between data and emulation
              was found. The changes we implemented then allow re-triggering on one wiregroup after
              some dead time once an earlier ALCT was constructed built on this wiregroup. Before this
              commit the ALCT processor would prohibit the wiregroup from triggering in one event after
              an ALCT was found on that wiregroup. In the firwmare, the wiregroup with ALCT is only dead
              for a few BX before it can be triggered by next muon. The implementation of ghost cancellation
              logic was changed to accommodate the re-triggering change while the idea of ghost cancellation
              logic is kept the same.
            */
            ghostCancellationLogicOneWire(i_wire, ghost_cleared);

            int bx = (use_corrected_bx) ? first_bx_corrected[i_wire] : first_bx[i_wire];
            if (bx >= CSCConstants::MAX_ALCT_TBINS)
              edm::LogError("CSCAnodeLCTProcessor")
                  << " bx of valid trigger : " << bx << " > max allowed value " << CSCConstants::MAX_ALCT_TBINS;

            //acceleration mode
            if (quality[i_wire][0] > 0 and bx < CSCConstants::MAX_ALCT_TBINS) {
              int valid = (ghost_cleared[0] == 0) ? 1 : 0;  //cancelled, valid=0, otherwise it is 1
              CSCALCTDigi newALCT(valid, quality[i_wire][0], 1, 0, i_wire, bx);

              // set the wire digis for this pattern
              setWireContainer(newALCT, hits_in_patterns[i_wire][0]);

              lct_list.emplace_back(newALCT);
              if (infoV > 1)
                LogTrace("CSCAnodeLCTProcessor") << "Add one ALCT to list " << lct_list.back();
            }

            //collision mode
            if (quality[i_wire][CSCConstants::ALCT_COLLISIONA_PATTERN] > 0 and bx < CSCConstants::MAX_ALCT_TBINS) {
              int valid = (ghost_cleared[CSCConstants::ALCT_COLLISIONA_PATTERN] == 0)
                              ? 1
                              : 0;  //cancelled, valid=0, otherwise it is 1

              CSCALCTDigi newALCT(valid,
                                  quality[i_wire][CSCConstants::ALCT_COLLISIONA_PATTERN],
                                  0,
                                  quality[i_wire][CSCConstants::ALCT_COLLISIONB_PATTERN],
                                  i_wire,
                                  bx);

              // set the wire digis for this pattern
              setWireContainer(newALCT, hits_in_patterns[i_wire][CSCConstants::ALCT_COLLISIONA_PATTERN]);

              lct_list.emplace_back(newALCT);
              if (infoV > 1)
                LogTrace("CSCAnodeLCTProcessor") << "Add one ALCT to list " << lct_list.back();
            }

            //break;
            // Assume that the earliest time when another pre-trigger can
            // occur in case pattern detection failed is bx_pretrigger+4:
            // this seems to match the data.
            start_bx = first_bx[i_wire] + drift_delay + pretrig_extra_deadtime;
          } else {
            //only pretrigger, no trigger ==> no dead time, continue to find next pretrigger
            start_bx = first_bx[i_wire] + 1;
          }
        } else {  //no pretrigger, skip this wiregroup
          break;
        }
      }  // end of while
    }
  }

  // Do the rest only if there is at least one trigger candidate.
  if (trigger) {
    /* In Run-1 and Run-2, the ghost cancellation was done after the trigger.
       In the firmware however, the ghost cancellation is done during the trigger
       on each wiregroup in parallel. For Run-3 and beyond, the ghost cancellation is
       implemented per wiregroup earlier in the code. See function
       "ghostCancellationLogicOneWire". There used to be a function ghostCancellationLogic
       call here.
    */
    lctSearch();
  }
}

bool CSCAnodeLCTProcessor::getDigis(const CSCWireDigiCollection* wiredc) {
  // Routine for getting digis and filling digiV vector.
  bool hasDigis = false;

  // Loop over layers and save wire digis on each one into digiV[layer].
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    digiV[i_layer].clear();

    CSCDetId detid(theEndcap, theStation, theRing, theChamber, i_layer + 1);
    getDigis(wiredc, detid);

    // If this is ME1/1, fetch digis in corresponding ME1/A (ring=4) as well.
    if (isME11_ && !disableME1a_) {
      CSCDetId detid_me1a(theEndcap, theStation, 4, theChamber, i_layer + 1);
      getDigis(wiredc, detid_me1a);
    }

    if (!digiV[i_layer].empty()) {
      hasDigis = true;
      if (infoV > 1) {
        LogTrace("CSCAnodeLCTProcessor") << "found " << digiV[i_layer].size() << " wire digi(s) in layer " << i_layer
                                         << " of " << theCSCName_ << " (trig. sector " << theSector << " subsector "
                                         << theSubsector << " id " << theTrigChamber << ")";
        for (const auto& wd : digiV[i_layer]) {
          LogTrace("CSCAnodeLCTProcessor") << "   " << wd;
        }
      }
    }
  }

  return hasDigis;
}

void CSCAnodeLCTProcessor::getDigis(const CSCWireDigiCollection* wiredc, const CSCDetId& id) {
  CSCWireDigiCollection::Range rwired = wiredc->get(id);
  for (CSCWireDigiCollection::const_iterator digiIt = rwired.first; digiIt != rwired.second; ++digiIt) {
    digiV[id.layer() - 1].push_back(*digiIt);
  }
}

void CSCAnodeLCTProcessor::readWireDigis(
    std::vector<int> wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIREGROUPS]) {
  // Loop over all 6 layers.
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    // Loop over all digis in the layer and find the wireGroup and bx
    // time for each.
    for (const auto& wd : digiV[i_layer]) {
      int i_wire = wd.getWireGroup() - 1;
      std::vector<int> bx_times = wd.getTimeBinsOn();

      // Check that the wires and times are appropriate.
      if (i_wire < 0 || i_wire >= numWireGroups) {
        if (infoV >= 0)
          edm::LogWarning("CSCAnodeLCTProcessor|WrongInput")
              << "+++ Found wire digi with wrong wire number = " << i_wire << " (max wires = " << numWireGroups
              << "); skipping it... +++\n";
        continue;
      }
      // Accept digis in expected time window.  Total number of time
      // bins in DAQ readout is given by fifo_tbins, which thus
      // determines the maximum length of time interval.  Anode raw
      // hits in DAQ readout start (fifo_pretrig - 6) clocks before
      // L1Accept.  If times earlier than L1Accept were recorded, we
      // use them since they can modify the ALCTs found later, via
      // ghost-cancellation logic.
      int last_time = -999;
      if (bx_times.size() == fifo_tbins) {
        wire[i_layer][i_wire].push_back(0);
        wire[i_layer][i_wire].push_back(6);
      } else {
        for (unsigned int i = 0; i < bx_times.size(); i++) {
          // Find rising edge change
          if (i > 0 && bx_times[i] == (bx_times[i - 1] + 1))
            continue;
          if (bx_times[i] < static_cast<int>(fifo_tbins)) {
            if (infoV > 2)
              LogTrace("CSCAnodeLCTProcessor")
                  << "Digi on layer " << i_layer << " wire " << i_wire << " at time " << bx_times[i];

            // Finally save times of hit wires.  One shot module will
            // not restart if a new pulse comes before the expiration
            // of the 6-bx period.
            if (last_time < 0 || ((bx_times[i] - last_time) >= 6)) {
              wire[i_layer][i_wire].push_back(bx_times[i]);
              last_time = bx_times[i];
            }
          } else {
            if (infoV > 1)
              LogTrace("CSCAnodeLCTProcessor") << "+++ Skipping wire digi: wire = " << i_wire << " layer = " << i_layer
                                               << ", bx = " << bx_times[i] << " +++";
          }
        }
      }
    }
  }
}

bool CSCAnodeLCTProcessor::pulseExtension(
    const std::vector<int> wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIREGROUPS]) {
  bool chamber_empty = true;
  int i_wire, i_layer, digi_num;

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

  for (i_wire = 0; i_wire < numWireGroups; i_wire++) {
    first_bx[i_wire] = -999;
    first_bx_corrected[i_wire] = -999;
    for (int j = 0; j < 3; j++)
      quality[i_wire][j] = -999;
  }

  for (i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    digi_num = 0;
    for (i_wire = 0; i_wire < numWireGroups; i_wire++) {
      if (!wire[i_layer][i_wire].empty()) {
        std::vector<int> bx_times = wire[i_layer][i_wire];
        for (unsigned int i = 0; i < bx_times.size(); i++) {
          // Check that min and max times are within the allowed range.
          if (bx_times[i] < 0 || bx_times[i] + hit_persist >= bits_in_pulse) {
            if (infoV > 0)
              edm::LogWarning("CSCAnodeLCTProcessor|OutOfTimeDigi")
                  << "+++ BX time of wire digi (wire = " << i_wire << " layer = " << i_layer << ") bx = " << bx_times[i]
                  << " is not within the range (0-" << bits_in_pulse
                  << "] allowed for pulse extension.  Skip this digi! +++\n";
            continue;
          }

          // Found at least one in-time digi; set chamber_empty to false
          if (chamber_empty)
            chamber_empty = false;

          // make the pulse
          pulse_.extend(i_layer, i_wire, bx_times[i], hit_persist);

          // Debug information.
          if (infoV > 1) {
            LogTrace("CSCAnodeLCTProcessor") << "Wire digi: layer " << i_layer << " digi #" << ++digi_num
                                             << " wire group " << i_wire << " time " << bx_times[i];
            if (infoV > 2) {
              std::ostringstream strstrm;
              for (int i = 1; i <= 32; i++) {
                strstrm << pulse_.oneShotAtBX(i_layer, i_wire, 32 - i);
              }
              LogTrace("CSCAnodeLCTProcessor") << "  Pulse: " << strstrm.str();
            }
          }
        }
      }
    }
  }

  if (infoV > 1 && !chamber_empty) {
    dumpDigis(wire);
  }

  return chamber_empty;
}

bool CSCAnodeLCTProcessor::preTrigger(const int key_wire, const int start_bx) {
  int nPreTriggers = 0;

  unsigned int layers_hit;
  bool hit_layer[CSCConstants::NUM_LAYERS];
  int this_wire;
  // If nplanes_hit_accel_pretrig is 0, the firmware uses the value
  // of nplanes_hit_pretrig instead.
  const unsigned int nplanes_hit_pretrig_acc =
      (nplanes_hit_accel_pretrig != 0) ? nplanes_hit_accel_pretrig : nplanes_hit_pretrig;
  const unsigned int pretrig_thresh[CSCConstants::NUM_ALCT_PATTERNS] = {
      nplanes_hit_pretrig_acc, nplanes_hit_pretrig, nplanes_hit_pretrig};

  // Loop over bx times, accelerator and collision patterns to
  // look for pretrigger.
  // Stop drift_delay bx's short of fifo_tbins since at later bx's we will
  // not have a full set of hits to start pattern search anyway.
  unsigned int stop_bx = fifo_tbins - drift_delay;
  for (unsigned int bx_time = start_bx; bx_time < stop_bx; bx_time++) {
    for (int i_pattern = 0; i_pattern < CSCConstants::NUM_ALCT_PATTERNS; i_pattern++) {
      // initialize the hit layers
      for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++)
        hit_layer[i_layer] = false;
      layers_hit = 0;

      // now run over all layers and wires
      for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
        for (int i_wire = 0; i_wire < CSCConstants::ALCT_PATTERN_WIDTH; i_wire++) {
          // check if the hit is valid
          if (alct_pattern_[i_pattern][i_layer][i_wire]) {
            this_wire = CSCPatternBank::alct_keywire_offset_[MESelection][i_wire] + key_wire;
            if ((this_wire >= 0) && (this_wire < numWireGroups)) {
              // Perform bit operation to see if pulse is 1 at a certain bx_time.
              if (pulse_.isOneShotHighAtBX(i_layer, this_wire, bx_time)) {
                // Store number of layers hit.
                if (!hit_layer[i_layer]) {
                  hit_layer[i_layer] = true;
                  layers_hit++;
                }

                // See if number of layers hit is greater than or equal to
                // pretrig_thresh.
                if (layers_hit >= pretrig_thresh[i_pattern]) {
                  first_bx[key_wire] = bx_time;
                  if (infoV > 1) {
                    LogTrace("CSCAnodeLCTProcessor") << "Pretrigger was satisfied for wire: " << key_wire
                                                     << " pattern: " << i_pattern << " bx_time: " << bx_time;
                  }
                  // make a new pre-trigger
                  nPreTriggers++;
                  // make a new pre-trigger digi
                  // useful for calculating DAQ rates
                  thePreTriggerDigis.emplace_back(
                      CSCALCTPreTriggerDigi(1, layers_hit - 3, 0, 0, this_wire, bx_time, nPreTriggers));
                  return true;
                }
              }
            }
          }
        }
      }
    }
  }
  // If the pretrigger was never satisfied, then return false.
  return false;
}

bool CSCAnodeLCTProcessor::patternDetection(
    const int key_wire, std::map<int, std::map<int, CSCALCTDigi::WireContainer>>& hits_in_patterns) {
  bool trigger = false;
  bool hit_layer[CSCConstants::NUM_LAYERS];
  unsigned int temp_quality;
  int this_wire, delta_wire;
  // If nplanes_hit_accel_pattern is 0, the firmware uses the value
  // of nplanes_hit_pattern instead.
  const unsigned int nplanes_hit_pattern_acc =
      (nplanes_hit_accel_pattern != 0) ? nplanes_hit_accel_pattern : nplanes_hit_pattern;
  const unsigned int pattern_thresh[CSCConstants::NUM_ALCT_PATTERNS] = {
      nplanes_hit_pattern_acc, nplanes_hit_pattern, nplanes_hit_pattern};
  const std::string ptn_label[] = {"Accelerator", "CollisionA", "CollisionB"};

  for (int i_pattern = 0; i_pattern < CSCConstants::NUM_ALCT_PATTERNS; i_pattern++) {
    temp_quality = 0;

    // initialize the hit layers
    for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++)
      hit_layer[i_layer] = false;

    // clear a single pattern!
    CSCALCTDigi::WireContainer hits_single_pattern;
    hits_single_pattern.clear();
    hits_single_pattern.resize(CSCConstants::NUM_LAYERS);
    for (auto& p : hits_single_pattern) {
      p.resize(CSCConstants::ALCT_PATTERN_WIDTH, CSCConstants::INVALID_WIREGROUP);
    }

    double num_pattern_hits = 0., times_sum = 0.;
    std::multiset<int> mset_for_median;
    mset_for_median.clear();

    for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
      for (int i_wire = 0; i_wire < CSCConstants::ALCT_PATTERN_WIDTH; i_wire++) {
        // check if the hit is valid
        if (alct_pattern_[i_pattern][i_layer][i_wire]) {
          delta_wire = CSCPatternBank::alct_keywire_offset_[MESelection][i_wire];
          this_wire = delta_wire + key_wire;
          if ((this_wire >= 0) && (this_wire < numWireGroups)) {
            // Wait a drift_delay time later and look for layers hit in
            // the pattern.
            if (pulse_.isOneShotHighAtBX(i_layer, this_wire, first_bx[key_wire] + drift_delay)) {
              // store hits in the temporary pattern vector
              hits_single_pattern[i_layer][i_wire] = this_wire;

              // If layer has never had a hit before, then increment number
              // of layer hits.
              if (!hit_layer[i_layer]) {
                temp_quality++;
                // keep track of which layers already had hits.
                hit_layer[i_layer] = true;
                if (infoV > 1)
                  LogTrace("CSCAnodeLCTProcessor")
                      << "bx_time: " << first_bx[key_wire] << " pattern: " << i_pattern << " keywire: " << key_wire
                      << " layer: " << i_layer << " quality: " << temp_quality;
              }

              // for averaged time use only the closest WGs around the key WG
              if (abs(delta_wire) < 2) {
                // find at what bx did pulse on this wire&layer start
                // use hit_pesrist constraint on how far back we can go
                int first_bx_layer = first_bx[key_wire] + drift_delay;
                for (unsigned int dbx = 0; dbx < hit_persist; dbx++) {
                  if (pulse_.isOneShotHighAtBX(i_layer, this_wire, first_bx_layer - 1)) {
                    first_bx_layer--;
                  } else
                    break;
                }
                times_sum += (double)first_bx_layer;
                num_pattern_hits += 1.;
                mset_for_median.insert(first_bx_layer);
                if (infoV > 2)
                  LogTrace("CSCAnodeLCTProcessor") << " 1st bx in layer: " << first_bx_layer << " sum bx: " << times_sum
                                                   << " #pat. hits: " << num_pattern_hits;
              }
            }
          }
        }
      }
    }

    // calculate median
    const int sz = mset_for_median.size();
    if (sz > 0) {
      std::multiset<int>::iterator im = mset_for_median.begin();
      if (sz > 1)
        std::advance(im, sz / 2 - 1);
      if (sz == 1)
        first_bx_corrected[key_wire] = *im;
      else if ((sz % 2) == 1)
        first_bx_corrected[key_wire] = *(++im);
      else
        first_bx_corrected[key_wire] = ((*im) + (*(++im))) / 2;

#if defined(EDM_ML_DEBUG)
      if (infoV > 1) {
        auto lt = LogTrace("CSCAnodeLCTProcessor")
                  << "bx=" << first_bx[key_wire] << " bx_cor=" << first_bx_corrected[key_wire] << "  bxset=";
        for (im = mset_for_median.begin(); im != mset_for_median.end(); im++) {
          lt << " " << *im;
        }
      }
#endif
    }

    // save the pattern information when a trigger was formed!
    if (temp_quality >= pattern_thresh[i_pattern]) {
      trigger = true;
      hits_in_patterns[key_wire][i_pattern] = hits_single_pattern;

      // Quality definition changed on 22 June 2007: it no longer depends
      // on pattern_thresh.
      temp_quality = getTempALCTQuality(temp_quality);

      if (i_pattern == CSCConstants::ALCT_ACCELERATOR_PATTERN) {
        // Accelerator pattern
        quality[key_wire][CSCConstants::ALCT_ACCELERATOR_PATTERN] = temp_quality;
      } else {
        // Only one collision pattern (of the best quality) is reported
        if (static_cast<int>(temp_quality) > quality[key_wire][CSCConstants::ALCT_COLLISIONA_PATTERN]) {
          quality[key_wire][CSCConstants::ALCT_COLLISIONA_PATTERN] = temp_quality;   //real quality
          quality[key_wire][CSCConstants::ALCT_COLLISIONB_PATTERN] = i_pattern - 1;  // pattern, left or right
        }
      }
      if (infoV > 1) {
        LogTrace("CSCAnodeLCTProcessor") << "Pattern found; keywire: " << key_wire << " type: " << ptn_label[i_pattern]
                                         << " quality: " << temp_quality << "\n";
      }
    }
  }
  if (infoV > 1 && quality[key_wire][CSCConstants::ALCT_COLLISIONA_PATTERN] > 0) {
    if (quality[key_wire][CSCConstants::ALCT_COLLISIONB_PATTERN] == 0)
      LogTrace("CSCAnodeLCTProcessor") << "Collision Pattern A is chosen"
                                       << "\n";
    else if (quality[key_wire][CSCConstants::ALCT_COLLISIONB_PATTERN] == 1)
      LogTrace("CSCAnodeLCTProcessor") << "Collision Pattern B is chosen"
                                       << "\n";
  }

  trigMode(key_wire);

  return trigger;
}

void CSCAnodeLCTProcessor::ghostCancellationLogicOneWire(const int key_wire, int* ghost_cleared) {
  for (int i_pattern = 0; i_pattern < 2; i_pattern++) {
    ghost_cleared[i_pattern] = 0;
    if (key_wire == 0)
      continue;  //ignore

    // Non-empty wire group.
    int qual_this = quality[key_wire][i_pattern];
    if (qual_this > 0) {
      // Previous wire.
      //int qual_prev = (key_wire > 0) ? quality[key_wire-1][i_pattern] : 0;
      //previous ALCTs were pushed to lct_list, stop use the array quality[key_wire-1][i_pattern]
      for (auto& p : lct_list) {
        //ignore whether ALCT is valid or not in ghost cancellation
        //if wiregroup 10, 11, 12 all have trigger and same quality, only wiregroup 10 can keep the trigger
        //this met with firmware
        if (not(p.getKeyWG() == key_wire - 1 and 1 - p.getAccelerator() == i_pattern))
          continue;

        bool ghost_cleared_prev = false;
        int qual_prev = p.getQuality();
        int first_bx_prev = p.getBX();
        if (infoV > 1)
          LogTrace("CSCAnodeLCTProcessor")
              << "ghost concellation logic "
              << ((i_pattern == CSCConstants::ALCT_ACCELERATOR_PATTERN) ? "Accelerator" : "Collision") << " key_wire "
              << key_wire << " quality " << qual_this << " bx " << first_bx[key_wire] << " previous key_wire "
              << key_wire - 1 << " quality " << qual_prev << " bx " << first_bx[key_wire - 1];

        //int dt = first_bx[key_wire] - first_bx[key_wire-1];
        int dt = first_bx[key_wire] - first_bx_prev;
        // Cancel this wire
        //   1) If the candidate at the previous wire is at the same bx
        //      clock and has better quality (or equal quality - this has
        //      been implemented only in 2004).
        //   2) If the candidate at the previous wire is up to 4 clocks
        //      earlier, regardless of quality.
        if (dt == 0) {
          if (qual_prev >= qual_this)
            ghost_cleared[i_pattern] = 1;
          else if (qual_prev < qual_this)
            ghost_cleared_prev = true;
        } else if (dt > 0 && dt <= ghost_cancellation_bx_depth) {
          if ((!ghost_cancellation_side_quality) || (qual_prev > qual_this))
            ghost_cleared[i_pattern] = 1;
        } else if (dt < 0 && dt * (-1) <= ghost_cancellation_bx_depth) {
          if ((!ghost_cancellation_side_quality) || (qual_prev < qual_this))
            ghost_cleared_prev = true;
        }

        if (ghost_cleared[i_pattern] == 1) {
          if (infoV > 1)
            LogTrace("CSCAnodeLCTProcessor")
                << ((i_pattern == CSCConstants::ALCT_ACCELERATOR_PATTERN) ? "Accelerator" : "Collision")
                << " pattern ghost cancelled on key_wire " << key_wire << " q=" << qual_this << "  by wire "
                << key_wire - 1 << " q=" << qual_prev;
          //cancellation for key_wire is done when ALCT is created and pushed to lct_list
        }

        if (ghost_cleared_prev) {
          if (infoV > 1)
            LogTrace("CSCAnodeLCTProcessor")
                << ((i_pattern == CSCConstants::ALCT_ACCELERATOR_PATTERN) ? "Accelerator" : "Collision")
                << " pattern ghost cancelled on key_wire " << key_wire - 1 << " q=" << qual_prev << "  by wire "
                << key_wire << " q=" << qual_this;
          p.setValid(0);  //clean prev ALCT
        }
      }

    }  // if qual_this > 0
  }    //i_pattern
}

void CSCAnodeLCTProcessor::lctSearch() {
  // Best track selector selects two collision and two accelerator ALCTs
  // with the best quality per time bin.
  const std::vector<CSCALCTDigi>& fourBest = bestTrackSelector(lct_list);

  if (infoV > 0) {
    int n_alct_all = 0, n_alct = 0;
    for (const auto& p : lct_list) {
      if (p.isValid() && p.getBX() == CSCConstants::LCT_CENTRAL_BX)
        n_alct_all++;
    }
    for (const auto& p : fourBest) {
      if (p.isValid() && p.getBX() == CSCConstants::LCT_CENTRAL_BX)
        n_alct++;
    }

    LogTrace("CSCAnodeLCTProcessor") << "alct_count E:" << theEndcap << "S:" << theStation << "R:" << theRing
                                     << "C:" << theChamber << "  all " << n_alct_all << "  found " << n_alct;
  }

  // Select two best of four per time bin, based on quality and
  // accel_mode parameter.
  for (const auto& p : fourBest) {
    const int bx = p.getBX();
    if (bx >= CSCConstants::MAX_ALCT_TBINS) {
      if (infoV > 0)
        edm::LogWarning("CSCAnodeLCTProcessor|OutOfTimeALCT")
            << "+++ Bx of ALCT candidate, " << bx << ", exceeds max allowed, " << CSCConstants::MAX_ALCT_TBINS - 1
            << "; skipping it... +++\n";
      continue;
    }

    if (isBetterALCT(p, bestALCT[bx])) {
      if (isBetterALCT(bestALCT[bx], secondALCT[bx])) {
        secondALCT[bx] = bestALCT[bx];
      }
      bestALCT[bx] = p;
    } else if (isBetterALCT(p, secondALCT[bx])) {
      secondALCT[bx] = p;
    }
  }

  for (int bx = 0; bx < CSCConstants::MAX_ALCT_TBINS; bx++) {
    if (bestALCT[bx].isValid()) {
      bestALCT[bx].setTrknmb(1);

      // check if the best ALCT is valid
      qualityControl_->checkValidReadout(bestALCT[bx]);

      if (infoV > 0) {
        LogDebug("CSCAnodeLCTProcessor") << bestALCT[bx] << " fullBX = " << bestALCT[bx].getFullBX() << " found in "
                                         << theCSCName_ << " (sector " << theSector << " subsector " << theSubsector
                                         << " trig id. " << theTrigChamber << ")"
                                         << "\n";
      }
      if (secondALCT[bx].isValid()) {
        secondALCT[bx].setTrknmb(2);

        // check if the second best ALCT is valid
        qualityControl_->checkValidReadout(secondALCT[bx]);

        if (infoV > 0) {
          LogDebug("CSCAnodeLCTProcessor")
              << secondALCT[bx] << " fullBX = " << secondALCT[bx].getFullBX() << " found in " << theCSCName_
              << " (sector " << theSector << " subsector " << theSubsector << " trig id. " << theTrigChamber << ")"
              << "\n";
        }
      }
    }
  }
}

std::vector<CSCALCTDigi> CSCAnodeLCTProcessor::bestTrackSelector(const std::vector<CSCALCTDigi>& all_alcts) {
  CSCALCTDigi bestALCTs[CSCConstants::MAX_ALCT_TBINS][CSCConstants::MAX_ALCTS_PER_PROCESSOR];
  CSCALCTDigi secondALCTs[CSCConstants::MAX_ALCT_TBINS][CSCConstants::MAX_ALCTS_PER_PROCESSOR];

  if (infoV > 1) {
    LogTrace("CSCAnodeLCTProcessor") << all_alcts.size() << " ALCTs at the input of best-track selector: ";
    for (const auto& p : all_alcts) {
      if (!p.isValid())
        continue;
      LogTrace("CSCAnodeLCTProcessor") << p;
    }
  }

  CSCALCTDigi tA[CSCConstants::MAX_ALCT_TBINS][CSCConstants::MAX_ALCTS_PER_PROCESSOR];
  CSCALCTDigi tB[CSCConstants::MAX_ALCT_TBINS][CSCConstants::MAX_ALCTS_PER_PROCESSOR];
  for (const auto& p : all_alcts) {
    if (!p.isValid())
      continue;

    // Select two collision and two accelerator ALCTs with the highest
    // quality at every bx.  The search for best ALCTs is done in parallel
    // for collision and accelerator patterns, and simultaneously for
    // two ALCTs, tA and tB.  If two or more ALCTs have equal qualities,
    // the priority is given to the ALCT with larger wiregroup number
    // in the search for tA (collision and accelerator), and to the ALCT
    // with smaller wiregroup number in the search for tB.
    int bx = p.getBX();
    int accel = p.getAccelerator();
    int qual = p.getQuality();
    int wire = p.getKeyWG();
    bool vA = tA[bx][accel].isValid();
    bool vB = tB[bx][accel].isValid();
    int qA = tA[bx][accel].getQuality();
    int qB = tB[bx][accel].getQuality();
    int wA = tA[bx][accel].getKeyWG();
    int wB = tB[bx][accel].getKeyWG();
    if (!vA || qual > qA || (qual == qA && wire > wA)) {
      tA[bx][accel] = p;
    }
    if (!vB || qual > qB || (qual == qB && wire < wB)) {
      tB[bx][accel] = p;
    }
  }

  for (int bx = 0; bx < CSCConstants::MAX_ALCT_TBINS; bx++) {
    for (int accel = 0; accel <= 1; accel++) {
      // Best ALCT is always tA.
      if (tA[bx][accel].isValid()) {
        if (infoV > 2) {
          LogTrace("CSCAnodeLCTProcessor") << "tA: " << tA[bx][accel];
          LogTrace("CSCAnodeLCTProcessor") << "tB: " << tB[bx][accel];
        }
        bestALCTs[bx][accel] = tA[bx][accel];

        // If tA exists, tB exists too.
        if (tA[bx][accel] != tB[bx][accel] && tA[bx][accel].getQuality() == tB[bx][accel].getQuality()) {
          secondALCTs[bx][accel] = tB[bx][accel];
        } else {
          // Funny part: if tA and tB are the same, or the quality of tB
          // is inferior to the quality of tA, the second best ALCT is
          // not tB.  Instead it is the largest-wiregroup ALCT among those
          // ALCT whose qualities are lower than the quality of the best one.
          for (const auto& p : all_alcts) {
            if (p.isValid() && p.getAccelerator() == accel && p.getBX() == bx &&
                p.getQuality() < bestALCTs[bx][accel].getQuality() &&
                p.getQuality() >= secondALCTs[bx][accel].getQuality() &&
                p.getKeyWG() >= secondALCTs[bx][accel].getKeyWG()) {
              secondALCTs[bx][accel] = p;
            }
          }
        }
      }
    }
  }

  // Fill the vector with up to four best ALCTs per bx and return it.
  std::vector<CSCALCTDigi> fourBest;
  for (int bx = 0; bx < CSCConstants::MAX_ALCT_TBINS; bx++) {
    for (int i = 0; i < CSCConstants::MAX_ALCTS_PER_PROCESSOR; i++) {
      if (bestALCTs[bx][i].isValid()) {
        fourBest.push_back(bestALCTs[bx][i]);
      }
    }
    for (int i = 0; i < CSCConstants::MAX_ALCTS_PER_PROCESSOR; i++) {
      if (secondALCTs[bx][i].isValid()) {
        fourBest.push_back(secondALCTs[bx][i]);
      }
    }
  }

  if (infoV > 1) {
    LogTrace("CSCAnodeLCTProcessor") << fourBest.size() << " ALCTs selected: ";
    for (const auto& p : fourBest) {
      LogTrace("CSCAnodeLCTProcessor") << p;
    }
  }

  return fourBest;
}

bool CSCAnodeLCTProcessor::isBetterALCT(const CSCALCTDigi& lhsALCT, const CSCALCTDigi& rhsALCT) const {
  bool returnValue = false;

  if (lhsALCT.isValid() && !rhsALCT.isValid()) {
    return true;
  }

  // ALCTs found at earlier bx times are ranked higher than ALCTs found at
  // later bx times regardless of the quality.
  if (lhsALCT.getBX() < rhsALCT.getBX()) {
    returnValue = true;
  }
  if (lhsALCT.getBX() != rhsALCT.getBX()) {
    return returnValue;
  }

  // First check the quality of ALCTs.
  const int qual1 = lhsALCT.getQuality();
  const int qual2 = rhsALCT.getQuality();
  if (qual1 > qual2) {
    returnValue = true;
  }
  // If qualities are the same, check accelerator bits of both ALCTs.
  // If they are not the same, rank according to accel_mode value.
  // If they are the same, keep the track selector assignment.
  //else if (qual1 == qual2 &&
  //         lhsALCT.getAccelerator() != rhsALCT.getAccelerator() &&
  //         quality[lhsALCT.getKeyWG()][1-lhsALCT.getAccelerator()] >
  //         quality[rhsALCT.getKeyWG()][1-rhsALCT.getAccelerator()])
  //  {returnValue = true;}
  else if (qual1 == qual2 && lhsALCT.getAccelerator() != rhsALCT.getAccelerator()) {
    if ((accel_mode == 0 || accel_mode == 1) && rhsALCT.getAccelerator() == 0)
      returnValue = true;
    if ((accel_mode == 2 || accel_mode == 3) && lhsALCT.getAccelerator() == 0)
      returnValue = true;
  }

  return returnValue;
}

void CSCAnodeLCTProcessor::trigMode(const int key_wire) {
  switch (trig_mode) {
    default:
    case 0:
      // Enables both collision and accelerator tracks
      break;
    case 1:
      // Disables collision tracks
      if (quality[key_wire][CSCConstants::ALCT_COLLISIONA_PATTERN] > 0) {
        quality[key_wire][CSCConstants::ALCT_COLLISIONA_PATTERN] = 0;
        if (infoV > 1)
          LogTrace("CSCAnodeLCTProcessor") << "trigMode(): collision track " << key_wire << " disabled"
                                           << "\n";
      }
      break;
    case 2:
      // Disables accelerator tracks
      if (quality[key_wire][CSCConstants::ALCT_ACCELERATOR_PATTERN] > 0) {
        quality[key_wire][CSCConstants::ALCT_ACCELERATOR_PATTERN] = 0;
        if (infoV > 1)
          LogTrace("CSCAnodeLCTProcessor") << "trigMode(): accelerator track " << key_wire << " disabled"
                                           << "\n";
      }
      break;
    case 3:
      // Disables collision track if there is an accelerator track found
      // in the same wire group at the same time
      if (quality[key_wire][0] > 0 && quality[key_wire][CSCConstants::ALCT_COLLISIONA_PATTERN] > 0) {
        quality[key_wire][CSCConstants::ALCT_COLLISIONA_PATTERN] = 0;
        if (infoV > 1)
          LogTrace("CSCAnodeLCTProcessor") << "trigMode(): collision track " << key_wire << " disabled"
                                           << "\n";
      }
      break;
  }
}

void CSCAnodeLCTProcessor::accelMode(const int key_wire) {
  int promotionBit = 1 << 2;

  switch (accel_mode) {
    default:
    case 0:
      // Ignore accelerator muons.
      if (quality[key_wire][0] > 0) {
        quality[key_wire][0] = 0;
        if (infoV > 1)
          LogTrace("CSCAnodeLCTProcessor") << "alctMode(): accelerator track " << key_wire << " ignored"
                                           << "\n";
      }
      break;
    case 1:
      // Prefer collision muons by adding promotion bit.
      if (quality[key_wire][CSCConstants::ALCT_COLLISIONA_PATTERN] > 0) {
        quality[key_wire][CSCConstants::ALCT_COLLISIONA_PATTERN] += promotionBit;
        if (infoV > 1)
          LogTrace("CSCAnodeLCTProcessor") << "alctMode(): collision track " << key_wire << " promoted"
                                           << "\n";
      }
      break;
    case 2:
      // Prefer accelerator muons by adding promotion bit.
      if (quality[key_wire][CSCConstants::ALCT_ACCELERATOR_PATTERN] > 0) {
        quality[key_wire][CSCConstants::ALCT_ACCELERATOR_PATTERN] += promotionBit;
        if (infoV > 1)
          LogTrace("CSCAnodeLCTProcessor") << "alctMode(): accelerator track " << key_wire << " promoted"
                                           << "\n";
      }
      break;
    case 3:
      // Ignore collision muons.
      if (quality[key_wire][CSCConstants::ALCT_COLLISIONA_PATTERN] > 0) {
        quality[key_wire][CSCConstants::ALCT_COLLISIONA_PATTERN] = 0;
        if (infoV > 1)
          LogTrace("CSCAnodeLCTProcessor") << "alctMode(): collision track " << key_wire << " ignored"
                                           << "\n";
      }
      break;
  }
}

// Dump of configuration parameters.
void CSCAnodeLCTProcessor::dumpConfigParams() const {
  std::ostringstream strm;
  strm << "\n";
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  strm << "+                  ALCT configuration parameters:                  +\n";
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  strm << " fifo_tbins   [total number of time bins in DAQ readout] = " << fifo_tbins << "\n";
  strm << " fifo_pretrig [start time of anode raw hits in DAQ readout] = " << fifo_pretrig << "\n";
  strm << " drift_delay  [drift delay after pre-trigger, in 25 ns bins] = " << drift_delay << "\n";
  strm << " nplanes_hit_pretrig [min. number of layers hit for pre-trigger] = " << nplanes_hit_pretrig << "\n";
  strm << " nplanes_hit_pattern [min. number of layers hit for trigger] = " << nplanes_hit_pattern << "\n";
  strm << " nplanes_hit_accel_pretrig [min. number of layers hit for accel."
       << " pre-trig.] = " << nplanes_hit_accel_pretrig << "\n";
  strm << " nplanes_hit_accel_pattern [min. number of layers hit for accel."
       << " trigger] = " << nplanes_hit_accel_pattern << "\n";
  strm << " trig_mode  [enabling/disabling collision/accelerator tracks] = " << trig_mode << "\n";
  strm << " accel_mode [preference to collision/accelerator tracks] = " << accel_mode << "\n";
  strm << " l1a_window_width [L1Accept window width, in 25 ns bins] = " << l1a_window_width << "\n";
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  LogDebug("CSCAnodeLCTProcessor") << strm.str();
}

// Dump of digis on wire groups.
void CSCAnodeLCTProcessor::dumpDigis(
    const std::vector<int> wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIREGROUPS]) const {
  LogDebug("CSCAnodeLCTProcessor") << theCSCName_ << " nWiregroups " << numWireGroups;

  std::ostringstream strstrm;
  for (int i_wire = 0; i_wire < numWireGroups; i_wire++) {
    if (i_wire % 10 == 0) {
      if (i_wire < 100)
        strstrm << i_wire / 10;
      else
        strstrm << (i_wire - 100) / 10;
    } else
      strstrm << " ";
  }
  strstrm << "\n";
  for (int i_wire = 0; i_wire < numWireGroups; i_wire++) {
    strstrm << i_wire % 10;
  }
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    strstrm << "\n";
    for (int i_wire = 0; i_wire < numWireGroups; i_wire++) {
      if (!wire[i_layer][i_wire].empty()) {
        std::vector<int> bx_times = wire[i_layer][i_wire];
        strstrm << std::hex << bx_times[0] << std::dec;
      } else {
        strstrm << ".";
      }
    }
  }
  LogTrace("CSCAnodeLCTProcessor") << strstrm.str();
}

// Returns vector of read-out ALCTs, if any.  Starts with the vector of
// all found ALCTs and selects the ones in the read-out time window.
std::vector<CSCALCTDigi> CSCAnodeLCTProcessor::readoutALCTs() const {
  std::vector<CSCALCTDigi> tmpV;

  // The number of LCT bins in the read-out is given by the
  // l1a_window_width parameter, but made even by setting the LSB of
  // l1a_window_width to 0.
  const int lct_bins = l1a_window_width;
  static std::atomic<int> late_tbins{early_tbins + lct_bins};

  static std::atomic<int> ifois{0};
  if (ifois == 0) {
    if (infoV >= 0 && early_tbins < 0) {
      edm::LogWarning("CSCAnodeLCTProcessor|SuspiciousParameters")
          << "+++ fifo_pretrig = " << fifo_pretrig << "; in-time ALCTs are not getting read-out!!! +++"
          << "\n";
    }

    if (late_tbins > CSCConstants::MAX_ALCT_TBINS - 1) {
      if (infoV >= 0)
        edm::LogWarning("CSCAnodeLCTProcessor|SuspiciousParameters")
            << "+++ Allowed range of time bins, [0-" << late_tbins << "] exceeds max allowed, "
            << CSCConstants::MAX_ALCT_TBINS - 1 << " +++\n"
            << "+++ Set late_tbins to max allowed +++\n";
      late_tbins = CSCConstants::MAX_ALCT_TBINS - 1;
    }
    ifois = 1;
  }

  // Start from the vector of all found ALCTs and select those within
  // the ALCT*L1A coincidence window.
  const std::vector<CSCALCTDigi>& all_alcts = getALCTs();
  for (const auto& p : all_alcts) {
    if (!p.isValid())
      continue;

    int bx = p.getBX();
    // Skip ALCTs found too early relative to L1Accept.
    if (bx <= early_tbins) {
      if (infoV > 1)
        LogDebug("CSCAnodeLCTProcessor") << " Do not report ALCT on keywire " << p.getKeyWG() << ": found at bx " << bx
                                         << ", whereas the earliest allowed bx is " << early_tbins + 1;
      continue;
    }

    // Skip ALCTs found too late relative to L1Accept.
    if (bx > late_tbins) {
      if (infoV > 1)
        LogDebug("CSCAnodeLCTProcessor") << " Do not report ALCT on keywire " << p.getKeyWG() << ": found at bx " << bx
                                         << ", whereas the latest allowed bx is " << late_tbins;
      continue;
    }

    tmpV.push_back(p);
  }

  // shift the BX from 8 to 3
  // ALCTs in real data have the central BX in bin 3
  // which is the middle of the 7BX wide L1A window
  // ALCTs used in the TMB emulator have central BX at bin 8
  // but right before we put emulated ALCTs in the event, we shift the BX
  // by -5 to make sure they are compatible with real data ALCTs!
  for (auto& p : tmpV) {
    p.setBX(p.getBX() - (CSCConstants::LCT_CENTRAL_BX - l1a_window_width / 2));
  }

  // do a final check on the ALCTs in readout
  qualityControl_->checkMultiplicityBX(tmpV);
  for (const auto& alct : tmpV) {
    qualityControl_->checkValid(alct);
  }

  return tmpV;
}

// Returns vector of all found ALCTs, if any.  Used in ALCT-CLCT matching.
std::vector<CSCALCTDigi> CSCAnodeLCTProcessor::getALCTs() const {
  std::vector<CSCALCTDigi> tmpV;
  for (int bx = 0; bx < CSCConstants::MAX_ALCT_TBINS; bx++) {
    if (bestALCT[bx].isValid())
      tmpV.push_back(bestALCT[bx]);
    if (secondALCT[bx].isValid())
      tmpV.push_back(secondALCT[bx]);
  }
  return tmpV;
}

CSCALCTDigi CSCAnodeLCTProcessor::getBestALCT(int bx) const { return bestALCT[bx]; }

CSCALCTDigi CSCAnodeLCTProcessor::getSecondALCT(int bx) const { return secondALCT[bx]; }

/** Returns shower bits */
CSCShowerDigi CSCAnodeLCTProcessor::readoutShower() const { return shower_; }

////////////////////////////////////////////////////////////////////////
////////////////////////////Test Routines///////////////////////////////

void CSCAnodeLCTProcessor::showPatterns(const int key_wire) {
  /* Method to test the pretrigger */
  for (int i_pattern = 0; i_pattern < CSCConstants::NUM_ALCT_PATTERNS; i_pattern++) {
    std::ostringstream strstrm_header;
    LogTrace("CSCAnodeLCTProcessor") << "\n"
                                     << "Pattern: " << i_pattern << " Key wire: " << key_wire;
    for (int i = 1; i <= 32; i++) {
      strstrm_header << ((32 - i) % 10);
    }
    LogTrace("CSCAnodeLCTProcessor") << strstrm_header.str();
    for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
      for (int i_wire = 0; i_wire < CSCConstants::ALCT_PATTERN_WIDTH; i_wire++) {
        // check if the hit is valid
        if (alct_pattern_[i_pattern][i_layer][i_wire]) {
          std::ostringstream strstrm_pulse;
          int this_wire = CSCPatternBank::alct_keywire_offset_[MESelection][i_wire] + key_wire;
          if (this_wire >= 0 && this_wire < numWireGroups) {
            for (int i = 1; i <= 32; i++) {
              strstrm_pulse << pulse_.oneShotAtBX(i_layer, this_wire, 32 - i);
            }
            LogTrace("CSCAnodeLCTProcessor") << strstrm_pulse.str() << " on layer " << i_layer << " wire " << this_wire;
          }
        }
      }
    }
    LogTrace("CSCAnodeLCTProcessor") << "-------------------------------------------";
  }
}

int CSCAnodeLCTProcessor::getTempALCTQuality(int temp_quality) const {
  int Q;
  if (temp_quality > 3)
    Q = temp_quality - 3;
  else
    Q = 0;  // quality code 0 is valid!

  return Q;
}

void CSCAnodeLCTProcessor::cleanWireContainer(CSCALCTDigi::WireContainer& wireHits) const {
  for (auto& p : wireHits) {
    p.erase(std::remove_if(p.begin(), p.end(), [](unsigned i) -> bool { return i == CSCConstants::INVALID_WIREGROUP; }),
            p.end());
  }
}

void CSCAnodeLCTProcessor::setWireContainer(CSCALCTDigi& alct, CSCALCTDigi::WireContainer& wireHits) const {
  // clean the wire digi container
  cleanWireContainer(wireHits);

  // set the hit container
  alct.setHits(wireHits);
}

void CSCAnodeLCTProcessor::encodeHighMultiplicityBits(
    const std::vector<int> wires[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIREGROUPS]) {
  inTimeHMT_ = 0;
  outTimeHMT_ = 0;

  // functions for in-time and out-of-time
  auto inTime = [=](unsigned time) { return time >= showerMinInTBin_ and time <= showerMaxInTBin_; };
  auto outTime = [=](unsigned time) { return time >= showerMinOutTBin_ and time <= showerMaxOutTBin_; };

  // count the wires in-time and out-time
  unsigned hitsInTime = 0;
  unsigned hitsOutTime = 0;
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    for (int i_wire = 0; i_wire < CSCConstants::MAX_NUM_WIREGROUPS; i_wire++) {
      auto times = wires[i_layer][i_wire];
      hitsInTime += std::count_if(times.begin(), times.end(), inTime);
      hitsOutTime += std::count_if(times.begin(), times.end(), outTime);
    }
  }

  // convert station and ring number to index
  // index runs from 2 to 10, subtract 2
  unsigned csc_idx = CSCDetId::iChamberType(theStation, theRing) - 2;

  // loose, nominal and tight
  std::vector<unsigned> station_thresholds = {
      thresholds_[csc_idx * 3], thresholds_[csc_idx * 3 + 1], thresholds_[csc_idx * 3 + 2]};

  // assign the bits
  for (unsigned i = 0; i < station_thresholds.size(); i++) {
    if (hitsInTime >= station_thresholds[i]) {
      inTimeHMT_ = i + 1;
    }
    if (hitsOutTime >= station_thresholds[i]) {
      outTimeHMT_ = i + 1;
    }
  }

  // create a new object
  shower_ = CSCShowerDigi(inTimeHMT_, outTimeHMT_, theTrigChamber);
}
