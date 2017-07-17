//-----------------------------------------------------------------------------
//
//   Class: CSCCathodeLCTProcessor
//
//   Description: 
//     This class simulates the functionality of the cathode LCT card.  It is
//     run by the MotherBoard and returns up to two CathodeLCTs. It can be
//     run either in a test mode, where it is passed arrays of halfstrip
//     and distrip times, or in normal mode where it determines
//     the time and comparator information from the comparator digis.
//
//     Additional comments by Jason Mumford 01/31/01 (mumford@physics.ucla.edu)
//     Removed the card boundaries.  Changed the Pretrigger to emulate
//     the hardware electronic logic.  Also changed the keylayer to be the 4th
//     layer in a chamber instead of the 3rd layer from the interaction region.
//     The code is a more realistic simulation of hardware LCT logic now.
//
//   Author List: Benn Tannenbaum (1999), Jason Mumford (2001-2), Slava Valuev.
//                Porting from ORCA by S. Valuev (Slava.Valuev@cern.ch),
//                May 2006.
//
//
//   Modifications: 
//
//-----------------------------------------------------------------------------

#include "L1Trigger/CSCTriggerPrimitives/src/CSCCathodeLCTProcessor.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h"
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <set>

//-----------------
// Static variables
//-----------------

// This is the strip pattern that we use for pretrigger.
// pre_hit_pattern[0][i] = layer. pre_hit_pattern[1][i] = key_strip offset.
const int CSCCathodeLCTProcessor::pre_hit_pattern[2][NUM_PATTERN_STRIPS] = {
  { 999,  0,  0,  0,  999,
    999,  1,  1,  1,  999,
    999,  2,  2,  2,  999,
              3,                // layer
    999,  4,  4,  4,  999,
    999,  5,  5,  5,  999},
  //-------------------------------------------
  { 999, -1,  0,  1,  999,
    999, -1,  0,  1,  999,
    999, -1,  0,  1,  999,
              0,                // offset
    999, -1,  0,  1,  999,
    999, -1,  0,  1,  999}
};

// The old set of half-strip/di-strip patterns used prior to 2007.
// For the given pattern, set the unused parts of the pattern to 999.
// Pattern[i][NUM_PATTERN_STRIPS] contains pt bend value. JM
// bend of 0 is left/straight and bend of 1 is right bht 21 June 2001
// note that the left/right-ness of this is exactly opposite of what one would
// expect naively (at least it was for me). The only way to make sure you've
// got the patterns you want is to use the printPatterns() method to dump
// them. BHT 21 June 2001
const int CSCCathodeLCTProcessor::pattern[CSCConstants::NUM_CLCT_PATTERNS_PRE_TMB07][NUM_PATTERN_STRIPS+1] = {
  { 999, 999, 999, 999, 999,
    999, 999, 999, 999, 999,
    999, 999, 999, 999, 999,
              999,            // dummy (reserved)
    999, 999, 999, 999, 999,
    999, 999, 999, 999, 999, 0},
  //-------------------------------------------------------------
  { 999, 999, 999,   0, 999,
    999, 999, 999,   1, 999,
    999, 999,   2,   2, 999,
                3,            // right bending pattern (large)
    999,   4,   4, 999, 999,
    999,   5, 999, 999, 999, 1},
  //-------------------------------------------------------------
  { 999,   0, 999, 999, 999,
    999,   1, 999, 999, 999,
    999,   2,   2, 999, 999,
                3,            // left bending pattern (large)
    999, 999,   4,   4, 999,
    999, 999, 999,   5, 999, 0},
  //-------------------------------------------------------------
  { 999, 999,   0, 999, 999,
    999, 999,   1, 999, 999,
    999, 999,   2, 999, 999,
                3,            // right bending pattern (medium)
    999,   4, 999, 999, 999,
    999,   5, 999, 999, 999, 1},
  //-------------------------------------------------------------
  { 999, 999,   0, 999, 999,
    999, 999,   1, 999, 999,
    999, 999,   2, 999, 999,
                3,            // left bending pattern (medium)
    999, 999, 999,   4, 999,
    999, 999, 999,   5, 999, 0},
  //-------------------------------------------------------------
  { 999, 999, 999,   0, 999,
    999, 999, 999,   1, 999,
    999, 999,   2,   2, 999,
                3,            // right bending pattern (small)
    999, 999,   4, 999, 999,
    999, 999,   5, 999, 999, 1},
  //-------------------------------------------------------------
  { 999,   0, 999, 999, 999,
    999,   1, 999, 999, 999,
    999,   2,   2, 999, 999,
                3,            // left bending pattern (small)
    999, 999,   4, 999, 999,
    999, 999,   5, 999, 999, 0},
  //-------------------------------------------------------------
  { 999, 999,   0, 999, 999,
    999, 999,   1, 999, 999,
    999, 999,   2, 999, 999,
                3,            // straight through pattern
    999, 999,   4, 999, 999,
    999, 999,   5, 999, 999, 1}
};

// New set of halfstrip patterns for 2007 version of the algorithm.
// For the given pattern, set the unused parts of the pattern to 999.
// Pattern[i][NUM_PATTERN_HALFSTRIPS] contains bend direction.
// Bend of 0 is right/straight and bend of 1 is left.
// Pattern[i][NUM_PATTERN_HALFSTRIPS+1] contains pattern maximum width
const int CSCCathodeLCTProcessor::pattern2007_offset[NUM_PATTERN_HALFSTRIPS] =
  {  -5,  -4,  -3,  -2,  -1,   0,   1,   2,   3,   4,   5,
                    -2,  -1,   0,   1,   2,
                               0,
                    -2,  -1,   0,   1,   2,
          -4,  -3,  -2,  -1,   0,   1,   2,   3,   4,
     -5,  -4,  -3,  -2,  -1,   0,   1,   2,   3,   4,   5 };

const int CSCCathodeLCTProcessor::pattern2007[CSCConstants::NUM_CLCT_PATTERNS][NUM_PATTERN_HALFSTRIPS+2] = {
  { 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999,
                   999, 999, 999, 999, 999,
                             999,             // pid=0: no pattern found
                   999, 999, 999, 999, 999,
         999, 999, 999, 999, 999, 999, 999, 999, 999,
    999, 999, 999, 999, 999, 999, 999, 999, 999, 999, 999, -1, 0},
  //-------------------------------------------------------------
  {   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
                     1,   1,   1,   1,   1,
                               2,             // pid=1: layer-OR trigger
                     3,   3,   3,   3,   3,
           4,   4,   4,   4,   4,   4,   4,   4,   4,
      5,   5,   5,   5,   5,   5,   5,   5,   5,   5,   5, -1, 11},
  //-------------------------------------------------------------
  { 999, 999, 999, 999, 999, 999, 999, 999,   0,   0,   0,
                   999, 999, 999,   1,   1,
                               2,             // pid=2: right-bending (large)
                     3,   3,   3, 999, 999,
           4,   4,   4, 999, 999, 999, 999, 999, 999,
      5,   5,   5, 999, 999, 999, 999, 999, 999, 999, 999,  0, 11},
  //-------------------------------------------------------------
  {   0,   0,   0, 999, 999, 999, 999, 999, 999, 999, 999,
                     1,   1, 999, 999, 999,
                               2,             // pid=3: left-bending (large)
                   999, 999,   3,   3,   3,
         999, 999, 999, 999, 999, 999,   4,   4,   4,
    999, 999, 999, 999, 999, 999, 999, 999,   5,   5,   5,  1, 11},
  //-------------------------------------------------------------
  { 999, 999, 999, 999, 999, 999, 999,   0,   0,   0, 999,
                   999, 999, 999,   1,   1,
                               2,             // pid=4: right-bending (medium)
                     3,   3, 999, 999, 999,
           4,   4,   4, 999, 999, 999, 999, 999, 999,
    999,   5,   5,   5, 999, 999, 999, 999, 999, 999, 999,  0, 9},
  //-------------------------------------------------------------
  { 999,   0,   0,   0, 999, 999, 999, 999, 999, 999, 999,
                     1,   1, 999, 999, 999,
                               2,             // pid=5: left-bending (medium)
                   999, 999, 999,   3,   3,
         999, 999, 999, 999, 999, 999,   4,   4,   4,
    999, 999, 999, 999, 999, 999, 999,   5,   5,   5, 999,  1, 9},
  //-------------------------------------------------------------
  { 999, 999, 999, 999, 999, 999,   0,   0,   0, 999, 999,
                   999, 999,   1,   1, 999,
                               2,             // pid=6: right-bending (medium)
                   999,   3,   3, 999, 999,
         999, 999,   4,   4, 999, 999, 999, 999, 999,
    999, 999,   5,   5,   5, 999, 999, 999, 999, 999, 999,  0, 7},
  //-------------------------------------------------------------
  { 999, 999,   0,   0,   0, 999, 999, 999, 999, 999, 999,
                   999,   1,   1, 999, 999,
                               2,             // pid=7: left-bending (medium)
                   999, 999,   3,   3, 999,
         999, 999, 999, 999, 999,   4,   4, 999, 999,
    999, 999, 999, 999, 999, 999,   5,   5,   5, 999, 999,  1, 7},
  //-------------------------------------------------------------
  { 999, 999, 999, 999, 999,   0,   0,   0, 999, 999, 999,
                   999, 999,   1,   1, 999,
                               2,             // pid=8: right-bending (small)
                   999,   3,   3, 999, 999,
         999, 999,   4,   4,   4, 999, 999, 999, 999,
    999, 999, 999,   5,   5,   5, 999, 999, 999, 999, 999,  0, 5},
  //-------------------------------------------------------------
  { 999, 999, 999,   0,   0,   0, 999, 999, 999, 999, 999,
                   999,   1,   1, 999, 999,
                               2,             // pid=9: left-bending (small)
                   999, 999,   3,   3, 999,
         999, 999, 999, 999,   4,   4,   4, 999, 999,
    999, 999, 999, 999, 999,   5,   5,   5, 999, 999, 999,  1, 5},
  //-------------------------------------------------------------
  { 999, 999, 999, 999,   0,   0,   0, 999, 999, 999, 999,
                   999, 999,   1, 999, 999,
                               2,             // pid=A: straight-through
                   999, 999,   3, 999, 999,
         999, 999, 999,   4,   4,   4, 999, 999, 999,
    999, 999, 999, 999,   5,   5,   5, 999, 999, 999, 999,  0, 3}
                                              // pid's=B-F are not yet defined
};

// Default values of configuration parameters.
const unsigned int CSCCathodeLCTProcessor::def_fifo_tbins   = 12;
const unsigned int CSCCathodeLCTProcessor::def_fifo_pretrig =  7;
const unsigned int CSCCathodeLCTProcessor::def_hit_persist  =  6;
const unsigned int CSCCathodeLCTProcessor::def_drift_delay  =  2;
const unsigned int CSCCathodeLCTProcessor::def_nplanes_hit_pretrig =  2;
const unsigned int CSCCathodeLCTProcessor::def_nplanes_hit_pattern =  4;
const unsigned int CSCCathodeLCTProcessor::def_pid_thresh_pretrig  =  2;
const unsigned int CSCCathodeLCTProcessor::def_min_separation      = 10;
const unsigned int CSCCathodeLCTProcessor::def_tmb_l1a_window_size =  7;

// Number of di-strips/half-strips per CFEB.
const int CSCCathodeLCTProcessor::cfeb_strips[2] = { 8, 32};

//----------------
// Constructors --
//----------------

CSCCathodeLCTProcessor::CSCCathodeLCTProcessor(unsigned endcap,
					       unsigned station,
					       unsigned sector,
					       unsigned subsector,
					       unsigned chamber,
					       const edm::ParameterSet& conf,
					       const edm::ParameterSet& comm,
					       const edm::ParameterSet& ctmb) :
		     theEndcap(endcap), theStation(station), theSector(sector),
                     theSubsector(subsector), theTrigChamber(chamber) {
  static std::atomic<bool> config_dumped{false};

  // CLCT configuration parameters.
  fifo_tbins   = conf.getParameter<unsigned int>("clctFifoTbins");
  hit_persist  = conf.getParameter<unsigned int>("clctHitPersist");
  drift_delay  = conf.getParameter<unsigned int>("clctDriftDelay");
  nplanes_hit_pretrig =
    conf.getParameter<unsigned int>("clctNplanesHitPretrig");
  nplanes_hit_pattern =
    conf.getParameter<unsigned int>("clctNplanesHitPattern");

  // Not used yet.
  fifo_pretrig = conf.getParameter<unsigned int>("clctFifoPretrig");

  // Defines pre-2007 version of the CLCT algorithm used in test beams and
  // MTCC.
  isMTCC       = comm.getParameter<bool>("isMTCC");

  // TMB07 firmware used since 2007: switch and config. parameters.
  isTMB07      = comm.getParameter<bool>("isTMB07");

  // Flag for SLHC studies
  isSLHC       = comm.getParameter<bool>("isSLHC");

  // special configuration parameters for ME11 treatment
  smartME1aME1b = comm.getParameter<bool>("smartME1aME1b");
  disableME1a = comm.getParameter<bool>("disableME1a");
  gangedME1a = comm.getParameter<bool>("gangedME1a");

  if (isSLHC && !smartME1aME1b) edm::LogError("L1CSCTPEmulatorConfigError")
    << "+++ SLHC upgrade configuration is used (isSLHC=True) but smartME1aME1b=False!\n"
    << "Only smartME1aME1b algorithm is so far supported for upgrade! +++\n";

  if (isTMB07) {
    pid_thresh_pretrig =
      conf.getParameter<unsigned int>("clctPidThreshPretrig");
    min_separation    =
      conf.getParameter<unsigned int>("clctMinSeparation");

    start_bx_shift = conf.getParameter<int>("clctStartBxShift");
  }

  if (smartME1aME1b) {
    // use of localized dead-time zones
    use_dead_time_zoning = conf.existsAs<bool>("useDeadTimeZoning")?conf.getParameter<bool>("useDeadTimeZoning"):true;
    clct_state_machine_zone = conf.existsAs<unsigned int>("clctStateMachineZone")?conf.getParameter<unsigned int>("clctStateMachineZone"):8;
    dynamic_state_machine_zone = conf.existsAs<bool>("useDynamicStateMachineZone")?conf.getParameter<bool>("useDynamicStateMachineZone"):true;

    // how far away may trigger happen from pretrigger
    pretrig_trig_zone = conf.existsAs<unsigned int>("clctPretriggerTriggerZone")?conf.getParameter<unsigned int>("clctPretriggerTriggerZone"):5;

    // whether to calculate bx as corrected_bx instead of pretrigger one
    use_corrected_bx = conf.existsAs<bool>("clctUseCorrectedBx")?conf.getParameter<bool>("clctUseCorrectedBx"):true;
  }
  
  // Motherboard parameters: common for all configurations.
  tmb_l1a_window_size = // Common to CLCT and TMB
    ctmb.getParameter<unsigned int>("tmbL1aWindowSize");

  // separate handle for early time bins
  early_tbins = ctmb.getParameter<int>("tmbEarlyTbins");
  static int fpga_latency = 3;
  if (early_tbins<0) early_tbins  = fifo_pretrig - fpga_latency;

  // wether to readout only the earliest two LCTs in readout window
  readout_earliest_2 = ctmb.getParameter<bool>("tmbReadoutEarliest2");

  // Verbosity level, set to 0 (no print) by default.
  infoV        = conf.getParameter<int>("verbosity");

  // Check and print configuration parameters.
  checkConfigParameters();
  if ((infoV > 0 || isSLHC) && !config_dumped) {
    //std::cerr<<"**** CLCT constructor parameters dump ****"<<std::endl;
    dumpConfigParams();
    config_dumped = true;
  }

  numStrips = 0; // Will be set later.
  // Provisional, but should be OK for all stations except ME1.
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    if ((i_layer+1)%2 == 0) stagger[i_layer] = 0;
    else                    stagger[i_layer] = 1;
  }

  theRing = CSCTriggerNumbering::ringFromTriggerLabels(theStation, theTrigChamber);

  theChamber = CSCTriggerNumbering::chamberFromTriggerLabels(theSector, theSubsector,
                                                             theStation, theTrigChamber);

  // trigger numbering doesn't distinguish between ME1a and ME1b chambers:
  isME11 = (theStation == 1 && theRing == 1);

  //if (theStation==1 && theRing==2) infoV = 3;

  ////engage in various and sundry tests, but only for a single chamber.
  //if (theStation == 2 && theSector == 1 &&
  //    theRing == 1 &&
  //    theChamber == 1) {
  ////  test all possible patterns in our uber pattern.
  //  testPatterns();
  ////  this tests to make sure what goes into an LCT is what comes out
  //  testLCTs();
  ////  print out all the patterns to make sure we've got what we think we've got.
  //  printPatterns();
  //}
}

CSCCathodeLCTProcessor::CSCCathodeLCTProcessor() :
  		     theEndcap(1), theStation(1), theSector(1),
                     theSubsector(1), theTrigChamber(1) {
  // constructor for debugging.
  static std::atomic<bool> config_dumped{false};

  // CLCT configuration parameters.
  setDefaultConfigParameters();
  infoV =  2;
  isMTCC  = false;
  isTMB07 = true;

  smartME1aME1b = false;
  disableME1a = false;
  gangedME1a = true;

  early_tbins = 4;

  start_bx_shift = 0;
  use_dead_time_zoning = 1;
  clct_state_machine_zone = 8;
  
  // Check and print configuration parameters.
  checkConfigParameters();
  if (!config_dumped) {
    //std::cerr<<"**** CLCT default constructor parameters dump ****"<<std::endl;
    dumpConfigParams();
    config_dumped = true;
  }

  numStrips = CSCConstants::MAX_NUM_STRIPS;
  // Should be OK for all stations except ME1.
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    if ((i_layer+1)%2 == 0) stagger[i_layer] = 0;
    else                    stagger[i_layer] = 1;
  }
  
  theRing = CSCTriggerNumbering::ringFromTriggerLabels(theStation, theTrigChamber);
  isME11 = (theStation == 1 && theRing == 1);
}

void CSCCathodeLCTProcessor::setDefaultConfigParameters() {
  // Set default values for configuration parameters.
  fifo_tbins   = def_fifo_tbins;
  fifo_pretrig = def_fifo_pretrig;
  hit_persist  = def_hit_persist;
  drift_delay  = def_drift_delay;
  nplanes_hit_pretrig = def_nplanes_hit_pretrig;
  nplanes_hit_pattern = def_nplanes_hit_pattern;

  isMTCC = false;

  // New TMB07 parameters.
  isTMB07 = true;
  if (isTMB07) {
    pid_thresh_pretrig = def_pid_thresh_pretrig;
    min_separation     = def_min_separation;
  }

  tmb_l1a_window_size = def_tmb_l1a_window_size;
}

// Set configuration parameters obtained via EventSetup mechanism.
void CSCCathodeLCTProcessor::setConfigParameters(const CSCDBL1TPParameters* conf) {
  static std::atomic<bool> config_dumped{false};

  fifo_tbins   = conf->clctFifoTbins();
  fifo_pretrig = conf->clctFifoPretrig();
  hit_persist  = conf->clctHitPersist();
  drift_delay  = conf->clctDriftDelay();
  nplanes_hit_pretrig = conf->clctNplanesHitPretrig();
  nplanes_hit_pattern = conf->clctNplanesHitPattern();

  // TMB07 parameters.
  if (isTMB07) {
    pid_thresh_pretrig = conf->clctPidThreshPretrig();
    min_separation     = conf->clctMinSeparation();
  }

  // Check and print configuration parameters.
  checkConfigParameters();
  if (!config_dumped) {
    //std::cerr<<"**** CLCT setConfigParams parameters dump ****"<<std::endl;
    dumpConfigParams();
    config_dumped = true;
  }
}

void CSCCathodeLCTProcessor::checkConfigParameters() {
  // Make sure that the parameter values are within the allowed range.

  // Max expected values.
  static const unsigned int max_fifo_tbins   = 1 << 5;
  static const unsigned int max_fifo_pretrig = 1 << 5;
  static const unsigned int max_hit_persist  = 1 << 4;
  static const unsigned int max_drift_delay  = 1 << 2;
  static const unsigned int max_nplanes_hit_pretrig = 1 << 3;
  static const unsigned int max_nplanes_hit_pattern = 1 << 3;
  static const unsigned int max_pid_thresh_pretrig  = 1 << 4;
  static const unsigned int max_min_separation = CSCConstants::NUM_HALF_STRIPS_7CFEBS;
  static const unsigned int max_tmb_l1a_window_size = 1 << 4;

  // Checks.
  if (fifo_tbins >= max_fifo_tbins) {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorConfigError")
      << "+++ Value of fifo_tbins, " << fifo_tbins
      << ", exceeds max allowed, " << max_fifo_tbins-1 << " +++\n"
      << "+++ Try to proceed with the default value, fifo_tbins="
      << def_fifo_tbins << " +++\n";
    fifo_tbins = def_fifo_tbins;
  }
  if (fifo_pretrig >= max_fifo_pretrig) {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorConfigError")
      << "+++ Value of fifo_pretrig, " << fifo_pretrig
      << ", exceeds max allowed, " << max_fifo_pretrig-1 << " +++\n"
      << "+++ Try to proceed with the default value, fifo_pretrig="
      << def_fifo_pretrig << " +++\n";
    fifo_pretrig = def_fifo_pretrig;
  }
  if (hit_persist >= max_hit_persist) {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorConfigError")
      << "+++ Value of hit_persist, " << hit_persist
      << ", exceeds max allowed, " << max_hit_persist-1 << " +++\n"
      << "+++ Try to proceed with the default value, hit_persist="
      << def_hit_persist << " +++\n";
    hit_persist = def_hit_persist;
  }
  if (drift_delay >= max_drift_delay) {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorConfigError")
      << "+++ Value of drift_delay, " << drift_delay
      << ", exceeds max allowed, " << max_drift_delay-1 << " +++\n"
      << "+++ Try to proceed with the default value, drift_delay="
      << def_drift_delay << " +++\n";
    drift_delay = def_drift_delay;
  }
  if (nplanes_hit_pretrig >= max_nplanes_hit_pretrig) {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorConfigError")
      << "+++ Value of nplanes_hit_pretrig, " << nplanes_hit_pretrig
      << ", exceeds max allowed, " << max_nplanes_hit_pretrig-1 << " +++\n"
      << "+++ Try to proceed with the default value, nplanes_hit_pretrig="
      << def_nplanes_hit_pretrig << " +++\n";
    nplanes_hit_pretrig = def_nplanes_hit_pretrig;
  }
  if (nplanes_hit_pattern >= max_nplanes_hit_pattern) {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorConfigError")
      << "+++ Value of nplanes_hit_pattern, " << nplanes_hit_pattern
      << ", exceeds max allowed, " << max_nplanes_hit_pattern-1 << " +++\n"
      << "+++ Try to proceed with the default value, nplanes_hit_pattern="
      << def_nplanes_hit_pattern << " +++\n";
    nplanes_hit_pattern = def_nplanes_hit_pattern;
  }

  if (isTMB07) {
    if (pid_thresh_pretrig >= max_pid_thresh_pretrig) {
      if (infoV >= 0) edm::LogError("L1CSCTPEmulatorConfigError")
	<< "+++ Value of pid_thresh_pretrig, " << pid_thresh_pretrig
	<< ", exceeds max allowed, " << max_pid_thresh_pretrig-1 << " +++\n"
	<< "+++ Try to proceed with the default value, pid_thresh_pretrig="
	<< def_pid_thresh_pretrig << " +++\n";
      pid_thresh_pretrig = def_pid_thresh_pretrig;
    }
    if (min_separation >= max_min_separation) {
      if (infoV >= 0) edm::LogError("L1CSCTPEmulatorConfigError")
	<< "+++ Value of min_separation, " << min_separation
	<< ", exceeds max allowed, " << max_min_separation-1 << " +++\n"
	<< "+++ Try to proceed with the default value, min_separation="
	<< def_min_separation << " +++\n";
      min_separation = def_min_separation;
    }
  }
  
  if (tmb_l1a_window_size >= max_tmb_l1a_window_size) {
    if (infoV > 0) edm::LogError("L1CSCTPEmulatorConfigError")
      << "+++ Value of tmb_l1a_window_size, " << tmb_l1a_window_size
      << ", exceeds max allowed, " << max_tmb_l1a_window_size-1 << " +++\n"
      << "+++ Try to proceed with the default value, tmb_l1a_window_size="
      << def_tmb_l1a_window_size << " +++\n";
    tmb_l1a_window_size = def_tmb_l1a_window_size;
  }
}

void CSCCathodeLCTProcessor::clear() {
  thePreTriggerBXs.clear();
  for (int bx = 0; bx < MAX_CLCT_BINS; bx++) {
    bestCLCT[bx].clear();
    secondCLCT[bx].clear();
  }
}

std::vector<CSCCLCTDigi>
CSCCathodeLCTProcessor::run(const CSCComparatorDigiCollection* compdc) {
  // This is the version of the run() function that is called when running
  // over the entire detector.  It gets the comparator & timing info from the
  // comparator digis and then passes them on to another run() function.

  // clear(); // redundant; called by L1MuCSCMotherboard.

  static std::atomic<bool> config_dumped{false};
  if ((infoV > 0 || isSLHC) && !config_dumped) {
    //std::cerr<<"**** CLCT run parameters dump ****"<<std::endl;
    dumpConfigParams();
    config_dumped = true;
  }

  // Get the number of strips and stagger of layers for the given chamber.
  // Do it only once per chamber.
  if (numStrips == 0) {
    CSCTriggerGeomManager* theGeom = CSCTriggerGeometry::get();
    CSCChamber* chamber = theGeom->chamber(theEndcap, theStation, theSector,
					      theSubsector, theTrigChamber);
    if (chamber) {
      numStrips = chamber->layer(1)->geometry()->numberOfStrips();
      // ME1/a is known to the readout hardware as strips 65-80 of ME1/1.
      // Still need to decide whether we do any special adjustments to
      // reconstruct LCTs in this region (3:1 ganged strips); for now, we
      // simply allow for hits in ME1/a and apply standard reconstruction
      // to them.
      // For SLHC ME1/1 is set to have 4 CFEBs in ME1/b and 3 CFEBs in ME1/a
      if (isME11) {
	if (!smartME1aME1b && !disableME1a && theRing == 1 && !gangedME1a) numStrips = 112;
	if (!smartME1aME1b && !disableME1a && theRing == 1 && gangedME1a) numStrips = 80;
	if (!smartME1aME1b &&  disableME1a && theRing == 1 ) numStrips = 64;
	if ( smartME1aME1b && !disableME1a && theRing == 1 ) numStrips = 64;
	if ( smartME1aME1b && !disableME1a && theRing == 4 ) {
	  if (gangedME1a) numStrips = 16;
	  else numStrips = 48;
	}
      }

      if (numStrips > CSCConstants::MAX_NUM_STRIPS_7CFEBS) {
	if (infoV >= 0) edm::LogError("L1CSCTPEmulatorSetupError")
	  << "+++ Number of strips, " << numStrips
	  << " found in ME" << ((theEndcap == 1) ? "+" : "-")
	  << theStation << "/" << theRing << "/" << theChamber
	  << " (sector " << theSector << " subsector " << theSubsector
	  << " trig id. " << theTrigChamber << ")"
	  << " exceeds max expected, " << CSCConstants::MAX_NUM_STRIPS_7CFEBS
	  << " +++\n" 
	  << "+++ CSC geometry looks garbled; no emulation possible +++\n";
	numStrips = -1;
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
	stagger[i_layer] =
	  (chamber->layer(i_layer+1)->geometry()->stagger() + 1) / 2;
      }
    }
    else {
      if (infoV >= 0) edm::LogError("L1CSCTPEmulatorConfigError")
	<< " ME" << ((theEndcap == 1) ? "+" : "-")
        << theStation << "/" << theRing << "/" << theChamber
	<< " (sector " << theSector << " subsector " << theSubsector
	<< " trig id. " << theTrigChamber << ")"
	<< " is not defined in current geometry! +++\n"
	<< "+++ CSC geometry looks garbled; no emulation possible +++\n";
      numStrips = -1;
    }
  }

  if (numStrips < 0) {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorConfigError")
      << " ME" << ((theEndcap == 1) ? "+" : "-")
      << theStation << "/" << theRing << "/" << theChamber
      << " (sector " << theSector << " subsector " << theSubsector
      << " trig id. " << theTrigChamber << "):"
      << " numStrips = " << numStrips << "; CLCT emulation skipped! +++";
    std::vector<CSCCLCTDigi> emptyV;
    return emptyV;
  }

  // Get comparator digis in this chamber.
  bool noDigis = getDigis(compdc);

  if (!noDigis) {
    // Get halfstrip (and possibly distrip) times from comparator digis.
    std::vector<int>
      halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS];
    std::vector<int>
      distrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS];
    if (isTMB07) { // TMB07 (latest) version: halfstrips only.
      readComparatorDigis(halfstrip);
    }
    else { // Earlier versions: halfstrips and distrips.
      readComparatorDigis(halfstrip, distrip);
    }

    // Pass arrays of halfstrips and distrips on to another run() doing the
    // LCT search.
    // If the number of layers containing digis is smaller than that
    // required to trigger, quit right away.  (If LCT-based digi suppression
    // is implemented one day, this condition will have to be changed
    // to the number of planes required to pre-trigger.)
    unsigned int layersHit = 0;
    for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
      for (int i_hstrip = 0; i_hstrip < CSCConstants::NUM_HALF_STRIPS_7CFEBS;
	   i_hstrip++) {
	if (!halfstrip[i_layer][i_hstrip].empty()) {layersHit++; break;}
      }
    }
    // Run the algorithm only if the probability for the pre-trigger
    // to fire is not null.  (Pre-trigger decisions are used for the
    // strip read-out conditions in DigiToRaw.)
    if (layersHit >= nplanes_hit_pretrig) run(halfstrip, distrip);
  }

  // Return vector of CLCTs.
  std::vector<CSCCLCTDigi> tmpV = getCLCTs();
  return tmpV;
}

void CSCCathodeLCTProcessor::run(
  const std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
  const std::vector<int> distrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]) {
  // This version of the run() function can either be called in a standalone
  // test, being passed the halfstrip and distrip times, or called by the 
  // run() function above.  It uses the findLCTs() method to find vectors
  // of LCT candidates. These candidates are sorted and the best two per bx
  // are returned.
  std::vector<CSCCLCTDigi> LCTlist;

  if (isTMB07) {
    // Upgrade version for ME11 with better dead-time handling
    if (isSLHC && smartME1aME1b && isME11 && use_dead_time_zoning) LCTlist = findLCTsSLHC(halfstrip);
    // TMB07 version of the CLCT algorithm.
    else LCTlist = findLCTs(halfstrip);
  }
  else if (isMTCC) { // MTCC version.
    LCTlist = findLCTs(halfstrip, distrip);
  }
  else { // Idealized algorithm of many years ago.
    std::vector<CSCCLCTDigi> halfStripLCTs = findLCTs(halfstrip, 1);
    std::vector<CSCCLCTDigi> diStripLCTs   = findLCTs(distrip,   0);
    // Put all the candidates into a single vector and sort them.
    for (unsigned int i = 0; i < halfStripLCTs.size(); i++)
      LCTlist.push_back(halfStripLCTs[i]);
    for (unsigned int i = 0; i < diStripLCTs.size(); i++)
      LCTlist.push_back(diStripLCTs[i]);
  }

  // LCT sorting.
  if (LCTlist.size() > 1)
    sort(LCTlist.begin(), LCTlist.end(), std::greater<CSCCLCTDigi>());

  // Take the best two candidates per bx.
  for (std::vector<CSCCLCTDigi>::const_iterator plct = LCTlist.begin();
       plct != LCTlist.end(); plct++) {
    int bx = plct->getBX();
    if (bx >= MAX_CLCT_BINS) {
      if (infoV > 0) edm::LogWarning("L1CSCTPEmulatorOutOfTimeCLCT")
	<< "+++ Bx of CLCT candidate, " << bx << ", exceeds max allowed, "
	<< MAX_CLCT_BINS-1 << "; skipping it... +++\n";
      continue;
    }

    if (!bestCLCT[bx].isValid()) bestCLCT[bx] = *plct;
    else if (!secondCLCT[bx].isValid()) {
      // Ignore CLCT if it is the same as the best (i.e. if the same
      // CLCT was found in both half- and di-strip pattern search).
      // This can never happen in the test beam and MTCC
      // implementations.
      if (!isMTCC && !isTMB07 && *plct == bestCLCT[bx]) continue;
      secondCLCT[bx] = *plct;
    }
  }

  for (int bx = 0; bx < MAX_CLCT_BINS; bx++) {
    if (bestCLCT[bx].isValid()) {
      bestCLCT[bx].setTrknmb(1);
      if (infoV > 0) LogDebug("CSCCathodeLCTProcessor")
	<< bestCLCT[bx] << " found in ME" << ((theEndcap == 1) ? "+" : "-")
        << theStation << "/" << theRing << "/" << theChamber
	<< " (sector " << theSector << " subsector " << theSubsector
	<< " trig id. " << theTrigChamber << ")" << "\n";
    }
    if (secondCLCT[bx].isValid()) {
      secondCLCT[bx].setTrknmb(2);
      if (infoV > 0) LogDebug("CSCCathodeLCTProcessor")
	<< secondCLCT[bx] << " found in ME" << ((theEndcap == 1) ? "+" : "-")
        << theStation << "/" << theRing << "/" << theChamber
	<< " (sector " << theSector << " subsector " << theSubsector
	<< " trig id. " << theTrigChamber << ")" << "\n";
    }
  }
  // Now that we have our best CLCTs, they get correlated with the best
  // ALCTs and then get sent to the MotherBoard.  -JM
}

bool CSCCathodeLCTProcessor::getDigis(const CSCComparatorDigiCollection* compdc) {
  bool noDigis = true;

  // Loop over layers and save comparator digis on each one into digiV[layer].
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    digiV[i_layer].clear();
    
    CSCDetId detid(theEndcap, theStation, theRing, theChamber, i_layer+1);
    getDigis(compdc, detid);

    // If this is ME1/1, fetch digis in corresponding ME1/A (ring=4) as well.
    if (theStation == 1 && theRing == 1 && !disableME1a && !smartME1aME1b) {
      CSCDetId detid_me1a(theEndcap, theStation, 4, theChamber, i_layer+1);
      getDigis(compdc, detid_me1a);
    }

    // If this is ME1/1, fetch digis in corresponding ME1/B (ring=1) as well.
    // needed only for the "smart" A/B case; and, actually, only for data
    if (theStation == 1 && theRing == 4 && !disableME1a && smartME1aME1b 
	&& digiV[i_layer].empty()) {
      CSCDetId detid_me1b(theEndcap, theStation, 1, theChamber, i_layer+1);
      getDigis(compdc, detid_me1b);
    }

    if (!digiV[i_layer].empty()) {
      noDigis = false;
      if (infoV > 1) {
	LogTrace("CSCCathodeLCTProcessor")
	  << "found " << digiV[i_layer].size()
	  << " comparator digi(s) in layer " << i_layer << " of ME"
	  << ((theEndcap == 1) ? "+" : "-") << theStation << "/" << theRing
	  << "/" << theChamber << " (trig. sector " << theSector
	  << " subsector " << theSubsector << " id " << theTrigChamber << ")";
      }
    }
  }

  return noDigis;
}

void CSCCathodeLCTProcessor::getDigis(const CSCComparatorDigiCollection* compdc,
				      const CSCDetId& id) {
  bool me1bProc = theStation == 1 && theRing == 1;
  bool me1aProc = theStation == 1 && theRing == 4;
  bool me1b = (id.station() == 1) && (id.ring() == 1);
  bool me1a = (id.station() == 1) && (id.ring() == 4);
  const CSCComparatorDigiCollection::Range rcompd = compdc->get(id);
  for (CSCComparatorDigiCollection::const_iterator digiIt = rcompd.first;
       digiIt != rcompd.second; ++digiIt) {
    unsigned int origStrip = digiIt->getStrip();
    unsigned int maxStripsME1a = gangedME1a ? 16 : 48;
    if (me1a && origStrip <= maxStripsME1a && !disableME1a && !smartME1aME1b) {
      // Move ME1/A comparators from CFEB=0 to CFEB=4 if this has not
      // been done already.
      CSCComparatorDigi digi_corr(origStrip+64,
				  digiIt->getComparator(),
				  digiIt->getTimeBinWord());
      digiV[id.layer()-1].push_back(digi_corr);
    }
    else if (smartME1aME1b && (me1bProc || me1aProc)){
      //stay within bounds; in data all comps are in ME11B DetId

      if (me1aProc && me1b && origStrip > 64){//this is data
	//shift back to start from 1
	CSCComparatorDigi digi_corr(origStrip-64,
				    digiIt->getComparator(),
				    digiIt->getTimeBinWord());
	digiV[id.layer()-1].push_back(digi_corr);
      } else if ((me1bProc && me1b && origStrip <= 64)
		 || ((me1aProc && me1a))//this is MC for ME11a
		 ){
	digiV[id.layer()-1].push_back(*digiIt);
      }
    }
    else {
      digiV[id.layer()-1].push_back(*digiIt);
    }
  }
}

void CSCCathodeLCTProcessor::readComparatorDigis(
        std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]) {
  // Single-argument version for TMB07 (halfstrip-only) firmware.
  // Takes the comparator & time info and stuffs it into halfstrip vector.
  // Multiple hits on the same strip are allowed.

  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    int i_digi = 0; // digi counter, for dumps.
    for (std::vector<CSCComparatorDigi>::iterator pld = digiV[i_layer].begin();
	 pld != digiV[i_layer].end(); pld++, i_digi++) {
      // Dump raw digi info.
      if (infoV > 1) {
	std::ostringstream strstrm;
	strstrm << "Comparator digi: comparator = " << pld->getComparator()
		<< " strip #" << pld->getStrip()
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
	if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongInput")
	  << "+++ Found comparator digi with wrong comparator value = "
	  << thisComparator << "; skipping it... +++\n";
	continue;
      }

      // Get strip number.
      int thisStrip = pld->getStrip() - 1; // count from 0
      if (thisStrip < 0 || thisStrip >= numStrips) {
	if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongInput")
	  << "+++ Found comparator digi with wrong strip number = "
	  << thisStrip
	  << " (max strips = " << numStrips << "); skipping it... +++\n";
	continue;
      }
      // 2*strip: convert strip to 1/2 strip
      // comp   : comparator output
      // stagger: stagger for this layer
      int thisHalfstrip = 2*thisStrip + thisComparator + stagger[i_layer];
      if (thisHalfstrip >= 2*numStrips + 1) {
	if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongInput")
	  << "+++ Found wrong halfstrip number = " << thisHalfstrip
	  << "; skipping this digi... +++\n";
	continue;
      }

      // Get bx times on this digi and check that they are within the bounds.
      std::vector<int> bx_times = pld->getTimeBinsOn();
      for (unsigned int i = 0; i < bx_times.size(); i++) {
	// Total number of time bins in DAQ readout is given by fifo_tbins,
	// which thus determines the maximum length of time interval.
	//
	// In TMB07 version, better data-emulator agreement is
	// achieved when hits in the first 2 time bins are excluded.
	// As of May 2009, the reasons for this are not fully
	// understood yet (the work is on-going).
	if (bx_times[i] > 1 && bx_times[i] < static_cast<int>(fifo_tbins)) {

	  if (i == 0 || (i > 0 && bx_times[i]-bx_times[i-1] >=
			 static_cast<int>(hit_persist))) {
	    // A later hit on the same strip is ignored during the
	    // number of clocks defined by the "hit_persist" parameter
	    // (i.e., 6 bx's by default).
	    if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	      << "Comp digi: layer " << i_layer+1
	      << " digi #"           << i_digi+1
	      << " strip "           << thisStrip
	      << " halfstrip "       << thisHalfstrip
	      << " distrip "         << thisStrip/2 + // [0-39]
			     ((thisStrip%2 == 1 && thisComparator == 1 && stagger[i_layer] == 1) ? 1 : 0)
	      << " time "            << bx_times[i]
	      << " comparator "      << thisComparator
	      << " stagger "         << stagger[i_layer];
	    halfstrip[i_layer][thisHalfstrip].push_back(bx_times[i]);
	  }
	  else if (i > 0) {
	    if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	      << " Skipping comparator digi: strip = " << thisStrip
	      << ", layer = " << i_layer+1 << ", bx = " << bx_times[i]
	      << ", bx of previous hit = " << bx_times[i-1];
	  }
	}
	else {
	  if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	    << "+++ Skipping comparator digi: strip = " << thisStrip
	    << ", layer = " << i_layer+1 << ", bx = " << bx_times[i] << " +++";
	}
      }
    }
  }
}

void CSCCathodeLCTProcessor::readComparatorDigis(
  std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
  std::vector<int> distrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]) {
  // Two-argument version for pre-TMB07 (halfstrip and distrips) firmware.
  // Takes the comparator & time info and stuffs it into halfstrip and (and
  // possibly distrip) vector.

  int time[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_STRIPS_7CFEBS];
  int comp[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_STRIPS_7CFEBS];
  int digiNum[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_STRIPS_7CFEBS];
  for (int i = 0; i < CSCConstants::NUM_LAYERS; i++){
    for (int j = 0; j < CSCConstants::MAX_NUM_STRIPS_7CFEBS; j++) {
      time[i][j]    = -999;
      comp[i][j]    =    0;
      digiNum[i][j] = -999;
    }
  }

  for (int i = 0; i < CSCConstants::NUM_LAYERS; i++) {
    std::vector <CSCComparatorDigi> layerDigiV = digiV[i];
    for (unsigned int j = 0; j < layerDigiV.size(); j++) {
      // Get one digi at a time for the layer.  -Jm
      CSCComparatorDigi thisDigi = layerDigiV[j];

      // Dump raw digi info
      if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	<< "Comparator digi: comparator = " << thisDigi.getComparator()
	<< " strip #" << thisDigi.getStrip()
	<< " time bin = " << thisDigi.getTimeBin();

      // Get comparator: 0/1 for left/right halfstrip for each comparator
      // that fired.
      int thisComparator = thisDigi.getComparator();
      if (thisComparator != 0 && thisComparator != 1) {
	if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongInput")
	  << "+++ Comparator digi with wrong comparator value: digi #" << j
	  << ", comparator = " << thisComparator << "; skipping it... +++\n";
	continue;
      }

      // Get strip number.
      int thisStrip = thisDigi.getStrip() - 1; // count from 0
      if (thisStrip < 0 || thisStrip >= numStrips) {
	if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongInput")
	  << "+++ Comparator digi with wrong strip number: digi #" << j
	  << ", strip = " << thisStrip
	  << ", max strips = " << numStrips << "; skipping it... +++\n";
	continue;
      }

      // Get Bx of this Digi and check that it is within the bounds
      int thisDigiBx = thisDigi.getTimeBin();

      // Total number of time bins in DAQ readout is given by fifo_tbins,
      // which thus determines the maximum length of time interval.
      if (thisDigiBx >= 0 && thisDigiBx < static_cast<int>(fifo_tbins)) {

	// If there is more than one hit in the same strip, pick one
	// which occurred earlier.
	// In reality, the second hit on the same distrip is ignored only
	// during the number of clocks defined by the "hit_persist" 
	// parameter (i.e., 6 bx's by default).  So if one simulates
	// a large number of bx's in a crowded environment, this
	// approximation here may not be sufficiently good.
	if (time[i][thisStrip] == -999 || time[i][thisStrip] > thisDigiBx) {
	  digiNum[i][thisStrip] = j;
	  time[i][thisStrip]    = thisDigiBx;
	  comp[i][thisStrip]    = thisComparator;
	  if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	    << "Comp digi: layer " << i+1
	    << " digi #"           << j+1
	    << " strip "           << thisStrip
	    << " halfstrip "       << 2*thisStrip + comp[i][thisStrip] + stagger[i]
	    << " distrip "         << thisStrip/2 + // [0-39]
	      ((thisStrip%2 == 1 && comp[i][thisStrip] == 1 && stagger[i] == 1) ? 1 : 0)
	    << " time "            <<    time[i][thisStrip]
	    << " comparator "      <<    comp[i][thisStrip]
	    << " stagger "         << stagger[i];
	}
      }
      else {
	if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	  << "+++ Skipping comparator digi: strip = " << thisStrip
	  << ", layer = " << i+1 << ", bx = " << thisDigiBx << " +++";
      }
    }
  }

  // Take the comparator & time info and stuff it into half- and di-strip
  // arrays.
  for (int i = 0; i < CSCConstants::NUM_LAYERS; i++) {
    // Use the comparator info to setup the halfstrips and distrips.  -BT
    // This loop is only for halfstrips.
    for (int j = 0; j < CSCConstants::MAX_NUM_STRIPS_7CFEBS; j++) {
      if (time[i][j] >= 0) {
	int i_halfstrip = 2*j + comp[i][j] + stagger[i];
	// 2*j    : convert strip to 1/2 strip
	// comp   : comparator output
	// stagger: stagger for this layer
	if (i_halfstrip >= 2*numStrips + 1) {
	  if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongInput")
	    << "+++ Found wrong halfstrip number = " << i_halfstrip
	    << "; skipping this digi... +++\n";
	  continue;
	}
	halfstrip[i][i_halfstrip].push_back(time[i][j]);
      }
    }

    // There are no di-strips in the 2007 version of the TMB firmware.
    if (!isTMB07) {
      // This loop is only for distrips.  We have to separate the routines
      // because triad and time arrays can be changed by the distripStagger
      // routine which could mess up the halfstrips.
      static std::atomic<int> test_iteration{0};
      for (int j = 0; j < CSCConstants::MAX_NUM_STRIPS; j++){
	if (time[i][j] >= 0) {
	  int i_distrip = j/2;
	  if (j%2 == 1 && comp[i][j] == 1 && stagger[i] == 1) {
	    // @@ Needs to be checked.
	    bool stagger_debug = (infoV > 2);
	    distripStagger(comp[i], time[i], digiNum[i], j, stagger_debug);
	  }
	  // comp[i][j] == 1	: hit on right half-strip.
	  // stagger[i] == 1	: half-strips are shifted by 1.
	  // if these conditions are met add 1; otherwise add 0.
	  // So if there is a hit on the far right half-strip, and the
	  // half-strips have been staggered to the right, then the di-strip
	  // would actually correspond to the next highest di-strip.  -JM
	  if (infoV > 2 && test_iteration == 1) {
	    testDistripStagger();
	    test_iteration++;
	  }
	  if (i_distrip >= numStrips/2 + 1) {
	    if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongInput")
	      << "+++ Found wrong distrip number = " << i_distrip
	      << "; skipping this digi... +++\n";
	    continue;
	  }
	  distrip[i][i_distrip].push_back(time[i][j]);
	}
      }
    }
  }
}

void CSCCathodeLCTProcessor::distripStagger(int stag_triad[CSCConstants::MAX_NUM_STRIPS_7CFEBS],
				   int stag_time[CSCConstants::MAX_NUM_STRIPS_7CFEBS],
				   int stag_digi[CSCConstants::MAX_NUM_STRIPS_7CFEBS],
				   int i_strip, bool debug) {
  // Author: Jason Mumford (mumford@physics.ucla.edu)
  // This routine takes care of the stagger situation where there is a hit
  // on the right half-strip of a di-strip.  If there is a stagger, then
  // we must associate that distrip with the next distrip. The situation
  // gets more complicated if the next distrip also has a hit on its right
  // half-strip.  One could imagine a whole chain of these in which case
  // we need to go into this routine recursively.  The formula is that
  // while this condition is satisfied, we enquire the next distrip,
  // until we have a hit on any other halfstrip (or triad!=3).  Then we
  // must compare the 2 different bx times and take the smallest one.
  // Afterwards, we must cycle out of the routine assigning the bx times
  // to the one strip over.

  // Used only for pre-TMB07 firmware.

  if (i_strip >= CSCConstants::MAX_NUM_STRIPS) {
    if (debug) edm::LogWarning("L1CSCTPEmulatorWrongInput")
      << "+++ Found wrong strip number = " << i_strip
      << "; cannot apply distrip staggering... +++\n";
    return;
  }

  if (debug)
    LogDebug("CSCCathodeLCTProcessor")
      << " Enter distripStagger: i_strip = " << i_strip
      << " stag_triad[i_strip] = "   << stag_triad[i_strip]
      << " stag_time[i_strip] =  "   << stag_time[i_strip]
      << " stag_triad[i_strip+2] = " << stag_triad[i_strip+2]
      << " stag_time[i_strip+2] = "  << stag_time[i_strip+2];

  // So if the next distrip has a stagger hit, go into the routine again
  // for the next distrip.
  if (i_strip+2 < CSCConstants::MAX_NUM_STRIPS && stag_triad[i_strip+2] == 1)
    distripStagger(stag_triad, stag_time, stag_digi, i_strip+2);

  // When we have reached a distrip that does not have a staggered hit,
  // if it has a hit, we compare the bx times of the
  // staggered distrip with the non-staggered distrip and we take the
  // smallest of the two and assign it to the shifted distrip time.
  if (stag_time[i_strip+2] >= 0) {
    if (stag_time[i_strip] < stag_time[i_strip+2]) {
      stag_time[i_strip+2] = stag_time[i_strip];
      stag_digi[i_strip+2] = stag_digi[i_strip];
    }
  }
  // If the next distrip did not have a hit, then we merely assign the
  // shifted time to the time associated with the staggered distrip.
  else {
    stag_time[i_strip+2] = stag_time[i_strip];
    stag_digi[i_strip+2] = stag_digi[i_strip];
  }

  // Then to prevent previous staggers from being overwritten, we assign
  // the unshifted time to -999, and then mark the triads that were shifted
  // so that we don't go into the routine ever again (such as when making
  // the next loop over strips).
  stag_time[i_strip]  = -999;
  stag_triad[i_strip] =    4;
  stag_digi[i_strip]  = -999;

  if (debug)
    LogDebug("CSCCathodeLCTProcessor")
      << " Exit  distripStagger: i_strip = " << i_strip
      << " stag_triad[i_strip] = "   << stag_triad[i_strip]
      << " stag_time[i_strip] = "    << stag_time[i_strip]
      << " stag_triad[i_strip+2] = " << stag_triad[i_strip+2]
      << " stag_time[i_strip+2] = "  << stag_time[i_strip+2];
}

// --------------------------------------------------------------------------
// The code below is a description of the idealized CLCT algorithm which
// was used in Monte Carlo studies since early ORCA days and until
// CMSSW_2_0_0 (March 2008) but was never realized in the firmware.
//
// Starting with CMSSW_3_1_0, it may no longer give the same results as
// before since old versions of overloaded < > == operators in CLCTDigi
// class were discarded.
// --------------------------------------------------------------------------
// Idealized version for MC studies.
std::vector<CSCCLCTDigi> CSCCathodeLCTProcessor::findLCTs(const std::vector<int> strip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS], int stripType)
{
  int j;
  int best_strip = 0;
  int first_bx = 999;
  const int max_lct_num = 2;
  const int adjacent_strips = 2;
  // Distrip, halfstrip pattern threshold.
  const unsigned int ptrn_thrsh[2] = {nplanes_hit_pattern, nplanes_hit_pattern};
  int highest_quality = 0;

  int keystrip_data[CSCConstants::NUM_HALF_STRIPS_7CFEBS][7];
  int final_lcts[max_lct_num];

  std::vector <CSCCLCTDigi> lctList;

  int nStrips = 0;
  if (stripType == 1)      nStrips = 2*numStrips + 1;
  else if (stripType == 0) nStrips = numStrips/2 + 1;

  if (infoV > 1) dumpDigis(strip, stripType, nStrips);

  // Send data to a pretrigger so that we don't excessively look at data
  // that won't give an LCT. If there is a pretrigger, then get all quality
  // and bend for all keystrips.
  if (preTrigger(strip, stripType, nStrips, first_bx)){
    getKeyStripData(strip, keystrip_data, nStrips, first_bx, best_strip, stripType);

    /* Set all final_lcts to impossible key_strip numbers */
    for (j = 0; j < max_lct_num; j++)
      final_lcts[j] = -999;

    // Now take the keystrip with the best quality, and do a search over the
    // rest of the strips for the next highest quality.  Do the search over a 
    // range excluding a certain number of keystrips adjacent to the original
    // best key_strip.
    final_lcts[0] = best_strip;

    for (int key_strip = 0; key_strip < (nStrips-stripType); key_strip++){
      // If indexed strip does not fall within excluded range, then continue
      if (abs(best_strip - key_strip) > adjacent_strips){
	// Match with highest quality
	if (keystrip_data[key_strip][CLCT_QUALITY] > highest_quality){
	  highest_quality = keystrip_data[key_strip][CLCT_QUALITY];
	  final_lcts[1] = key_strip;
	}
      }
    }

    for (j = 0; j < max_lct_num; j++){
      // Only report LCTs if the number of layers hit is greater than or
      // equal to the (variable) valid pattern threshold ptrn_thrsh.
      int keystrip = final_lcts[j];
      if (keystrip >= 0 &&
	  keystrip_data[keystrip][CLCT_QUALITY] >= static_cast<int>(ptrn_thrsh[stripType])) {
     	// assign the stripType here. 1 = halfstrip, 0 = distrip.
     	keystrip_data[keystrip][CLCT_STRIP_TYPE] = stripType;
	// Now make the LCT words for the 2 highest, and store them in a list
	int theHalfStrip = (keystrip_data[keystrip][CLCT_STRIP_TYPE] ?
			    keystrip_data[keystrip][CLCT_STRIP] :
			    4*keystrip_data[keystrip][CLCT_STRIP]);
	keystrip_data[keystrip][CLCT_CFEB] = theHalfStrip/32;
	int halfstrip_in_cfeb =
	  theHalfStrip - 32*keystrip_data[keystrip][CLCT_CFEB];

	CSCCLCTDigi thisLCT(1, keystrip_data[keystrip][CLCT_QUALITY],
			    keystrip_data[keystrip][CLCT_PATTERN],
			    keystrip_data[keystrip][CLCT_STRIP_TYPE],
			    keystrip_data[keystrip][CLCT_BEND],
			    halfstrip_in_cfeb,
			    keystrip_data[keystrip][CLCT_CFEB],
			    keystrip_data[keystrip][CLCT_BX]);
	if (infoV > 2) {
	  char stripType =
	    (keystrip_data[keystrip][CLCT_STRIP_TYPE] == 0) ? 'D' : 'H';
	  char bend =
	    (keystrip_data[keystrip][CLCT_BEND] == 0) ? 'L' : 'R';
	  LogTrace("CSCCathodeLCTProcessor")
	    << " Raw Find: "
	    << "Key Strip: "  << std::setw(3)
	    << keystrip_data[keystrip][CLCT_STRIP]
	    << " Pattern: "   << std::setw(2)
	    << keystrip_data[keystrip][CLCT_PATTERN]
	    << " Bend: "      << std::setw(1) << bend
	    << " Quality: "   << std::setw(1)
	    << keystrip_data[keystrip][CLCT_QUALITY]
	    << " stripType: " << std::setw(1) << stripType
	    << " BX: "        << std::setw(1)
	    << keystrip_data[keystrip][CLCT_BX];
	}
	lctList.push_back(thisLCT);
      }
    }
  }

  return lctList;
} // findLCTs -- idealized version for MC studies.


// Idealized version for MC studies.
bool CSCCathodeLCTProcessor::preTrigger(const std::vector<int> strip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
					const int stripType, const int nStrips,
					int& first_bx)
{
  static const int hs_thresh = nplanes_hit_pretrig;
  static const int ds_thresh = nplanes_hit_pretrig;

  unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS];
  int i_layer, i_strip, this_layer, this_strip;
  int hits, layers_hit;
  bool hit_layer[CSCConstants::NUM_LAYERS];

  const int pre_trigger_layer_min = (stripType == 1) ? hs_thresh : ds_thresh;

  // Fire half-strip/di-strip one-shots for hit_persist bx's (6 bx's by
  // default).
  pulseExtension(strip, nStrips, pulse);

  // Now do a loop over different bunch-crossing times.
  for (unsigned int bx_time = 0; bx_time < fifo_tbins; bx_time++) {
    // For any given bunch-crossing, start at the lowest keystrip and look for
    // the number of separate layers in the pattern for that keystrip that have
    // pulses at that bunch-crossing time.  Do the same for the next keystrip, 
    // etc.  Then do the entire process again for the next bunch-crossing, etc
    // until you find a pre-trigger.
    for (int key_strip = 0; key_strip < nStrips; key_strip++){
      // Clear variables
      hits = 0;
      layers_hit = 0;
      for (i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++)
	hit_layer[i_layer] = false;
      // Loop over pattern strips and look for hits.
      for (i_strip = 0; i_strip < NUM_PATTERN_STRIPS; i_strip++){
	this_layer = pre_hit_pattern[0][i_strip];
	this_strip = pre_hit_pattern[1][i_strip]+key_strip;
	if (this_strip >= 0 && this_strip < nStrips) {
	  // Perform bit operation to see if pulse is 1 at a certain bx_time.
	  if (((pulse[this_layer][this_strip] >> bx_time) & 1) == 1) {
	    hits++;
	    // Store number of layers hit.
	    if (hit_layer[this_layer] == false) {
	      hit_layer[this_layer] = true;
	      layers_hit++;

	      // Look if number of layers hit is greater or equal than some
	      // pre-defined threshold.
	      if (layers_hit >= pre_trigger_layer_min) {
		first_bx = bx_time;
		return true;
	      }
	    }
	  }
	}
      }
    }
  }
  // If the pretrigger was never satisfied, then return false.
  return false;
} // preTrigger -- idealized version for MC studies.


// Idealized version for MC studies.
void CSCCathodeLCTProcessor::getKeyStripData(const std::vector<int> strip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
        int keystrip_data[CSCConstants::NUM_HALF_STRIPS_7CFEBS][7],
        int nStrips, int first_bx, int& best_strip, int stripType) {
  int lct_pattern[NUM_PATTERN_STRIPS];
  int key_strip, this_layer, this_strip;
  int quality, best_quality;
  int bend = 0;
  int highest_quality = 0;
  bool nullPattern;

  for (key_strip = 0; key_strip < nStrips; key_strip++)
    for (int i = 0; i < 7; i++)
      keystrip_data[key_strip][i] = 0;

  // Now we need to look at all the keystrips and take the best pattern
  // for each.  There are multiple patterns available for each keystrip.

  for (key_strip = 0; key_strip < (nStrips-stripType); key_strip++){
    nullPattern = true;
    for (int pattern_strip = 0; pattern_strip < NUM_PATTERN_STRIPS; pattern_strip++){
      this_layer = pre_hit_pattern[0][pattern_strip];
      this_strip = pre_hit_pattern[1][pattern_strip] + key_strip;
      // This conditional statement prevents us from looking at strips
      // that don't exist along the chamber boundaries.
      if ((this_strip >= 0 && this_strip < nStrips) &&
	  !strip[this_layer][this_strip].empty()) {
	if (nullPattern) nullPattern = false;
	std::vector<int> bx_times = strip[this_layer][this_strip];
	lct_pattern[pattern_strip] = bx_times[0];
      }
      else
	lct_pattern[pattern_strip] = -999;
      }
    // do the rest only if there is at least one DIGI in the pattern for
    // this keystrip
    if (nullPattern) continue;

    // Initialize best_quality to zero so that we can look for best pattern
    // within a keystrip.
    best_quality = 0;

    // Loop over all possible patterns.
    // Loop in reverse order, in order to give priority to a straighter
    // pattern (larger pattern_num) in case of equal qualities.
    // Exclude pattern 0 since it is not defined.
    for (int pattern_num = CSCConstants::NUM_CLCT_PATTERNS_PRE_TMB07-1; pattern_num > 0; pattern_num--) {
      // Get the pattern quality from lct_pattern.
      // TMB latches LCTs drift_delay clocks after pretrigger.
      int latch_bx = first_bx + drift_delay;
      getPattern(pattern_num, lct_pattern, latch_bx, quality, bend);
      if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	<< "Key_strip " << key_strip << " quality of pattern_num "
	<< pattern_num << ": " << quality;
      if (quality > best_quality){
	// Store the best pattern, quality, etc., for each key_strip.
	keystrip_data[key_strip][CLCT_PATTERN] = pattern_num;
	keystrip_data[key_strip][CLCT_BEND]    = bend;
	keystrip_data[key_strip][CLCT_STRIP]   = key_strip;
	keystrip_data[key_strip][CLCT_BX]      = first_bx;
	// keystrip_data[key_strip][CLCT_STRIP_TYPE] = stripType; //assign the stripType elsewhere
	keystrip_data[key_strip][CLCT_QUALITY] = quality;
	if (quality > highest_quality){
	  // Keep track of which strip had the highest quality.
	  // highest_quality refers to the overall highest quality for all
	  // key strips. This is different than best_quality which refers
	  // to the best quality in a keystrip from different patterns.
	  best_strip = key_strip;
	  highest_quality = quality;
	}
	best_quality = quality;
      }
    }
  }
} // getKeyStripData -- idealized version for MC studies.


// Idealized version for MC studies.
void CSCCathodeLCTProcessor::getPattern(int pattern_num,
       int strip_value[NUM_PATTERN_STRIPS], int bx_time,
       int& quality, int& bend){
  // This function takes strip values and bx_time to find out which hits fall
  // within a certain pattern.  Quality, and bend are then calculated based on
  // which strip pattern and how many layers were hit within the pattern.
  int layers_hit = 0;
  bool hit_layer[CSCConstants::NUM_LAYERS];

  // Clear hit_layer array to keep track of number of layers hit.
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++)
    hit_layer[i_layer] = false;

  // Loop over all designated patterns.
  for (int strip_num = 0; strip_num < NUM_PATTERN_STRIPS; strip_num++){
    if (hitIsGood(strip_value[strip_num], bx_time)){
      for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++){
	// Loop over layer and see if corresponding strip is on same layer
	// If so then increment number of hits.
	if (i_layer == pattern[pattern_num][strip_num]){
	  // If layer has no hits, then increment number of layers hit.
	  if (hit_layer[i_layer] == false){
	    layers_hit++;
	    hit_layer[i_layer] = true;
	  }
	}
      }
    }
  }
  // Get bend value from pattern.
  bend = pattern[pattern_num][NUM_PATTERN_STRIPS];
  quality = layers_hit;
} // getPattern -- idealized version for MC studies.


// Idealized version for MC studies.
bool CSCCathodeLCTProcessor::hitIsGood(int hitTime, int BX) {
  // Find out if hit time is good.  Hit should have occurred no more than
  // hit_persist clocks before the latching time.
  int dt = BX - hitTime;
  if (dt >= 0 && dt <= static_cast<int>(hit_persist)) {return true;}
  else {return false;}
} // hitIsGood -- idealized version for MC studies.


// --------------------------------------------------------------------------
// The code below is a description of the pre-2007 version of the CLCT
// algorithm.  It was used in numerous CSC test beams and MTCC for
// firmware-emulator comparisons, but due to a number of known limitations
// it was never used in Monte Carlo studies.
//
// Starting with CMSSW_3_1_0, it may no longer give the same results as
// before since old versions of overloaded < > == operators in CLCTDigi
// class were discarded.
// --------------------------------------------------------------------------
// Pre-2007 version.
std::vector <CSCCLCTDigi> CSCCathodeLCTProcessor::findLCTs(
 const std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
 const std::vector<int> distrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]) {
  std::vector <CSCCLCTDigi> lctList;
  int _bx[2] = {999, 999};
  int first_bx = 999;

  const int nhStrips = 2*numStrips + 1;
  const int ndStrips = numStrips/2 + 1;

  if (infoV > 1) {
    dumpDigis(halfstrip, 1, nhStrips);
    dumpDigis(distrip,   0, ndStrips);
  }

  // Test beam version of TMB pretrigger and LCT sorting
  int h_keyStrip[MAX_CFEBS];       // one key per CFEB
  unsigned int h_nhits[MAX_CFEBS]; // number of hits in envelope for each key
  int d_keyStrip[MAX_CFEBS];       // one key per CFEB
  unsigned int d_nhits[MAX_CFEBS]; // number of hits in envelope for each key
  int keystrip_data[2][7];    // 2 possible LCTs per CSC x 7 LCT quantities
  unsigned int h_pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]; // simulate digital one-shot
  unsigned int d_pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]; // simulate digital one-shot
  bool pre_trig[2] = {false, false};

  // All half-strip and di-strip pattern envelopes are evaluated
  // simultaneously, on every clock cycle.
  pre_trig[0] = preTrigger(halfstrip, h_pulse, 1, nhStrips, 0, _bx[0]);
  pre_trig[1] = preTrigger(  distrip, d_pulse, 0, ndStrips, 0, _bx[1]);

  // If any of 200 half-strip and di-strip envelopes has enough layers hit in
  // it, TMB will pre-trigger.
  if (pre_trig[0] || pre_trig[1]) {
    first_bx = (_bx[0] < _bx[1]) ? _bx[0] : _bx[1];
    if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
      << "half bx " << _bx[0] << " di bx " << _bx[1] << " first " << first_bx
      << "\n ..... waiting drift delay ..... ";

    // Empirically-found trick allowing to dramatically improve agreement
    // with MTCC-II data.
    // The trick is to ignore hits in a few first time bins when latching
    // hits for priority encode envelopes.  For MTCC-II, we need to ignore
    // hits in time bins 0-3 inclusively.
    //
    // Firmware configuration has been fixed for most of 2007 runs, so
    // this trick should NOT be used when emulating 2007 trigger.
    /*
    int max_bx = 4;
    for (int ilayer = 0; ilayer < CSCConstants::NUM_LAYERS; ilayer++) {
      for (int istrip = 0; istrip < CSCConstants::NUM_HALF_STRIPS; istrip++) {
	for (int bx = 0; bx < max_bx; bx++) {
	  if (((h_pulse[ilayer][istrip] >> bx) & 1) == 1) {
	    h_pulse[ilayer][istrip] = 0;
	  }
	}
      }
      for (int istrip = 0; istrip < CSCConstants::NUM_DI_STRIPS; istrip++) {
	for (int bx = 0; bx < max_bx; bx++) {
	  if (((d_pulse[ilayer][istrip] >> bx) & 1) == 1) {
	    d_pulse[ilayer][istrip] = 0;
	  }
	}
      }
    }
    */

    // TMB latches LCTs drift_delay clocks after pretrigger.
    int latch_bx = first_bx + drift_delay;
    latchLCTs(h_pulse, h_keyStrip, h_nhits, 1, CSCConstants::NUM_HALF_STRIPS,
	      latch_bx);
    latchLCTs(d_pulse, d_keyStrip, d_nhits, 0,   CSCConstants::NUM_DI_STRIPS,
	      latch_bx);

    if (infoV > 1) {
      LogTrace("CSCCathodeLCTProcessor")
	<< "...............................\n"
	<< "Final halfstrip hits and keys (after drift delay) ...";
      for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) {
	LogTrace("CSCCathodeLCTProcessor")
	  << "cfeb " << icfeb << " key: " << h_keyStrip[icfeb]
	  << " hits " << h_nhits[icfeb];
      }
      LogTrace("CSCCathodeLCTProcessor")
	<< "Final distrip hits and keys (after drift delay) ...";
      for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) {
	LogTrace("CSCCathodeLCTProcessor")
	  << "cfeb " << icfeb << " key: " << d_keyStrip[icfeb]
	  << " hits " << d_nhits[icfeb];
      }
    }
    priorityEncode(h_keyStrip, h_nhits, d_keyStrip, d_nhits, keystrip_data);
    getKeyStripData(h_pulse, d_pulse, keystrip_data, first_bx);

    for (int ilct = 0; ilct < 2; ilct++) {
      if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	<< "found lcts: ilct " << ilct
	<< "  key strip " << keystrip_data[ilct][CLCT_STRIP];
      if (keystrip_data[ilct][CLCT_STRIP] != -1) {
	int halfstrip_in_cfeb = 0;
	if (keystrip_data[ilct][CLCT_STRIP_TYPE] == 0)
	  halfstrip_in_cfeb = 4*keystrip_data[ilct][CLCT_STRIP] -
                             32*keystrip_data[ilct][CLCT_CFEB];
	else
	  halfstrip_in_cfeb = keystrip_data[ilct][CLCT_STRIP] -
	                     32*keystrip_data[ilct][CLCT_CFEB];

	CSCCLCTDigi thisLCT(1, keystrip_data[ilct][CLCT_QUALITY],
			    keystrip_data[ilct][CLCT_PATTERN],
			    keystrip_data[ilct][CLCT_STRIP_TYPE],
			    keystrip_data[ilct][CLCT_BEND],
			    halfstrip_in_cfeb,
			    keystrip_data[ilct][CLCT_CFEB],
			    keystrip_data[ilct][CLCT_BX]);
	lctList.push_back(thisLCT);
      }
    }
  }

  return lctList;

} // findLCTs -- pre-2007 version.


// Pre-2007 version.
bool CSCCathodeLCTProcessor::preTrigger(
   const std::vector<int> strip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
   unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
					const int stripType, const int nStrips,
					const int start_bx, int& first_bx) {
  if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
    << "....................PreTrigger...........................";

  if (start_bx == 0) {
    // Fire one-shots for hit_persist bx's (6 bx's by default).
    pulseExtension(strip, nStrips, pulse);
  }

  bool pre_trig = false;
  // Now do a loop over bx times to see (if/when) track goes over threshold
  for (unsigned int bx_time = start_bx; bx_time < fifo_tbins; bx_time++) {
    // For any given bunch-crossing, start at the lowest keystrip and look for
    // the number of separate layers in the pattern for that keystrip that have
    // pulses at that bunch-crossing time.  Do the same for the next keystrip, 
    // etc.  Then do the entire process again for the next bunch-crossing, etc
    // until you find a pre-trigger.
    pre_trig = preTrigLookUp(pulse, stripType, nStrips, bx_time);
    if (pre_trig) {
      first_bx = bx_time; // bx at time of pretrigger
      return true;
    }
  } // end loop over bx times

  if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
    << "no pretrigger for strip type " << stripType << ", returning \n";
  first_bx = fifo_tbins;
  return false;
} // preTrigger -- pre-2007 version.


// Pre-2007 version.
bool CSCCathodeLCTProcessor::preTrigLookUp(
	   const unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
	   const int stripType, const int nStrips,
	   const unsigned int bx_time) {
  static const int hs_thresh = nplanes_hit_pretrig;
  static const int ds_thresh = nplanes_hit_pretrig;

  bool hit_layer[CSCConstants::NUM_LAYERS];
  int key_strip, this_layer, this_strip, layers_hit;

  // Layers hit threshold for pretrigger
  const int pre_trigger_layer_min = (stripType == 1) ? hs_thresh : ds_thresh;

  if (stripType != 0 && stripType != 1) {
    if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongInput")
      << "+++ preTrigLookUp: stripType = " << stripType
      << " does not correspond to half-strip/di-strip patterns! +++\n";
    return false;
  }

  for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) { // loop over cfebs
    // Loop over (di-/half-)strips in CFEB.
    for (int istrip = 0; istrip < cfeb_strips[stripType]; istrip++) {
      // Calculate candidate key.
      key_strip = icfeb*cfeb_strips[stripType] + istrip;
      layers_hit = 0;
      for (int ilayer = 0; ilayer < CSCConstants::NUM_LAYERS; ilayer++)
	hit_layer[ilayer] = false;

      // Loop over strips in pretrigger pattern mask and look for hits.
      for (int pstrip = 0; pstrip < NUM_PATTERN_STRIPS; pstrip++) {
	this_layer = pre_hit_pattern[0][pstrip];
	this_strip = pre_hit_pattern[1][pstrip]+key_strip;

	if (this_strip >= 0 && this_strip < nStrips) {
	  // Determine if "one shot" is high at this bx_time
	  if (((pulse[this_layer][this_strip] >> bx_time) & 1) == 1) {
	    if (hit_layer[this_layer] == false) {
	      hit_layer[this_layer] = true;
	      layers_hit++;                  // determines number of layers hit
	      if (layers_hit >= pre_trigger_layer_min) {
		if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
		  << "pretrigger at bx: " << bx_time
		  << ", cfeb " << icfeb << ", returning";
		return true;
	      }
	    }
	  }
	}
      } // end loop over strips in pretrigger pattern
    } // end loop over candidate key strips in cfeb
  } // end loop over cfebs, if pretrigger is found, stop looking and return

  return false;

} // preTrigLookUp -- pre-2007 version.


// Pre-2007 version.
void CSCCathodeLCTProcessor::latchLCTs(
	   const unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
	   int keyStrip[MAX_CFEBS], unsigned int n_hits[MAX_CFEBS],
	   const int stripType, const int nStrips, const int bx_time) {

  bool hit_layer[CSCConstants::NUM_LAYERS];
  int key_strip, this_layer, this_strip;
  int layers_hit, prev_hits;

  for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) {
    keyStrip[icfeb] = -1;
    n_hits[icfeb]   =  0;
  }

  if (stripType != 0 && stripType != 1) {
    if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongInput")
      << "+++ latchLCTs: stripType = " << stripType
      << " does not correspond to half-strip/di-strip patterns! +++\n";
    return;
  }

  for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) { // loop over CFEBs
    prev_hits = 0;
    // Loop over (di-/half-)strips in CFEB.
    for (int istrip = 0; istrip < cfeb_strips[stripType]; istrip++) {
      // Calculate candidate key.
      key_strip = icfeb*cfeb_strips[stripType] + istrip;
      layers_hit = 0;
      for (int ilayer = 0; ilayer < CSCConstants::NUM_LAYERS; ilayer++)
	hit_layer[ilayer] = false;

      // Loop over strips in pretrigger pattern mask and look for hits.
      for (int pstrip = 0; pstrip < NUM_PATTERN_STRIPS; pstrip++) {
	this_layer = pre_hit_pattern[0][pstrip];
	this_strip = pre_hit_pattern[1][pstrip]+key_strip;

	if (this_strip >= 0 && this_strip < nStrips) {
	  // Determine if "one shot" is high at this bx_time
	  if (((pulse[this_layer][this_strip] >> bx_time) & 1) == 1) {
	    if (hit_layer[this_layer] == false) {
	      hit_layer[this_layer] = true;
	      layers_hit++;                  // number of layers hit
	    }
	  }
	}
      } // end loop over strips in pretrigger pattern
      if (infoV > 1) {
	if (layers_hit > 0) LogTrace("CSCCathodeLCTProcessor")
	  << "cfeb: " << icfeb << "  key_strip: " << key_strip
	  << "  n_hits: " << layers_hit;
      }
      // If two or more keys have an equal number of hits, the lower number
      // key is taken.  Hence, replace the previous key only if this key has
      // more hits.
      if (layers_hit > prev_hits) {
	prev_hits = layers_hit;
	keyStrip[icfeb] = key_strip;  // key with highest hits is LCT key strip
	n_hits[icfeb] = layers_hit;   // corresponding hits in envelope
      }
    }  // end loop over candidate key strips in cfeb
  }  // end loop over cfebs
} // latchLCTs -- pre-2007 version.


// Pre-2007 version.
void CSCCathodeLCTProcessor::priorityEncode(
        const int h_keyStrip[MAX_CFEBS], const unsigned int h_nhits[MAX_CFEBS],
	const int d_keyStrip[MAX_CFEBS], const unsigned int d_nhits[MAX_CFEBS],
	int keystrip_data[2][7]) {
  static const unsigned int hs_thresh = nplanes_hit_pretrig;
  //static const unsigned int ds_thresh = nplanes_hit_pretrig;

  int ihits[2]; // hold hits for sorting
  int cfebs[2]; // holds CFEB numbers corresponding to highest hits
  const int nlcts = 2;
  int key_strip[MAX_CFEBS], key_phits[MAX_CFEBS], strip_type[MAX_CFEBS];

  // initialize arrays
  for (int ilct = 0; ilct < nlcts; ilct++) {
    for (int j = 0; j < 7; j++) keystrip_data[ilct][j] = -1;
    ihits[ilct] = 0;
    cfebs[ilct] = -1;
  }
  for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) {
    key_strip[icfeb]  = -1;
    key_phits[icfeb]  = -1;
    strip_type[icfeb] = -1;
  }

  if (infoV > 1) {
    LogTrace("CSCCathodeLCTProcessor")
      << ".....................PriorityEncode.......................";
    std::ostringstream strstrm;
    strstrm << "hkeys:";
    for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) {
      strstrm << std::setw(4) << h_keyStrip[icfeb];
    }
    strstrm << "\ndkeys:";
    for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) {
      strstrm << std::setw(4) << d_keyStrip[icfeb];
    }
    LogTrace("CSCCathodeLCTProcessor") << strstrm.str();
  }

  // Loop over CFEBs and determine better of half- or di- strip pattern.
  // If select halfstrip, promote it by adding an extra bit to its hits.
  for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) {
    if (h_keyStrip[icfeb] != -1 && d_keyStrip[icfeb] != -1) {
      if (h_nhits[icfeb] >= hs_thresh) {
	key_strip[icfeb] = h_keyStrip[icfeb];
	key_phits[icfeb] = h_nhits[icfeb] + 8; // halfstrip promotion
	strip_type[icfeb]= 1;
      }
      // For di-strip envelope there is no requirement that the number of
      // layers hit is >= ds_thresh!!!
      // else if (d_nhits[icfeb] >= ds_thresh) {
      else {
	key_strip[icfeb] = d_keyStrip[icfeb];
	key_phits[icfeb] = d_nhits[icfeb];
	strip_type[icfeb]= 0;
      }
    }
    else if (h_keyStrip[icfeb] != -1) {
      if (h_nhits[icfeb] >= hs_thresh) {
	key_strip[icfeb] = h_keyStrip[icfeb];
	key_phits[icfeb] = h_nhits[icfeb] + 8; // halfstrip promotion
	strip_type[icfeb]= 1;
      }
    }
    else if (d_keyStrip[icfeb] != -1) {
      // if (d_nhits[icfeb] >= ds_thresh) {
	key_strip[icfeb] = d_keyStrip[icfeb];
	key_phits[icfeb] = d_nhits[icfeb];
	strip_type[icfeb]= 0;
      // }
    }
    if (infoV > 1 && strip_type[icfeb] != -1) {
      if (strip_type[icfeb] == 0)
	LogTrace("CSCCathodeLCTProcessor")
	  << "  taking distrip pattern on cfeb " << icfeb;
      else if (strip_type[icfeb] == 1)
	LogTrace("CSCCathodeLCTProcessor")
	  << "  taking halfstrip pattern on cfeb " << icfeb;
      LogTrace("CSCCathodeLCTProcessor")
	<< "     cfeb " << icfeb << " key " << key_strip[icfeb]
	<< " hits " << key_phits[icfeb] << " type " << strip_type[icfeb];
    }
  }

  // Remove duplicate LCTs at boundaries -- it is possilbe to have key[0]
  // be the higher of the two key strips, take this into account, but
  // preserve rank of lcts.
  int key[MAX_CFEBS];
  int loedge, hiedge;

  if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
    << "...... Remove Duplicates ......";
  for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) {
    if(strip_type[icfeb] == 0) key[icfeb] = key_strip[icfeb]*4;
    else                       key[icfeb] = key_strip[icfeb];
  }
  for (int icfeb = 0; icfeb < MAX_CFEBS-1; icfeb++) {
    if (key[icfeb] >= 0 && key[icfeb+1] >= 0) {
      loedge = cfeb_strips[1]*(icfeb*8+7)/8;
      hiedge = cfeb_strips[1]*(icfeb*8+9)/8 - 1;
      if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	<< "  key 1: " << key[icfeb] << "  key 2: " << key[icfeb+1]
	<< "  low edge:  " << loedge << "  high edge: " << hiedge;
      if (key[icfeb] >= loedge && key[icfeb+1] <= hiedge) {
	if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	  << "Duplicate LCTs found at boundary of CFEB " << icfeb << " ...";
	if (key_phits[icfeb+1] > key_phits[icfeb]) {
	  if (infoV > 1) LogTrace("CSCCathodeLCTProcessor") 
	    << "   deleting LCT on CFEB " << icfeb;
	  key_strip[icfeb] = -1;
	  key_phits[icfeb] = -1;
	}
	else {
	  if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	    << "   deleting LCT on CFEB " << icfeb+1;
	  key_strip[icfeb+1] = -1;
	  key_phits[icfeb+1] = -1;
	}
      }
    }
  }

  // Now loop over CFEBs and pick best two lcts based on no. hits in envelope.
  // In case of equal quality, select the one on lower-numbered CFEBs.
  if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
    << "\n...... Select best LCTs  ......";
  for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) {
    if (key_phits[icfeb] > ihits[0]) {
      ihits[1] = ihits[0];
      cfebs[1] = cfebs[0];
      ihits[0] = key_phits[icfeb];
      cfebs[0] = icfeb;
      if (infoV > 1) {
	std::ostringstream strstrm;
	for (int icfeb = 0; icfeb < MAX_CFEBS; icfeb++) {
	  strstrm << std::setw(4) << strip_type[icfeb];
	}
	LogTrace("CSCCathodeLCTProcessor")
	  << "strip_type" << strstrm.str()
	  << "\n best: ihits " << ihits[0] << " cfeb " << cfebs[0]
	  << " strip_type " << ((cfebs[0] >= 0) ? strip_type[cfebs[0]] : -1)
	  << "\n next: ihits " << ihits[1] << " cfeb " << cfebs[1]
	  << " strip_type " << ((cfebs[1] >= 0) ? strip_type[cfebs[1]] : -1);
      }
    }
    else if (key_phits[icfeb] > ihits[1]) {
      ihits[1] = key_phits[icfeb];
      cfebs[1] = icfeb;
      if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	<< "\n next: ihits " << ihits[1] << " cfeb " << cfebs[1]
	<< " strip_type " << ((cfebs[1] >= 0) ? strip_type[cfebs[1]] : -1);
    }
  }

  // fill lct data array key strip with 2 highest hit lcts (if they exist)
  int jlct = 0;
  for (int ilct = 0; ilct < nlcts; ilct++) {
    if (cfebs[ilct] != -1) {
      keystrip_data[jlct][CLCT_CFEB]       = cfebs[ilct];
      keystrip_data[jlct][CLCT_STRIP]      = key_strip[cfebs[ilct]];
      keystrip_data[jlct][CLCT_STRIP_TYPE] = strip_type[cfebs[ilct]];
      if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	<< "filling key: " << key_strip[cfebs[ilct]]
	<< " type: " << strip_type[cfebs[ilct]];
      jlct++;
    }
  }
} // priorityEncode -- pre-2007 version.


// Pre-2007 version.
void CSCCathodeLCTProcessor::getKeyStripData(
		const unsigned int h_pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
		const unsigned int d_pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
		int keystrip_data[2][7], const int first_bx) {

  int lct_pattern[NUM_PATTERN_STRIPS];
  int this_layer, this_strip;
  unsigned int quality = 0, bend = 0;
  unsigned int best_quality, best_pattern;
  bool valid[2] = {false,false};

  // Time at which TMB latches LCTs.
  int latch_bx = first_bx + drift_delay;

  // Look at keystrips determined from priorityEncode and find their best
  // pattern based on number of hits matching that pattern (quality).  Also
  // find bend angle.  There are multiple patterns available for each keystrip.

  if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
    << "...............getKeyStripData....................";

  for (int ilct = 0; ilct < 2; ilct++) {
    if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
      << "lct " << ilct << " keystrip " << keystrip_data[ilct][CLCT_STRIP]
      << " type " << keystrip_data[ilct][CLCT_STRIP_TYPE];
    if (keystrip_data[ilct][CLCT_STRIP] == -1) {// flag set in priorityEncode()
      if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	<< "no lct at ilct " << ilct;
      continue;
    }
    for (int pattern_strip = 0; pattern_strip < NUM_PATTERN_STRIPS;
	 pattern_strip++) {
      lct_pattern[pattern_strip] = -999;
      this_layer = pre_hit_pattern[0][pattern_strip];
      this_strip = pre_hit_pattern[1][pattern_strip] +
	keystrip_data[ilct][CLCT_STRIP];
      // This conditional statement prevents us from looking at strips
      // that don't exist along the chamber boundaries.
      if (keystrip_data[ilct][CLCT_STRIP_TYPE] == 1) {
	if (this_strip >= 0 && this_strip < CSCConstants::NUM_HALF_STRIPS) {
	  // Now look at one-shots in bx where TMB latches the LCTs
	  if (((h_pulse[this_layer][this_strip] >> latch_bx) & 1) == 1)
	    lct_pattern[pattern_strip] = 1;
	}
      }
      else {
	if (this_strip >= 0 && this_strip < CSCConstants::NUM_DI_STRIPS) {
	  // Now look at one-shots in bx where TMB latches the LCTs
	  if (((d_pulse[this_layer][this_strip] >> latch_bx) & 1) == 1)
	    lct_pattern[pattern_strip] = 1;
	}
      }
    }

    // Find best pattern and quality associated with key by looping over all 
    // possible patterns
    best_quality = 0;
    best_pattern = 0;

    for (unsigned int pattern_num = 0;
	 pattern_num < CSCConstants::NUM_CLCT_PATTERNS_PRE_TMB07; pattern_num++) {
      getPattern(pattern_num, lct_pattern, quality, bend);
      if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	  << "pattern " << pattern_num << " quality " << quality
	  << " bend " << bend;
      // Number of layers hit matching a pattern template is compared
      // to nplanes_hit_pattern.  The threshold is the same for both half- and
      // di-strip patterns.
      if (quality >= nplanes_hit_pattern) {
	// If the number of matches is the same for two pattern templates,
	// the higher pattern-template number is selected.
	if ((quality == best_quality && pattern_num > best_pattern) ||
	    (quality >  best_quality)) {
	  if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	    << "valid = true at quality " << quality
	    << "  thresh " << nplanes_hit_pattern;
	  valid[ilct] = true;
	  keystrip_data[ilct][CLCT_PATTERN]    = pattern_num;
	  keystrip_data[ilct][CLCT_BEND]       = bend;
	  keystrip_data[ilct][CLCT_BX]         = first_bx;
	  //keystrip_data[ilct][CLCT_STRIP_TYPE] = stripType;
	  keystrip_data[ilct][CLCT_QUALITY]    = quality;
	  best_quality = quality;
	  best_pattern = pattern_num;
	}
      }
    }

    if (!valid[ilct]) {
      keystrip_data[ilct][CLCT_STRIP] = -1;  // delete lct
      if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	<< "lct " << ilct << " not over threshold: deleting";
    }
    else {
      if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	<< "\n" << "--------- final LCT: " << ilct << " -------------\n"
	<< " key strip "   << keystrip_data[ilct][CLCT_STRIP]
	<< " pattern_num " << keystrip_data[ilct][CLCT_PATTERN]
	<< " quality "     << keystrip_data[ilct][CLCT_QUALITY]
	<< " bend "        << keystrip_data[ilct][CLCT_BEND]
	<< " bx "          << keystrip_data[ilct][CLCT_BX]
	<< " type "        << keystrip_data[ilct][CLCT_STRIP_TYPE] << "\n";
    }
  } // end loop over lcts
} // getKeyStripData -- pre-2007 version.


// Pre-2007 version.
void CSCCathodeLCTProcessor::getPattern(unsigned int pattern_num,
			 const int strip_value[NUM_PATTERN_STRIPS],
			 unsigned int& quality, unsigned int& bend) {

  // This function takes strip "one-shots" at the correct bx to find out
  // which hits fall within a certain pattern.  Quality and bend are then
  // calculated based on which strip pattern and how many layers were hit
  // within the pattern.

  unsigned int layers_hit = 0;
  bool hit_layer[CSCConstants::NUM_LAYERS];

  // Clear hit_layer array to keep track of number of layers hit.
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++)
    hit_layer[i_layer] = false;

  // Loop over all designated patterns.
  for (int strip_num = 0; strip_num < NUM_PATTERN_STRIPS; strip_num++){
    if (strip_value[strip_num] == 1){
      for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++){
	// Loop over layer and see if corresponding strip is on same layer
	// If so then increment number of hits.
	if (i_layer == pattern[pattern_num][strip_num]){
	  // If layer has no hits, then increment number of layers hit.
	  if (hit_layer[i_layer] == false){
	    layers_hit++;
	    hit_layer[i_layer] = true;
	  }
	}
      }
    }
  }
  // Get bend value from pattern.
  bend = pattern[pattern_num][NUM_PATTERN_STRIPS];
  quality = layers_hit;

} // getPattern -- pre-2007 version.


// --------------------------------------------------------------------------
// The code below is a description of the 2007 version of the CLCT
// algorithm (half-strips only).  It was first used in 2008 CRUZET runs,
// and later in CRAFT.  The algorithm became the default version for
// Monte Carlo studies in March 2008 (CMSSW_2_0_0).
// --------------------------------------------------------------------------
// TMB-07 version.
std::vector<CSCCLCTDigi> CSCCathodeLCTProcessor::findLCTs(const std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]) {
  std::vector<CSCCLCTDigi> lctList;

  // Max. number of half-strips for this chamber.
  const int maxHalfStrips = 2*numStrips + 1;

  if (infoV > 1) dumpDigis(halfstrip, 1, maxHalfStrips);

  // Test beam version of TMB pretrigger and LCT sorting
  enum {max_lcts = 2};
  // 2 possible LCTs per CSC x 7 LCT quantities
  int keystrip_data[max_lcts][7] = {{0}};
  unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS];

  // Fire half-strip one-shots for hit_persist bx's (4 bx's by default).
  pulseExtension(halfstrip, maxHalfStrips, pulse);

  unsigned int start_bx = start_bx_shift;
  // Stop drift_delay bx's short of fifo_tbins since at later bx's we will
  // not have a full set of hits to start pattern search anyway.
  unsigned int stop_bx  = fifo_tbins - drift_delay;
  // Allow for more than one pass over the hits in the time window.
  while (start_bx < stop_bx) {
    // All half-strip pattern envelopes are evaluated simultaneously, on every
    // clock cycle.
    int first_bx = 999;
    bool pre_trig = preTrigger(pulse, start_bx, first_bx);

    // If any of half-strip envelopes has enough layers hit in it, TMB
    // will pre-trigger.
    if (pre_trig) {
      thePreTriggerBXs.push_back(first_bx);
      if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	<< "..... pretrigger at bx = " << first_bx
	<< "; waiting drift delay .....";

      // TMB latches LCTs drift_delay clocks after pretrigger.
      int latch_bx = first_bx + drift_delay;
      bool hits_in_time = ptnFinding(pulse, maxHalfStrips, latch_bx);
      if (infoV > 1) {
	if (hits_in_time) {
	  for (int hstrip = stagger[CSCConstants::KEY_CLCT_LAYER-1];
	       hstrip < maxHalfStrips; hstrip++) {
	    if (nhits[hstrip] > 0) {
	      LogTrace("CSCCathodeLCTProcessor")
		<< " bx = " << std::setw(2) << latch_bx << " --->"
		<< " halfstrip = " << std::setw(3) << hstrip
		<< " best pid = "  << std::setw(2) << best_pid[hstrip]
		<< " nhits = "     << nhits[hstrip];
	    }
	  }
	}
      }
      // The pattern finder runs continuously, so another pre-trigger
      // could occur already at the next bx.
      start_bx = first_bx + 1;

      // Quality for sorting.
      int quality[CSCConstants::NUM_HALF_STRIPS_7CFEBS];
      int best_halfstrip[max_lcts], best_quality[max_lcts];
      for (int ilct = 0; ilct < max_lcts; ilct++) {
	best_halfstrip[ilct] = -1;
	best_quality[ilct]   =  0;
      }

      // Calculate quality from pattern id and number of hits, and
      // simultaneously select best-quality LCT.
      if (hits_in_time) {
	for (int hstrip = stagger[CSCConstants::KEY_CLCT_LAYER-1];
	     hstrip < maxHalfStrips; hstrip++) {
	  // The bend-direction bit pid[0] is ignored (left and right
	  // bends have equal quality).
	  quality[hstrip] = (best_pid[hstrip] & 14) | (nhits[hstrip] << 5);
	  if (quality[hstrip] > best_quality[0]) {
	    best_halfstrip[0] = hstrip;
	    best_quality[0]   = quality[hstrip];
	  }
	  if (infoV > 1 && quality[hstrip] > 0) {
	    LogTrace("CSCCathodeLCTProcessor")
	      << " 1st CLCT: halfstrip = " << std::setw(3) << hstrip
	      << " quality = "             << std::setw(3) << quality[hstrip]
	      << " best halfstrip = " << std::setw(3) << best_halfstrip[0]
	      << " best quality = "   << std::setw(3) << best_quality[0];
	  }
	}
      }

      // If 1st best CLCT is found, look for the 2nd best.
      if (best_halfstrip[0] >= 0) {
	// Mark keys near best CLCT as busy by setting their quality to
	// zero, and repeat the search.
	markBusyKeys(best_halfstrip[0], best_pid[best_halfstrip[0]], quality);

        for (int hstrip = stagger[CSCConstants::KEY_CLCT_LAYER-1];
	   hstrip < maxHalfStrips; hstrip++) {
	  if (quality[hstrip] > best_quality[1]) {
	    best_halfstrip[1] = hstrip;
	    best_quality[1]   = quality[hstrip];
	  }
	  if (infoV > 1 && quality[hstrip] > 0) {
	    LogTrace("CSCCathodeLCTProcessor")
	      << " 2nd CLCT: halfstrip = " << std::setw(3) << hstrip
	      << " quality = "             << std::setw(3) << quality[hstrip]
	      << " best halfstrip = " << std::setw(3) << best_halfstrip[1]
	      << " best quality = "   << std::setw(3) << best_quality[1];
	  }
	}

	// Pattern finder.
	bool ptn_trig = false;
	for (int ilct = 0; ilct < max_lcts; ilct++) {
	  int best_hs = best_halfstrip[ilct];
	  if (best_hs >= 0 && nhits[best_hs] >= nplanes_hit_pattern) {
	    ptn_trig = true;
	    keystrip_data[ilct][CLCT_PATTERN]    = best_pid[best_hs];
	    keystrip_data[ilct][CLCT_BEND]       =
	      pattern2007[best_pid[best_hs]][NUM_PATTERN_HALFSTRIPS];
	    // Remove stagger if any.
	    keystrip_data[ilct][CLCT_STRIP]      =
	      best_hs - stagger[CSCConstants::KEY_CLCT_LAYER-1];
	    keystrip_data[ilct][CLCT_BX]         = first_bx;
	    keystrip_data[ilct][CLCT_STRIP_TYPE] = 1;           // obsolete
	    keystrip_data[ilct][CLCT_QUALITY]    = nhits[best_hs];
	    keystrip_data[ilct][CLCT_CFEB]       =
	      keystrip_data[ilct][CLCT_STRIP]/cfeb_strips[1];
	    int halfstrip_in_cfeb = keystrip_data[ilct][CLCT_STRIP] -
	      cfeb_strips[1]*keystrip_data[ilct][CLCT_CFEB];

	    if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
	      << " Final selection: ilct " << ilct
	      << " key halfstrip " << keystrip_data[ilct][CLCT_STRIP]
	      << " quality "       << keystrip_data[ilct][CLCT_QUALITY]
	      << " pattern "       << keystrip_data[ilct][CLCT_PATTERN]
	      << " bx "            << keystrip_data[ilct][CLCT_BX];

	    CSCCLCTDigi thisLCT(1, keystrip_data[ilct][CLCT_QUALITY],
				keystrip_data[ilct][CLCT_PATTERN],
				keystrip_data[ilct][CLCT_STRIP_TYPE],
				keystrip_data[ilct][CLCT_BEND],
				halfstrip_in_cfeb,
				keystrip_data[ilct][CLCT_CFEB],
				keystrip_data[ilct][CLCT_BX]);
	    lctList.push_back(thisLCT);
	  }
	}

	if (ptn_trig) {
	  // Once there was a trigger, CLCT pre-trigger state machine
	  // checks the number of hits that lie on a pattern template
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
	    bool hits_in_time = ptnFinding(pulse, maxHalfStrips, bx);
	    if (hits_in_time) {
	      for (int hstrip = stagger[CSCConstants::KEY_CLCT_LAYER-1];
		   hstrip < maxHalfStrips; hstrip++) {
		if (nhits[hstrip] >= nplanes_hit_pattern) {
		  if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
		    << " State machine busy at bx = " << bx;
		  return_to_idle = false;
		  break;
		}
	      }
	    }
	    if (return_to_idle) {
	      if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
		<< " State machine returns to idle state at bx = " << bx;
	      start_bx = bx;
	      break;
	    }
	  }
	}
      }
    }
    else {
      start_bx = first_bx + 1; // no dead time
    }
  }

  return lctList;
} // findLCTs -- TMB-07 version.


// Common to all versions.
void CSCCathodeLCTProcessor::pulseExtension(
 const std::vector<int> time[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
 const int nStrips,
 unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]) {

  static const unsigned int bits_in_pulse = 8*sizeof(pulse[0][0]);

  // Clear pulse array.  This array will be used as a bit representation of
  // hit times.  For example: if strip[1][2] has a value of 3, then 1 shifted
  // left 3 will be bit pattern of pulse[1][2].  This would make the pattern
  // look like 0000000000001000.  Then add on additional bits to signify
  // the duration of a signal (hit_persist, formerly bx_width) to simulate
  // the TMB's drift delay.  So for the same pulse[1][2] with a hit_persist
  // of 3 would look like 0000000000111000.  This is similating the digital
  // one-shot in the TMB.
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++)
    for (int i_strip = 0; i_strip < nStrips; i_strip++)
      pulse[i_layer][i_strip] = 0;

  // Loop over all layers and halfstrips.
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    for (int i_strip = 0; i_strip < nStrips; i_strip++) {
      // If there is a hit, simulate digital one-shot persistence starting
      // in the bx of the initial hit.  Fill this into pulse[][].
      if (time[i_layer][i_strip].size() > 0) {
	std::vector<int> bx_times = time[i_layer][i_strip];
	for (unsigned int i = 0; i < bx_times.size(); i++) {
	  // Check that min and max times are within the allowed range.
	  if (bx_times[i] < 0 || bx_times[i] + hit_persist >= bits_in_pulse) {
	    if (infoV > 0) edm::LogWarning("L1CSCTPEmulatorOutOfTimeDigi")
	      << "+++ BX time of comparator digi (halfstrip = " << i_strip
	      << " layer = " << i_layer << ") bx = " << bx_times[i]
	      << " is not within the range (0-" << bits_in_pulse
	      << "] allowed for pulse extension.  Skip this digi! +++\n";
	    continue;
	  }
	  if (bx_times[i] >= start_bx_shift) {
	    for (unsigned int bx = bx_times[i]; bx < bx_times[i] + hit_persist; ++bx)
              pulse[i_layer][i_strip] = pulse[i_layer][i_strip] | (1 << bx);
          }
	}
      }
    }
  }
} // pulseExtension.


// TMB-07 version.
bool CSCCathodeLCTProcessor::preTrigger(
  const unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
					const int start_bx, int& first_bx) {
  if (infoV > 1) LogTrace("CSCCathodeLCTProcessor")
    << "....................PreTrigger...........................";

  // Max. number of half-strips for this chamber.
  const int nStrips = 2*numStrips + 1;

  bool pre_trig = false;
  // Now do a loop over bx times to see (if/when) track goes over threshold
  for (unsigned int bx_time = start_bx; bx_time < fifo_tbins; bx_time++) {
    // For any given bunch-crossing, start at the lowest keystrip and look for
    // the number of separate layers in the pattern for that keystrip that have
    // pulses at that bunch-crossing time.  Do the same for the next keystrip, 
    // etc.  Then do the entire process again for the next bunch-crossing, etc
    // until you find a pre-trigger.
    bool hits_in_time = ptnFinding(pulse, nStrips, bx_time);
    if (hits_in_time) {
      for (int hstrip = stagger[CSCConstants::KEY_CLCT_LAYER-1];
	   hstrip < nStrips; hstrip++) {
	if (infoV > 1) {
	  if (nhits[hstrip] > 0) {
	    LogTrace("CSCCathodeLCTProcessor")
	      << " bx = " << std::setw(2) << bx_time << " --->"
	      << " halfstrip = " << std::setw(3) << hstrip
	      << " best pid = "  << std::setw(2) << best_pid[hstrip]
	      << " nhits = "     << nhits[hstrip];
	  }
	}
	ispretrig[hstrip] = 0;
	if (nhits[hstrip]    >= nplanes_hit_pretrig &&
	    best_pid[hstrip] >= pid_thresh_pretrig) {
	  pre_trig = true;
	  ispretrig[hstrip] = 1;
	}
      }

      if (pre_trig) {
	first_bx = bx_time; // bx at time of pretrigger
	return true;
      }
    }
  } // end loop over bx times

  if (infoV > 1) LogTrace("CSCCathodeLCTProcessor") <<
		   "no pretrigger, returning \n";
  first_bx = fifo_tbins;
  return false;
} // preTrigger -- TMB-07 version.


// TMB-07 version.
bool CSCCathodeLCTProcessor::ptnFinding(
	   const unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
	   const int nStrips, const unsigned int bx_time)
{
  if (bx_time >= fifo_tbins) return false;

  // This loop is a quick check of a number of layers hit at bx_time: since
  // most of the time it is 0, this check helps to speed-up the execution
  // substantially.
  unsigned int layers_hit = 0;
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++)
  {
    for (int i_hstrip = 0; i_hstrip < nStrips; i_hstrip++)
    {
      if (((pulse[i_layer][i_hstrip] >> bx_time) & 1) == 1)
      {
	layers_hit++;
	break;
      }
    }
  }
  if (layers_hit < nplanes_hit_pretrig) return false;

  for (int key_hstrip = 0; key_hstrip < nStrips; key_hstrip++)
  {
    best_pid[key_hstrip] = 0;
    nhits[key_hstrip] = 0;
    first_bx_corrected[key_hstrip] = -999;
  }

  // Loop over candidate key strips.
  bool hit_layer[CSCConstants::NUM_LAYERS];
  for (int key_hstrip = stagger[CSCConstants::KEY_CLCT_LAYER - 1]; key_hstrip < nStrips; key_hstrip++)
  {
    // Loop over patterns and look for hits matching each pattern.
    for (unsigned int pid = CSCConstants::NUM_CLCT_PATTERNS - 1; pid >= pid_thresh_pretrig; pid--)
    {
      layers_hit = 0;
      for (int ilayer = 0; ilayer < CSCConstants::NUM_LAYERS; ilayer++)
	hit_layer[ilayer] = false;

      double num_pattern_hits=0., times_sum=0.;
      std::multiset<int> mset_for_median;
      mset_for_median.clear();

      // Loop over halfstrips in trigger pattern mask and calculate the
      // "absolute" halfstrip number for each.
      for (int strip_num = 0; strip_num < NUM_PATTERN_HALFSTRIPS; strip_num++)
      {
	int this_layer = pattern2007[pid][strip_num];
        if (this_layer >= 0 && this_layer < CSCConstants::NUM_LAYERS)
        {
	  int this_strip = pattern2007_offset[strip_num] + key_hstrip;
	  if (this_strip >= 0 && this_strip < nStrips) {
	    if (infoV > 3) LogTrace("CSCCathodeLCTProcessor")
	      << " In ptnFinding: key_strip = " << key_hstrip
	      << " pid = " << pid << " strip_num = " << strip_num
	      << " layer = " << this_layer << " strip = " << this_strip;
	    // Determine if "one shot" is high at this bx_time
            if (((pulse[this_layer][this_strip] >> bx_time) & 1) == 1)
            {
              if (hit_layer[this_layer] == false)
              {
		hit_layer[this_layer] = true;
		layers_hit++;     // determines number of layers hit
	      }

              // find at what bx did pulse on this halsfstrip&layer have started
              // use hit_pesrist constraint on how far back we can go
              int first_bx_layer = bx_time;
              for (unsigned int dbx = 0; dbx < hit_persist; dbx++)
              {
                if (((pulse[this_layer][this_strip] >> (first_bx_layer - 1)) & 1) == 1)
                  first_bx_layer--;
                else
                  break;
              }
              times_sum += (double) first_bx_layer;
              num_pattern_hits += 1.;
              mset_for_median.insert(first_bx_layer);
              if (infoV > 2)
                LogTrace("CSCCathodeLCTProcessor") << " 1st bx in layer: " << first_bx_layer << " sum bx: " << times_sum
                    << " #pat. hits: " << num_pattern_hits;
	    }
	  }
	}
      } // end loop over strips in pretrigger pattern

      if (layers_hit > nhits[key_hstrip])
      {
	best_pid[key_hstrip] = pid;
	nhits[key_hstrip] = layers_hit;

        // calculate median
        const int sz = mset_for_median.size();
        if (sz>0){
          std::multiset<int>::iterator im = mset_for_median.begin();
          if (sz>1) std::advance(im,sz/2-1);
          if (sz==1) first_bx_corrected[key_hstrip] = *im;
          else if ((sz % 2) == 1) first_bx_corrected[key_hstrip] = *(++im);
          else first_bx_corrected[key_hstrip] = ((*im) + (*(++im)))/2;

          if (infoV > 1) {
            char bxs[300]="";
            for (im = mset_for_median.begin(); im != mset_for_median.end(); im++)
              sprintf(bxs,"%s %d", bxs, *im);
            LogTrace("CSCCathodeLCTProcessor")
              <<"bx="<<bx_time<<" bx_cor="<< first_bx_corrected[key_hstrip]<<"  bxset="<<bxs;
          }
        }

	// Do not loop over the other (worse) patterns if max. numbers of
	// hits is found.
	if (nhits[key_hstrip] == CSCConstants::NUM_LAYERS) break;
      }
    } // end loop over pid
  } // end loop over candidate key strips
  return true;
} // ptnFinding -- TMB-07 version.


// TMB-07 version.
void CSCCathodeLCTProcessor::markBusyKeys(const int best_hstrip,
					  const int best_patid,
                                int quality[CSCConstants::NUM_HALF_STRIPS_7CFEBS]) {
  int nspan = min_separation;
  int pspan = min_separation;

  // if dynamic spacing is enabled, separation is defined by pattern width
  //if (dynamic_spacing)
  //  nspan = pspan = pattern2007[best_patid][NUM_PATTERN_HALFSTRIPS+1]-1;

  for (int hstrip = best_hstrip-nspan; hstrip <= best_hstrip+pspan; hstrip++) {
    if (hstrip >= 0 && hstrip < CSCConstants::NUM_HALF_STRIPS_7CFEBS) {
      quality[hstrip] = 0;
    }
  }
} // markBusyKeys -- TMB-07 version.



// --------------------------------------------------------------------------
// The code below is for SLHC studies of the CLCT algorithm (half-strips only).
// --------------------------------------------------------------------------
// SLHC version.
std::vector<CSCCLCTDigi>
CSCCathodeLCTProcessor::findLCTsSLHC(const std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS])
{
  std::vector<CSCCLCTDigi> lctList;

  // Max. number of half-strips for this chamber.
  const int maxHalfStrips = 2 * numStrips + 1;

  if (infoV > 1) dumpDigis(halfstrip, 1, maxHalfStrips);

  enum { max_lcts = 2 };

  // keeps dead-time zones around key halfstrips of triggered CLCTs
  bool busyMap[CSCConstants::NUM_HALF_STRIPS_7CFEBS][MAX_CLCT_BINS];
  for (int i = 0; i < CSCConstants::NUM_HALF_STRIPS_7CFEBS; i++)
    for (int j = 0; j < MAX_CLCT_BINS; j++)
      busyMap[i][j] = false;

  std::vector<CSCCLCTDigi> lctListBX;

  unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS];

  // Fire half-strip one-shots for hit_persist bx's (4 bx's by default).
  pulseExtension(halfstrip, maxHalfStrips, pulse);

  unsigned int start_bx = start_bx_shift;
  // Stop drift_delay bx's short of fifo_tbins since at later bx's we will
  // not have a full set of hits to start pattern search anyway.
  unsigned int stop_bx = fifo_tbins - drift_delay;

  // Allow for more than one pass over the hits in the time window.
  // Do search in every BX
  while (start_bx < stop_bx)
  {
    lctListBX.clear();

    // All half-strip pattern envelopes are evaluated simultaneously, on every clock cycle.
    int first_bx = 999;
    bool pre_trig = preTrigger(pulse, start_bx, first_bx);

    // If any of half-strip envelopes has enough layers hit in it, TMB
    // will pre-trigger.
    if (pre_trig)
    {
      if (infoV > 1)
        LogTrace("CSCCathodeLCTProcessor") << "..... pretrigger at bx = " << first_bx << "; waiting drift delay .....";

      // TMB latches LCTs drift_delay clocks after pretrigger.
      int latch_bx = first_bx + drift_delay;
      bool hits_in_time = ptnFinding(pulse, maxHalfStrips, latch_bx);
      if (infoV > 1)
      {
        if (hits_in_time)
        {
          for (int hstrip = stagger[CSCConstants::KEY_CLCT_LAYER - 1]; hstrip < maxHalfStrips; hstrip++)
          {
            if (nhits[hstrip] > 0)
            {
              LogTrace("CSCCathodeLCTProcessor") << " bx = " << std::setw(2) << latch_bx << " --->" << " halfstrip = "
                  << std::setw(3) << hstrip << " best pid = " << std::setw(2) << best_pid[hstrip] << " nhits = " << nhits[hstrip];
            }
          }
        }
      }
      // The pattern finder runs continuously, so another pre-trigger
      // could occur already at the next bx.
      start_bx = first_bx + 1;

      // 2 possible LCTs per CSC x 7 LCT quantities per BX
      int keystrip_data[max_lcts][7] = {{0}};

      // Quality for sorting.
      int quality[CSCConstants::NUM_HALF_STRIPS_7CFEBS];
      int best_halfstrip[max_lcts], best_quality[max_lcts];
      for (int ilct = 0; ilct < max_lcts; ilct++)
      {
        best_halfstrip[ilct] = -1;
        best_quality[ilct] = 0;
      }

      bool pretrig_zone[CSCConstants::NUM_HALF_STRIPS_7CFEBS];

      // Calculate quality from pattern id and number of hits, and
      // simultaneously select best-quality LCT.
      if (hits_in_time)
      {
        // first, mark half-strip zones around pretriggers
        // that happened at the current first_bx
        for (int hstrip = 0; hstrip < CSCConstants::NUM_HALF_STRIPS_7CFEBS; hstrip++)
          pretrig_zone[hstrip] = 0;
        for (int hstrip = 0; hstrip < CSCConstants::NUM_HALF_STRIPS_7CFEBS; hstrip++)
        {
          if (ispretrig[hstrip])
          {
            int min_hs = hstrip - pretrig_trig_zone;
            int max_hs = hstrip + pretrig_trig_zone;
            if (min_hs < 0)
              min_hs = 0;
            if (max_hs > CSCConstants::NUM_HALF_STRIPS_7CFEBS - 1)
              max_hs = CSCConstants::NUM_HALF_STRIPS_7CFEBS - 1;
            for (int hs = min_hs; hs <= max_hs; hs++)
              pretrig_zone[hs] = 1;
            if (infoV > 1)
              LogTrace("CSCCathodeLCTProcessor") << " marked pretrigger halfstrip zone [" << min_hs << "," << max_hs << "]";
          }
        }

        for (int hstrip = stagger[CSCConstants::KEY_CLCT_LAYER - 1]; hstrip < maxHalfStrips; hstrip++)
        {
          // The bend-direction bit pid[0] is ignored (left and right bends have equal quality).
          quality[hstrip] = (best_pid[hstrip] & 14) | (nhits[hstrip] << 5);
          // do not consider halfstrips:
          //   - out of pretrigger-trigger zones
          //   - in busy zones from previous trigger
          if (quality[hstrip] > best_quality[0] &&
              pretrig_zone[hstrip] &&
              !busyMap[hstrip][first_bx] )
          {
            best_halfstrip[0] = hstrip;
            best_quality[0] = quality[hstrip];
            if (infoV > 1)
            {
              LogTrace("CSCCathodeLCTProcessor") << " 1st CLCT: halfstrip = " << std::setw(3) << hstrip << " quality = "
                  << std::setw(3) << quality[hstrip] << " best halfstrip = " << std::setw(3) << best_halfstrip[0]
                  << " best quality = " << std::setw(3) << best_quality[0];
            }
          }
        }
      }

      // If 1st best CLCT is found, look for the 2nd best.
      if (best_halfstrip[0] >= 0)
      {
        // Mark keys near best CLCT as busy by setting their quality to zero, and repeat the search.
        markBusyKeys(best_halfstrip[0], best_pid[best_halfstrip[0]], quality);

        for (int hstrip = stagger[CSCConstants::KEY_CLCT_LAYER - 1]; hstrip < maxHalfStrips; hstrip++)
        {
          if (quality[hstrip] > best_quality[1] &&
              pretrig_zone[hstrip] &&
              !busyMap[hstrip][first_bx] )
          {
            best_halfstrip[1] = hstrip;
            best_quality[1] = quality[hstrip];
            if (infoV > 1)
            {
              LogTrace("CSCCathodeLCTProcessor") << " 2nd CLCT: halfstrip = " << std::setw(3) << hstrip << " quality = "
                  << std::setw(3) << quality[hstrip] << " best halfstrip = " << std::setw(3) << best_halfstrip[1]
                  << " best quality = " << std::setw(3) << best_quality[1];
            }
          }
        }

        // Pattern finder.
        bool ptn_trig = false;
        for (int ilct = 0; ilct < max_lcts; ilct++)
        {
          int best_hs = best_halfstrip[ilct];
          if (best_hs >= 0 && nhits[best_hs] >= nplanes_hit_pattern)
          {
            int bx  = first_bx;
            int fbx = first_bx_corrected[best_hs];
            if (use_corrected_bx) {
              bx  = fbx;
              fbx = first_bx;
            }
            ptn_trig = true;
            keystrip_data[ilct][CLCT_PATTERN] = best_pid[best_hs];
            keystrip_data[ilct][CLCT_BEND] = pattern2007[best_pid[best_hs]][NUM_PATTERN_HALFSTRIPS];
            // Remove stagger if any.
            keystrip_data[ilct][CLCT_STRIP] = best_hs - stagger[CSCConstants::KEY_CLCT_LAYER - 1];
            keystrip_data[ilct][CLCT_BX] = bx;
            keystrip_data[ilct][CLCT_STRIP_TYPE] = 1; // obsolete
            keystrip_data[ilct][CLCT_QUALITY] = nhits[best_hs];
            keystrip_data[ilct][CLCT_CFEB] = keystrip_data[ilct][CLCT_STRIP] / cfeb_strips[1];
            int halfstrip_in_cfeb = keystrip_data[ilct][CLCT_STRIP] - cfeb_strips[1] * keystrip_data[ilct][CLCT_CFEB];

            if (infoV > 1)
              LogTrace("CSCCathodeLCTProcessor") << " Final selection: ilct " << ilct << " key halfstrip "
                  << keystrip_data[ilct][CLCT_STRIP] << " quality " << keystrip_data[ilct][CLCT_QUALITY] << " pattern "
                  << keystrip_data[ilct][CLCT_PATTERN] << " bx " << keystrip_data[ilct][CLCT_BX];

            CSCCLCTDigi thisLCT(1, keystrip_data[ilct][CLCT_QUALITY], keystrip_data[ilct][CLCT_PATTERN],
                keystrip_data[ilct][CLCT_STRIP_TYPE], keystrip_data[ilct][CLCT_BEND], halfstrip_in_cfeb,
                keystrip_data[ilct][CLCT_CFEB], keystrip_data[ilct][CLCT_BX]);
            thisLCT.setFullBX(fbx);
            lctList.push_back(thisLCT);
            lctListBX.push_back(thisLCT);
          }
        }

        // state-machine
        if (ptn_trig)
        {
          // Once there was a trigger, CLCT pre-trigger state machine checks the number of hits
          // that lie on a key halfstrip pattern template at every bx, and waits for it to drop below threshold.
          // During that time no CLCTs could be found with its key halfstrip in the area of
          // [clct_key-clct_state_machine_zone, clct_key+clct_state_machine_zone]
          // starting from first_bx+1.
          // The search for CLCTs resumes only when the number of hits on key halfstrip drops below threshold.
          for (unsigned int ilct = 0; ilct < lctListBX.size(); ilct++)
          {
            int key_hstrip = lctListBX[ilct].getKeyStrip() + stagger[CSCConstants::KEY_CLCT_LAYER - 1];

            int delta_hs = clct_state_machine_zone;
            if (dynamic_state_machine_zone)
              delta_hs = pattern2007[lctListBX[ilct].getPattern()][NUM_PATTERN_HALFSTRIPS + 1] - 1;

            int min_hstrip = key_hstrip - delta_hs;
            int max_hstrip = key_hstrip + delta_hs;

            if (min_hstrip < stagger[CSCConstants::KEY_CLCT_LAYER - 1])
              min_hstrip = stagger[CSCConstants::KEY_CLCT_LAYER - 1];
            if (max_hstrip > maxHalfStrips)
              max_hstrip = maxHalfStrips;

            if (infoV > 2)
              LogTrace("CSCCathodeLCTProcessor") << " marking post-trigger zone after bx=" << lctListBX[ilct].getBX() << " ["
                  << min_hstrip << "," << max_hstrip << "]";

            // Stop checking drift_delay bx's short of fifo_tbins since
            // at later bx's we won't have a full set of hits for a
            // pattern search anyway.
            //int stop_time = fifo_tbins - drift_delay;
            // -- no, need to extend busyMap over fifo_tbins - drift_delay
            for (size_t bx = first_bx + 1; bx < fifo_tbins; bx++)
            {
              bool busy_bx = false;
              if (bx <= (size_t)latch_bx)
                busy_bx = true; // always busy before drift time
              if (!busy_bx)
              {
                bool hits_in_time = ptnFinding(pulse, maxHalfStrips, bx);
                if (hits_in_time && nhits[key_hstrip] >= nplanes_hit_pattern)
                  busy_bx = true;
                if (infoV > 2)
                  LogTrace("CSCCathodeLCTProcessor") << "  at bx=" << bx << " hits_in_time=" << hits_in_time << " nhits="
                      << nhits[key_hstrip];
              }
              if (infoV > 2)
                LogTrace("CSCCathodeLCTProcessor") << "  at bx=" << bx << " busy=" << busy_bx;
              if (busy_bx)
                for (int hstrip = min_hstrip; hstrip <= max_hstrip; hstrip++)
                  busyMap[hstrip][bx] = true;
              else
                break;
            }
          }
        } // if (ptn_trig)
      }
    }
    else
    {
      start_bx = first_bx + 1; // no dead time
    }
  }

  return lctList;
} // findLCTs -- SLHC version.


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
  strm << " fifo_tbins   [total number of time bins in DAQ readout] = "
       << fifo_tbins << "\n";
  strm << " fifo_pretrig [start time of cathode raw hits in DAQ readout] = "
       << fifo_pretrig << "\n";
  strm << " hit_persist  [duration of signal pulse, in 25 ns bins] = "
       << hit_persist << "\n";
  strm << " drift_delay  [time after pre-trigger before TMB latches LCTs] = "
       << drift_delay << "\n";
  strm << " nplanes_hit_pretrig [min. number of layers hit for pre-trigger] = "
       << nplanes_hit_pretrig << "\n";
  strm << " nplanes_hit_pattern [min. number of layers hit for trigger] = "
       << nplanes_hit_pattern << "\n";
  if (isTMB07) {
    strm << " pid_thresh_pretrig [lower threshold on pattern id] = "
	 << pid_thresh_pretrig << "\n";
    strm << " min_separation     [region of busy key strips] = "
	 << min_separation << "\n";
  }
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  LogDebug("CSCCathodeLCTProcessor") << strm.str();
  //std::cerr<<strm.str()<<std::endl;
}

// Reasonably nice dump of digis on half-strips and di-strips.
void CSCCathodeLCTProcessor::dumpDigis(const std::vector<int> strip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS], const int stripType, const int nStrips) const
{
  LogDebug("CSCCathodeLCTProcessor")
    << "ME" << ((theEndcap == 1) ? "+" : "-")
    << theStation << "/" << theRing << "/" << theChamber
    << " strip type " << stripType << " nStrips " << nStrips;

  std::ostringstream strstrm;
  for (int i_strip = 0; i_strip < nStrips; i_strip++) {
    if (i_strip%10 == 0) {
      if (i_strip < 100) strstrm << i_strip/10;
      else               strstrm << (i_strip-100)/10;
    }
    else                 strstrm << " ";
    if ((i_strip+1)%cfeb_strips[stripType] == 0) strstrm << " ";
  }
  strstrm << "\n";
  for (int i_strip = 0; i_strip < nStrips; i_strip++) {
    strstrm << i_strip%10;
    if ((i_strip+1)%cfeb_strips[stripType] == 0) strstrm << " ";
  }
  for (int i_layer = 0; i_layer < CSCConstants::NUM_LAYERS; i_layer++) {
    strstrm << "\n";
    for (int i_strip = 0; i_strip < nStrips; i_strip++) {
      if (!strip[i_layer][i_strip].empty()) {
	std::vector<int> bx_times = strip[i_layer][i_strip];
	// Dump only the first in time.
	strstrm << std::hex << bx_times[0] << std::dec;
      }
      else {
	strstrm << "-";
      }
      if ((i_strip+1)%cfeb_strips[stripType] == 0) strstrm << " ";
    }
  }
  LogTrace("CSCCathodeLCTProcessor") << strstrm.str();
}

// Returns vector of read-out CLCTs, if any.  Starts with the vector
// of all found CLCTs and selects the ones in the read-out time window.
std::vector<CSCCLCTDigi> CSCCathodeLCTProcessor::readoutCLCTs() {
  std::vector<CSCCLCTDigi> tmpV;

  // The start time of the L1A*CLCT coincidence window should be
  // related to the fifo_pretrig parameter, but I am not completely
  // sure how.  For now, just choose it such that the window is
  // centered at bx=7.  This may need further tweaking if the value of
  // tmb_l1a_window_size changes.
  // static int fpga_latency = 3;
  // static int early_tbins  = fifo_pretrig - fpga_latency;
  // static int early_tbins = 4;
  
  // The number of CLCT bins in the read-out is given by the
  // tmb_l1a_window_size parameter, but made even by setting the LSB
  // of tmb_l1a_window_size to 0.
  //
  static std::atomic<int> lct_bins; 
    lct_bins = (tmb_l1a_window_size%2 == 0) ? tmb_l1a_window_size : tmb_l1a_window_size-1;
  static std::atomic<int> late_tbins;
    late_tbins = early_tbins + lct_bins;

  static std::atomic<int> ifois{0};
  if (ifois == 0) {
    if (infoV >= 0 && early_tbins < 0) {
      edm::LogWarning("L1CSCTPEmulatorSuspiciousParameters")
	<< "+++ early_tbins = " << early_tbins
	<< "; in-time CLCTs are not getting read-out!!! +++" << "\n";
    }

    if (late_tbins > MAX_CLCT_BINS-1) {
      if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorSuspiciousParameters")
	<< "+++ Allowed range of time bins, [0-" << late_tbins
	<< "] exceeds max allowed, " << MAX_CLCT_BINS-1 << " +++\n"
	<< "+++ Set late_tbins to max allowed +++\n";
      late_tbins = MAX_CLCT_BINS-1;
    }
    ifois = 1;
  }

  // Start from the vector of all found CLCTs and select those within
  // the CLCT*L1A coincidence window.
  int bx_readout = -1;
  std::vector<CSCCLCTDigi> all_lcts = getCLCTs();
  for (std::vector <CSCCLCTDigi>::const_iterator plct = all_lcts.begin();
       plct != all_lcts.end(); plct++) {
    if (!plct->isValid()) continue;

    int bx = (*plct).getBX();
    // Skip CLCTs found too early relative to L1Accept.
    if (bx <= early_tbins) {
      if (infoV > 1) LogDebug("CSCCathodeLCTProcessor")
	<< " Do not report CLCT on key halfstrip " << plct->getKeyStrip()
	<< ": found at bx " << bx << ", whereas the earliest allowed bx is "
	<< early_tbins+1;
      continue;
    }

    // Skip CLCTs found too late relative to L1Accept.
    if (bx > late_tbins) {
      if (infoV > 1) LogDebug("CSCCathodeLCTProcessor")
	<< " Do not report CLCT on key halfstrip " << plct->getKeyStrip()
	<< ": found at bx " << bx << ", whereas the latest allowed bx is "
	<< late_tbins;
      continue;
    }

    // If (readout_earliest_2) take only CLCTs in the earliest bx in the read-out window:
    // in digi->raw step, LCTs have to be packed into the TMB header, and
    // currently there is room just for two.
    if (readout_earliest_2) {
      if (bx_readout == -1 || bx == bx_readout) {
        tmpV.push_back(*plct);
        if (bx_readout == -1) bx_readout = bx;
      }
    }
    else tmpV.push_back(*plct);
  }
  return tmpV;
}

// Returns vector of all found CLCTs, if any.  Used for ALCT-CLCT matching.
std::vector<CSCCLCTDigi> CSCCathodeLCTProcessor::getCLCTs() {
  std::vector<CSCCLCTDigi> tmpV;
  for (int bx = 0; bx < MAX_CLCT_BINS; bx++) {
    if (bestCLCT[bx].isValid())   tmpV.push_back(bestCLCT[bx]);
    if (secondCLCT[bx].isValid()) tmpV.push_back(secondCLCT[bx]);
  }
  return tmpV;
}


// --------------------------------------------------------------------------
// Test routines.  Mostly for older versions of the algorithm and outdated.
// --------------------------------------------------------------------------
void CSCCathodeLCTProcessor::testDistripStagger() {
  // Author: Jason Mumford (mumford@physics.ucla.edu)
  // This routine tests the distripStagger routine.
  // @@
  bool debug = true;
  int test_triad[CSCConstants::NUM_DI_STRIPS], test_time[CSCConstants::NUM_DI_STRIPS];
  int test_digi[CSCConstants::NUM_DI_STRIPS];
  int distrip = 0;
  test_triad[distrip] = 3;    //After routine, I expect 4
  test_triad[distrip+1] = 3;  //                        4
  test_triad[distrip+2] = 3;  //                        4 
  test_triad[distrip+3] = 3;  //                        4
  test_triad[distrip+4] = 3;  //                        4
  test_triad[distrip+5] = 3;  //                        4
  test_triad[distrip+6] = 3;  //                        4
  test_triad[distrip+7] = 3;  //                        4
  test_triad[distrip+8] = 3;  //                        4
  test_triad[distrip+9] = 3;  //                        4
  test_triad[distrip+10] = 2;  //                       2

  test_time[distrip] = 4;     //      ""      ""        0
  test_time[distrip+1] = 10;  //                        4
  test_time[distrip+2] = 2;   //                        10
  test_time[distrip+3] = 0;   //                        2
  test_time[distrip+4] = 6;   //                        2
  test_time[distrip+5] = 8;   //                        2
  test_time[distrip+6] = 10;   //                        2
  test_time[distrip+7] = 1;   //                        2
  test_time[distrip+8] = 8;   //                        2
  test_time[distrip+9] = 5;   //                        2
  test_time[distrip+10] = 6;   //                        2

  std::cout << "\n ------------------------------------------------- \n";
  std::cout << "!!!!!!Testing distripStagger routine!!!!!!" << std::endl;
  std::cout << "Values before distripStagger routine:" << std::endl;
  for (int i=distrip; i<distrip+11; i++){
    test_digi[i] = 999;
    std::cout << "test_triad[" << i << "] = " << test_triad[i];
    std::cout << "   test_time[" << i << "] = " << test_time[i] << std::endl;
  }
  distripStagger(test_triad, test_time, test_digi, distrip, debug);
  std::cout << "Values after distripStagger routine:" << std::endl;
  for (int i=distrip; i<distrip+11; i++){
    std::cout << "test_triad[" << i << "] = " << test_triad[i];
    std::cout << "   test_time[" << i << "] = " << test_time[i] << std::endl;
  }
  std::cout << "\n ------------------------------------------------- \n \n";
}

void CSCCathodeLCTProcessor::testLCTs() {
  // test to make sure what goes into an LCT is what comes out.
  for (int ptn = 0; ptn < 8; ptn++) {
    for (int bend = 0; bend < 2; bend++) {
      for (int cfeb = 0; cfeb < MAX_CFEBS; cfeb++) {
	for (int key_strip = 0; key_strip < 32; key_strip++) {
	  for (int bx = 0; bx < 7; bx++) {
	    for (int stripType = 0; stripType < 2; stripType++) {
	      for (int quality = 3; quality < 6; quality++) {
		CSCCLCTDigi thisLCT(1, quality, ptn, stripType, bend,
				    key_strip, cfeb, bx);
		if (ptn != thisLCT.getPattern())
		  LogTrace("CSCCathodeLCTProcessor")
		    << "pattern mismatch: " << ptn << " "
		    << thisLCT.getPattern();
		if (bend != thisLCT.getBend())
		  LogTrace("CSCCathodeLCTProcessor")
		    << "bend mismatch: " << bend << " " << thisLCT.getBend();
		if (cfeb != thisLCT.getCFEB()) 
		  LogTrace("CSCCathodeLCTProcessor")
		    << "cfeb mismatch: " << cfeb << " " << thisLCT.getCFEB();
		if (key_strip != thisLCT.getKeyStrip())
		  LogTrace("CSCCathodeLCTProcessor")
		    << "strip mismatch: " << key_strip << " "
		    << thisLCT.getKeyStrip();
		if (bx != thisLCT.getBX())
		  LogTrace("CSCCathodeLCTProcessor")
		    << "bx mismatch: " << bx << " " << thisLCT.getBX();
		if (stripType != thisLCT.getStripType())
		  LogTrace("CSCCathodeLCTProcessor")
		    << "Strip Type mismatch: " << stripType << " "
		    << thisLCT.getStripType();
		if (quality != thisLCT.getQuality())
		  LogTrace("CSCCathodeLCTProcessor")
		    << "quality mismatch: " << quality << " "
		    << thisLCT.getQuality();
	      }
	    }
	  }
	}
      }
    }
  }
}

void CSCCathodeLCTProcessor::printPatterns() {
  // @@
  std::cout<<" Printing patterns for Cathode LCT"<<std::endl;
  std::cout<<"       ";
  for (int patternNum = 0; patternNum < CSCConstants::NUM_CLCT_PATTERNS_PRE_TMB07; patternNum++) {
    std::cout<<" Pattern "<<patternNum<<" ";
  }
  std::cout<<std::endl;
  std::cout<<" Layer ";
  for (int patternNum = 0; patternNum < CSCConstants::NUM_CLCT_PATTERNS_PRE_TMB07; patternNum++) {
    std::cout<<"   Bend "<<(pattern[patternNum][NUM_PATTERN_STRIPS]==0 ? "L": "R")<<"  ";
  }
  std::cout<<std::endl;
  for (int layer = 0; layer < CSCConstants::NUM_LAYERS; layer++) {
    for (int patternNum = 0; patternNum < CSCConstants::NUM_CLCT_PATTERNS_PRE_TMB07; patternNum++) {
      if (patternNum == 0) std::cout<<"   "<<layer<<"       ";
      if ((isTMB07  && layer != CSCConstants::KEY_CLCT_LAYER-1)	||
	  (!isTMB07 && layer != CSCConstants::KEY_CLCT_LAYER_PRE_TMB07-1)) {//that old counting from 1 vs 0 thing.
        int minStrip =0;
	if ((isTMB07  && layer < CSCConstants::KEY_CLCT_LAYER-1) ||
	    (!isTMB07 && layer < CSCConstants::KEY_CLCT_LAYER_PRE_TMB07-1)) {
	  minStrip = 3*layer;
	} else {
	  minStrip = 3*layer - 2;// since on the key layer we only have 1 strip
	}
        for (int strip = minStrip; strip < minStrip + 3; strip++) {
	  if (layer == pattern[patternNum][strip]) {
	    std::cout<<"X";
	  } else {
	    std::cout<<"_";
	  }
	}
      } else {// on the key layer we always have a hit, right?
	std::cout<<" X ";
      }
      std::cout<<"        ";
    }
    std::cout<<std::endl;
  }
}
    
void CSCCathodeLCTProcessor::testPatterns() {
//generate all possible combinations of hits in a given area and see what we find.
// Benn Tannenbaum 21 June 2001
  
  //there are 16 strips in our uber-pattern, each of which can be on or off.
  // 2^16 = 65536
  for (int possibleHits = 0; possibleHits < 65536; possibleHits++) {
    std::vector<int> stripsHit[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS];
    //assign one bit to each strip in an array. I'll start centered around strip 10.
    stripsHit[0][ 9].push_back(( possibleHits &     1 ) != 0);     // 2^0
    stripsHit[0][10].push_back(( possibleHits &     2 ) != 0);     // 2^1
    stripsHit[0][11].push_back(( possibleHits &     4 ) != 0);     // 2^2
    stripsHit[1][ 9].push_back(( possibleHits &     8 ) != 0);     // 2^3
    stripsHit[1][10].push_back(( possibleHits &    16 ) != 0);     // 2^4
    stripsHit[1][11].push_back(( possibleHits &    32 ) != 0);     // 2^5
    stripsHit[2][ 9].push_back(( possibleHits &    64 ) != 0);     // 2^6
    stripsHit[2][10].push_back(( possibleHits &   128 ) != 0);     // 2^7
    stripsHit[2][11].push_back(( possibleHits &   256 ) != 0);     // 2^8
    stripsHit[3][10].push_back(( possibleHits &   512 ) != 0);     // 2^9
    stripsHit[4][ 9].push_back(( possibleHits &  1024 ) != 0);     // 2^10
    stripsHit[4][10].push_back(( possibleHits &  2048 ) != 0);     // 2^11
    stripsHit[4][11].push_back(( possibleHits &  4096 ) != 0);     // 2^12
    stripsHit[5][ 9].push_back(( possibleHits &  8192 ) != 0);     // 2^13
    stripsHit[5][10].push_back(( possibleHits & 16384 ) != 0);     // 2^14
    stripsHit[5][11].push_back(( possibleHits & 32768 ) != 0);     // 2^15
    int numLayersHit = findNumLayersHit(stripsHit);
    std::vector <CSCCLCTDigi> results = findLCTs(stripsHit, 1);
// print out whatever we find-- but only ones where 4 or more layers are hit
// OR ones where we find something
// key: X    a hit there and was used to find pattern
//      x    a hit not involved in pattern
//      _    empty strip
//      o    a hit was there, but no pattern was found
    if (numLayersHit > 3 || results.size() > 0) {
      std::cout<<"Input "<<possibleHits<<"/"<< 65536 <<" # Found Patterns "<<results.size()<<std::endl<<" ";
      for (int layer = 0; layer < CSCConstants::NUM_LAYERS; layer++) {
	if ((isTMB07  && layer != CSCConstants::KEY_CLCT_LAYER - 1) || 
	    (!isTMB07 && layer != CSCConstants::KEY_CLCT_LAYER_PRE_TMB07 - 1)) {
	  for (int strip = 9; strip < 12; strip++) {
	    if (!stripsHit[layer][strip].empty()) {
	      if (results.size() > 0) {
	        int thePatternStrip = strip - (results[0].getKeyStrip() - 2) + 3*layer;
		if ((isTMB07 && layer>=CSCConstants::KEY_CLCT_LAYER) ||
		    (!isTMB07 && layer>=CSCConstants::KEY_CLCT_LAYER_PRE_TMB07))
		  thePatternStrip -= 2;

	        if (pattern[results[0].getPattern()][thePatternStrip] == layer)
		{
		  std::cout<<"X";
		} else {
		  std::cout<<"x";
		}
              } else {
	        std::cout<<"o";
              }
	    } else {
	      std::cout<<"_";
	    }
	  }
	  std::cout<<"   ";
	  for (unsigned int output = 0; output < results.size(); output++) {
	    int minStrip;
	    if ((isTMB07 && layer < CSCConstants::KEY_CLCT_LAYER-1) ||
		(!isTMB07 && layer < CSCConstants::KEY_CLCT_LAYER_PRE_TMB07-1))  {
	      minStrip = 3*layer;
	    } else {
	      minStrip = 3*layer - 2;// since on the key layer we only have 1 strip
	    }
            for (int strip = minStrip; strip < minStrip + 3; strip++) {
	      if (layer == pattern[results[output].getPattern()][strip]) {
		std::cout<<"X";
	      } else {
		std::cout<<"_";
	      }
	    }
	    std::cout<<"  ";
          }
	} else {
          if (!stripsHit[layer][10].empty()) {
	    std::cout<<" X ";
	  } else {
	    std::cout<<" _ ";
	  }
	  for (unsigned int output = 0; output < results.size(); output++)
	    std::cout<<"    X   ";
	}
	if (layer < static_cast<int>(results.size()) ) {
	  std::cout<<results[layer];
	  std::cout<<" ";
	} else {
	  std::cout<<" "<<std::endl<<" ";
	}
      }
    }
  }
}

int CSCCathodeLCTProcessor::findNumLayersHit(std::vector<int> 
          stripsHit[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]) {
  int number = 0;
  for (int layer = 0; layer < CSCConstants::NUM_LAYERS; layer++) {
    if ((!stripsHit[layer][ 9].empty()) || 
        (!stripsHit[layer][10].empty()) ||
	(!stripsHit[layer][11].empty()) ) number++;
  }
  return number;
}

//  LocalWords:  CMSSW pretrig
