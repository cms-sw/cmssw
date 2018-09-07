//-----------------------------------------------------------------------------
//
//   Class: CSCCathodeLCTProcessor
//
//   Description:
//     This class simulates the functionality of the cathode LCT card.  It is
//     run by the MotherBoard and returns up to two CathodeLCTs. It can be
//     run either in a test mode, where it is passed arrays of halfstrip
//     times, or in normal mode where it determines
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
#include "DataFormats/MuonDetId/interface/CSCTriggerNumbering.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "Geometry/CSCGeometry/interface/CSCGeometry.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <set>

//-----------------
// Static variables
//-----------------

// New set of halfstrip patterns for 2007 version of the algorithm.
// For the given pattern, set the unused parts of the pattern to 999.
// Pattern[i][CSCConstants::MAX_HALFSTRIPS_IN_PATTERN] contains bend direction.
// Bend of 0 is right/straight and bend of 1 is left.
// Pattern[i][CSCConstants::MAX_HALFSTRIPS_IN_PATTERN+1] contains pattern maximum width
const int CSCCathodeLCTProcessor::pattern2007_offset[CSCConstants::MAX_HALFSTRIPS_IN_PATTERN] =
  {  -5,  -4,  -3,  -2,  -1,   0,   1,   2,   3,   4,   5,
                    -2,  -1,   0,   1,   2,
                               0,
                    -2,  -1,   0,   1,   2,
          -4,  -3,  -2,  -1,   0,   1,   2,   3,   4,
     -5,  -4,  -3,  -2,  -1,   0,   1,   2,   3,   4,   5 };

const int CSCCathodeLCTProcessor::pattern2007[CSCConstants::NUM_CLCT_PATTERNS][CSCConstants::MAX_HALFSTRIPS_IN_PATTERN+2] = {
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
const int CSCCathodeLCTProcessor::cfeb_strips[2] = {
  CSCConstants::NUM_DISTRIPS_PER_CFEB, //8
  CSCConstants::NUM_HALF_STRIPS_PER_CFEB//32
};

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

  // Flag for SLHC studies
  isSLHC       = comm.getParameter<bool>("isSLHC");

  // shift the BX from 7 to 8
  // the unpacked real data CLCTs have central BX at bin 7
  // however in simulation the central BX  is bin 8
  // to make a proper comparison with ALCTs we need
  // CLCT and ALCT to have the central BX in the same bin
  // this shift does not affect the readout of the CLCTs
  // emulated CLCTs put in the event should be centered at bin 7 (as in data)
  alctClctOffset = comm.getParameter<unsigned int>("alctClctOffset");

  // special configuration parameters for ME11 treatment
  smartME1aME1b = comm.getParameter<bool>("smartME1aME1b");
  disableME1a = comm.getParameter<bool>("disableME1a");
  gangedME1a = comm.getParameter<bool>("gangedME1a");

  if (isSLHC && !smartME1aME1b) edm::LogError("L1CSCTPEmulatorConfigError")
    << "+++ SLHC upgrade configuration is used (isSLHC=True) but smartME1aME1b=False!\n"
    << "Only smartME1aME1b algorithm is so far supported for upgrade! +++\n";

  pid_thresh_pretrig =
    conf.getParameter<unsigned int>("clctPidThreshPretrig");
  min_separation    =
    conf.getParameter<unsigned int>("clctMinSeparation");

  start_bx_shift = conf.getParameter<int>("clctStartBxShift");

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
  if (early_tbins<0) early_tbins  = fifo_pretrig - CSCConstants::CLCT_EMUL_TIME_OFFSET;

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

  thePreTriggerDigis.clear();
}

CSCCathodeLCTProcessor::CSCCathodeLCTProcessor() :
  		     theEndcap(1), theStation(1), theSector(1),
                     theSubsector(1), theTrigChamber(1) {
  // constructor for debugging.
  static std::atomic<bool> config_dumped{false};

  // CLCT configuration parameters.
  setDefaultConfigParameters();
  infoV =  2;

  smartME1aME1b = false;
  disableME1a = false;
  gangedME1a = true;

  early_tbins = 4;

  start_bx_shift = 0;
  use_dead_time_zoning = true;
  clct_state_machine_zone = 8;

  // Check and print configuration parameters.
  checkConfigParameters();
  if (!config_dumped) {
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

  thePreTriggerDigis.clear();
}

void CSCCathodeLCTProcessor::setDefaultConfigParameters()
{
  // Set default values for configuration parameters.
  fifo_tbins   = def_fifo_tbins;
  fifo_pretrig = def_fifo_pretrig;
  hit_persist  = def_hit_persist;
  drift_delay  = def_drift_delay;
  nplanes_hit_pretrig = def_nplanes_hit_pretrig;
  nplanes_hit_pattern = def_nplanes_hit_pattern;
  pid_thresh_pretrig = def_pid_thresh_pretrig;
  min_separation     = def_min_separation;
  tmb_l1a_window_size = def_tmb_l1a_window_size;
}

// Set configuration parameters obtained via EventSetup mechanism.
void CSCCathodeLCTProcessor::setConfigParameters(const CSCDBL1TPParameters* conf)
{
  static std::atomic<bool> config_dumped{false};

  fifo_tbins   = conf->clctFifoTbins();
  fifo_pretrig = conf->clctFifoPretrig();
  hit_persist  = conf->clctHitPersist();
  drift_delay  = conf->clctDriftDelay();
  nplanes_hit_pretrig = conf->clctNplanesHitPretrig();
  nplanes_hit_pattern = conf->clctNplanesHitPattern();
  pid_thresh_pretrig = conf->clctPidThreshPretrig();
  min_separation     = conf->clctMinSeparation();

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
  thePreTriggerDigis.clear();
  thePreTriggerBXs.clear();
  for (int bx = 0; bx < CSCConstants::MAX_CLCT_TBINS; bx++) {
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
    const int ring = CSCTriggerNumbering::ringFromTriggerLabels(theStation, theTrigChamber);
    const int chid = CSCTriggerNumbering::chamberFromTriggerLabels(theSector, theSubsector, theStation, theTrigChamber);
    CSCDetId detid(theEndcap, theStation, ring, chid, 0);
    const auto& chamber = csc_g->chamber(detid);

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
    << " found in " << detid.chamberName()
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
                        << " " << CSCDetId::chamberName(theEndcap, theStation, theRing, theChamber)
                        << " (sector " << theSector << " subsector " << theSubsector
                        << " trig id. " << theTrigChamber << ")"
                        << " is not defined in current geometry! +++\n"
                        << "+++ CSC geometry looks garbled; no emulation possible +++\n";
      numStrips = -1;
    }
  }

  if (numStrips < 0) {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorConfigError")
                        << " " << CSCDetId::chamberName(theEndcap, theStation, theRing, theChamber)
                        << " (sector " << theSector << " subsector " << theSubsector
                        << " trig id. " << theTrigChamber << "):"
                        << " numStrips = " << numStrips << "; CLCT emulation skipped! +++";
    std::vector<CSCCLCTDigi> emptyV;
    return emptyV;
  }

  // Get comparator digis in this chamber.
  bool noDigis = getDigis(compdc);

  if (!noDigis) {
    // Get halfstrip times from comparator digis.
    std::vector<int>
      halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS];
    readComparatorDigis(halfstrip);

    // Pass arrays of halfstrips on to another run() doing the
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
    if (layersHit >= nplanes_hit_pretrig) run(halfstrip);
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
  for (auto& p : tmpV){
    p.setBX(p.getBX() + alctClctOffset);
  }

  return tmpV;
}

void CSCCathodeLCTProcessor::run(const std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS])
{
  // This version of the run() function can either be called in a standalone
  // test, being passed the halfstrip times, or called by the
  // run() function above.  It uses the findLCTs() method to find vectors
  // of LCT candidates. These candidates are sorted and the best two per bx
  // are returned.
  std::vector<CSCCLCTDigi> CLCTlist;

  // Upgrade version for ME11 with better dead-time handling
  if (isSLHC && smartME1aME1b && isME11 && use_dead_time_zoning) CLCTlist = findLCTsSLHC(halfstrip);
  // TMB07 version of the CLCT algorithm.
  else CLCTlist = findLCTs(halfstrip);

  // LCT sorting.
  if (CLCTlist.size() > 1)
    sort(CLCTlist.begin(), CLCTlist.end(), std::greater<CSCCLCTDigi>());

  // Take the best two candidates per bx.
  for (std::vector<CSCCLCTDigi>::const_iterator plct = CLCTlist.begin();
       plct != CLCTlist.end(); plct++) {
    int bx = plct->getBX();
    if (bx >= CSCConstants::MAX_CLCT_TBINS) {
      if (infoV > 0) edm::LogWarning("L1CSCTPEmulatorOutOfTimeCLCT")
                       << "+++ Bx of CLCT candidate, " << bx << ", exceeds max allowed, "
                       << CSCConstants::MAX_CLCT_TBINS-1 << "; skipping it... +++\n";
      continue;
    }

    if (!bestCLCT[bx].isValid()) bestCLCT[bx] = *plct;
    else if (!secondCLCT[bx].isValid()) {
      secondCLCT[bx] = *plct;
    }
  }

  for (int bx = 0; bx < CSCConstants::MAX_CLCT_TBINS; bx++) {
    if (bestCLCT[bx].isValid()) {
      bestCLCT[bx].setTrknmb(1);
      if (infoV > 0) LogDebug("CSCCathodeLCTProcessor")
                       << bestCLCT[bx] << " found in " <<
                       CSCDetId::chamberName(theEndcap, theStation, theRing, theChamber)
                       << " (sector " << theSector << " subsector " << theSubsector
                       << " trig id. " << theTrigChamber << ")" << "\n";
    }
    if (secondCLCT[bx].isValid()) {
      secondCLCT[bx].setTrknmb(2);
      if (infoV > 0) LogDebug("CSCCathodeLCTProcessor")
                       << secondCLCT[bx] << " found in " <<
                       CSCDetId::chamberName(theEndcap, theStation, theRing, theChamber)
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
	  << " comparator digi(s) in layer " << i_layer << " of " <<
    detid.chamberName() << " (trig. sector " << theSector
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

  // 2 possible LCTs per CSC x 7 LCT quantities
  int keystrip_data[CSCConstants::MAX_CLCTS_PER_PROCESSOR][CLCT_NUM_QUANTITIES] = {{0}};
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
      int best_halfstrip[CSCConstants::MAX_CLCTS_PER_PROCESSOR], best_quality[CSCConstants::MAX_CLCTS_PER_PROCESSOR];
      for (int ilct = 0; ilct < CSCConstants::MAX_CLCTS_PER_PROCESSOR; ilct++) {
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
        << " nhits = " << std::setw(3) << nhits[hstrip]
        << " pid = " << std::setw(3) << best_pid[hstrip]
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
        << " nhits = " << std::setw(3) << nhits[hstrip]
        << " pid = " << std::setw(3) << best_pid[hstrip]
	      << " best halfstrip = " << std::setw(3) << best_halfstrip[1]
	      << " best quality = "   << std::setw(3) << best_quality[1];
	  }
	}

	// Pattern finder.
	bool ptn_trig = false;
	for (int ilct = 0; ilct < CSCConstants::MAX_CLCTS_PER_PROCESSOR; ilct++) {
	  int best_hs = best_halfstrip[ilct];
	  if (best_hs >= 0 && nhits[best_hs] >= nplanes_hit_pattern) {
	    ptn_trig = true;
	    keystrip_data[ilct][CLCT_PATTERN]    = best_pid[best_hs];
	    keystrip_data[ilct][CLCT_BEND]       =
	      pattern2007[best_pid[best_hs]][CSCConstants::MAX_HALFSTRIPS_IN_PATTERN];
	    // Remove stagger if any.
	    keystrip_data[ilct][CLCT_STRIP]      =
	      best_hs - stagger[CSCConstants::KEY_CLCT_LAYER-1];
	    keystrip_data[ilct][CLCT_BX]         = first_bx;
	    keystrip_data[ilct][CLCT_STRIP_TYPE] = 1;           // obsolete
	    keystrip_data[ilct][CLCT_QUALITY]    = nhits[best_hs];
	    keystrip_data[ilct][CLCT_CFEB]       =
	      keystrip_data[ilct][CLCT_STRIP]/CSCConstants::NUM_HALF_STRIPS_PER_CFEB;
	    int halfstrip_in_cfeb = keystrip_data[ilct][CLCT_STRIP] -
	      CSCConstants::NUM_HALF_STRIPS_PER_CFEB*keystrip_data[ilct][CLCT_CFEB];

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
      if (!time[i_layer][i_strip].empty()) {
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

  int nPreTriggers = 0;

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
        ispretrig[hstrip] = false;
        if (nhits[hstrip]    >= nplanes_hit_pretrig &&
            best_pid[hstrip] >= pid_thresh_pretrig) {
          pre_trig = true;
          ispretrig[hstrip] = true;

          // write each pre-trigger to output
          nPreTriggers++;
          const int bend = pattern2007[best_pid[hstrip]][CSCConstants::MAX_HALFSTRIPS_IN_PATTERN];
          thePreTriggerDigis.push_back(CSCCLCTPreTriggerDigi(1, nhits[hstrip], best_pid[hstrip],
                                                          1, bend, hstrip%32, hstrip/32, bx_time, nPreTriggers, 0));

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
      for (int strip_num = 0; strip_num < CSCConstants::MAX_HALFSTRIPS_IN_PATTERN; strip_num++)
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

#if defined(EDM_ML_DEBUG)
          //LogTrace only ever prints if EDM_ML_DEBUG is defined
          if (infoV > 1) {
            auto lt = LogTrace("CSCCathodeLCTProcessor")
              <<"bx="<<bx_time<<" bx_cor="<< first_bx_corrected[key_hstrip]<<"  bxset=";
            for (im = mset_for_median.begin(); im != mset_for_median.end(); im++) {
              lt<<" "<<*im;
            }
          }
#endif
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
  //  nspan = pspan = pattern2007[best_patid][CSCConstants::MAX_HALFSTRIPS_IN_PATTERN+1]-1;

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

  // keeps dead-time zones around key halfstrips of triggered CLCTs
  bool busyMap[CSCConstants::NUM_HALF_STRIPS_7CFEBS][CSCConstants::MAX_CLCT_TBINS];
  for (int i = 0; i < CSCConstants::NUM_HALF_STRIPS_7CFEBS; i++)
    for (int j = 0; j < CSCConstants::MAX_CLCT_TBINS; j++)
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
      int keystrip_data[CSCConstants::MAX_CLCTS_PER_PROCESSOR][CLCT_NUM_QUANTITIES] = {{0}};

      // Quality for sorting.
      int quality[CSCConstants::NUM_HALF_STRIPS_7CFEBS];
      int best_halfstrip[CSCConstants::MAX_CLCTS_PER_PROCESSOR], best_quality[CSCConstants::MAX_CLCTS_PER_PROCESSOR];
      for (int ilct = 0; ilct < CSCConstants::MAX_CLCTS_PER_PROCESSOR; ilct++)
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
          pretrig_zone[hstrip] = false;
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
              pretrig_zone[hs] = true;
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
        for (int ilct = 0; ilct < CSCConstants::MAX_CLCTS_PER_PROCESSOR; ilct++)
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
            keystrip_data[ilct][CLCT_BEND] = pattern2007[best_pid[best_hs]][CSCConstants::MAX_HALFSTRIPS_IN_PATTERN];
            // Remove stagger if any.
            keystrip_data[ilct][CLCT_STRIP] = best_hs - stagger[CSCConstants::KEY_CLCT_LAYER - 1];
            keystrip_data[ilct][CLCT_BX] = bx;
            keystrip_data[ilct][CLCT_STRIP_TYPE] = 1; // obsolete
            keystrip_data[ilct][CLCT_QUALITY] = nhits[best_hs];
            keystrip_data[ilct][CLCT_CFEB] = keystrip_data[ilct][CLCT_STRIP] / CSCConstants::NUM_HALF_STRIPS_PER_CFEB;
            int halfstrip_in_cfeb = keystrip_data[ilct][CLCT_STRIP] - CSCConstants::NUM_HALF_STRIPS_PER_CFEB * keystrip_data[ilct][CLCT_CFEB];

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
              delta_hs = pattern2007[lctListBX[ilct].getPattern()][CSCConstants::MAX_HALFSTRIPS_IN_PATTERN + 1] - 1;

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
  strm << " pid_thresh_pretrig [lower threshold on pattern id] = "
       << pid_thresh_pretrig << "\n";
  strm << " min_separation     [region of busy key strips] = "
       << min_separation << "\n";
  strm << "++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n";
  LogDebug("CSCCathodeLCTProcessor") << strm.str();
  //std::cerr<<strm.str()<<std::endl;
}

// Reasonably nice dump of digis on half-strips and di-strips.
void CSCCathodeLCTProcessor::dumpDigis(const std::vector<int> strip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS], const int stripType, const int nStrips) const
{
  LogDebug("CSCCathodeLCTProcessor")
    << CSCDetId::chamberName(theEndcap, theStation, theRing, theChamber)
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

    if (late_tbins > CSCConstants::MAX_CLCT_TBINS-1) {
      if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorSuspiciousParameters")
	<< "+++ Allowed range of time bins, [0-" << late_tbins
	<< "] exceeds max allowed, " << CSCConstants::MAX_CLCT_TBINS-1 << " +++\n"
	<< "+++ Set late_tbins to max allowed +++\n";
      late_tbins = CSCConstants::MAX_CLCT_TBINS-1;
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
  for (int bx = 0; bx < CSCConstants::MAX_CLCT_TBINS; bx++) {
    if (bestCLCT[bx].isValid())   tmpV.push_back(bestCLCT[bx]);
    if (secondCLCT[bx].isValid()) tmpV.push_back(secondCLCT[bx]);
  }
  return tmpV;
}



void CSCCathodeLCTProcessor::testLCTs() {
  // test to make sure what goes into an LCT is what comes out.
  for (int ptn = 0; ptn < 8; ptn++) {
    for (int bend = 0; bend < 2; bend++) {
      for (int cfeb = 0; cfeb < CSCConstants::MAX_CFEBS; cfeb++) {
	for (int key_strip = 0; key_strip < CSCConstants::NUM_HALF_STRIPS_PER_CFEB; key_strip++) {
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
