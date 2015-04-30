#ifndef L1Trigger_CSCTriggerPrimitives_CSCAnodeLCTProcessor_h
#define L1Trigger_CSCTriggerPrimitives_CSCAnodeLCTProcessor_h

/** \class CSCAnodeLCTProcessor
 *
 * This class simulates the functionality of the anode LCT card. It is run by
 * the MotherBoard and returns up to two AnodeLCTs.  It can be run either in a
 * test mode, where it is passed an array of wire times, or in normal mode
 * where it determines the wire times from the wire digis.
 *
 * \author Benn Tannenbaum  benn@physics.ucla.edu 13 July 1999
 * Numerous later improvements by Jason Mumford and Slava Valuev (see cvs
 * in ORCA).
 * Porting from ORCA by S. Valuev (Slava.Valuev@cern.ch), May 2006.
 *
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "CondFormats/CSCObjects/interface/CSCDBL1TPParameters.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"

#include <vector>

class CSCAnodeLCTProcessor
{
 public:
  /** Normal constructor. */
  CSCAnodeLCTProcessor(unsigned endcap, unsigned station, unsigned sector,
		       unsigned subsector, unsigned chamber,
		       const edm::ParameterSet& conf,
		       const edm::ParameterSet& comm);

  /** Default constructor. Used for testing. */
  CSCAnodeLCTProcessor();

  /** Sets configuration parameters obtained via EventSetup mechanism. */
  void setConfigParameters(const CSCDBL1TPParameters* conf);

  /** Clears the LCT containers. */
  void clear();

  /** Runs the LCT processor code. Called in normal running -- gets info from
      a collection of wire digis. */
  std::vector<CSCALCTDigi> run(const CSCWireDigiCollection* wiredc);

  /** Runs the LCT processor code. Called in normal running or in testing
      mode. */
  void run(const std::vector<int> wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES]);

  /** Access routines to wire digis. */
  bool getDigis(const CSCWireDigiCollection* wiredc);
  void getDigis(const CSCWireDigiCollection* wiredc, const CSCDetId& id);

  /** Maximum number of time bins reported in the ALCT readout. */
  enum {MAX_ALCT_BINS = 16};

  /** Best LCTs in this chamber, as found by the processor.
      In old ALCT algorithms, up to two best ALCT per Level-1 accept window
      had been reported.
      In the ALCT-2006 algorithms, up to two best ALCTs PER EVERY TIME BIN in
      Level-1 accept window are reported. */
  CSCALCTDigi bestALCT[MAX_ALCT_BINS];

  /** Second best LCTs in this chamber, as found by the processor. */
  CSCALCTDigi secondALCT[MAX_ALCT_BINS];

  /** Returns vector of ALCTs in the read-out time window, if any. */
  std::vector<CSCALCTDigi> readoutALCTs();

  /** Returns vector of all found ALCTs, if any. */
  std::vector<CSCALCTDigi> getALCTs();

  /** set ring number. Important only for ME1a */
  void setRing(unsigned r) {theRing = r;}

  /** Pre-defined patterns. */
  enum {NUM_PATTERN_WIRES = 14};
  static const int pattern_envelope[CSCConstants::NUM_ALCT_PATTERNS][NUM_PATTERN_WIRES];
  static const int pattern_mask_slim[CSCConstants::NUM_ALCT_PATTERNS][NUM_PATTERN_WIRES];
  static const int pattern_mask_open[CSCConstants::NUM_ALCT_PATTERNS][NUM_PATTERN_WIRES];
  static const int pattern_mask_r1[CSCConstants::NUM_ALCT_PATTERNS][NUM_PATTERN_WIRES];
  static const int time_weights[NUM_PATTERN_WIRES];

 private:
  /** Verbosity level: 0: no print (default).
   *                   1: print only ALCTs found.
   *                   2: info at every step of the algorithm.
   *                   3: add special-purpose prints. */
  int infoV;

  /** Chamber id (trigger-type labels). */
  const unsigned theEndcap;
  const unsigned theStation;
  const unsigned theSector;
  const unsigned theSubsector;
  const unsigned theTrigChamber;

  /** ring number. Only matters for ME1a */
  unsigned theRing;

  unsigned theChamber;

  bool isME11;

  int numWireGroups;
  int MESelection;

  int first_bx[CSCConstants::MAX_NUM_WIRES];
  int first_bx_corrected[CSCConstants::MAX_NUM_WIRES];
  int quality[CSCConstants::MAX_NUM_WIRES][3];
  std::vector<CSCWireDigi> digiV[CSCConstants::NUM_LAYERS];
  unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES];

  /** Flag for MTCC data (i.e., "open" patterns). */
  bool isMTCC;

  /** Use TMB07 flag for DAQ-2006 version (implemented in late 2007). */
  bool isTMB07;

  /** Flag for SLHC studies. */
  bool isSLHC;

  /** Configuration parameters. */
  unsigned int fifo_tbins, fifo_pretrig, drift_delay;
  unsigned int nplanes_hit_pretrig, nplanes_hit_accel_pretrig;
  unsigned int nplanes_hit_pattern, nplanes_hit_accel_pattern;
  unsigned int trig_mode, accel_mode, l1a_window_width;

  /** SLHC: hit persistency length */
  unsigned int hit_persist;

  /** SLHC: special configuration parameters for ME1a treatment */
  bool disableME1a;

  /** SLHC: separate handle for early time bins */
  int early_tbins;

  /** SLHC: delta BX time depth for ghostCancellationLogic */
  int ghost_cancellation_bx_depth;

  /** SLHC: whether to consider ALCT candidates' qualities 
      while doing ghostCancellationLogic on +-1 wire groups */
  bool ghost_cancellation_side_quality;

  /** SLHC: deadtime clocks after pretrigger (extra in addition to drift_delay) */
  unsigned int pretrig_extra_deadtime;

  /** SLHC: whether to use corrected_bx instead of pretrigger BX */
  bool use_corrected_bx;

  /** SLHC: whether to use narrow pattern mask for the rings close to the beam */
  bool narrow_mask_r1;

  /** SLHC: run the ALCT processor for the Phase-II ME2/1 integrated local trigger */
  bool runME21ILT_;

  /** SLHC: run the ALCT processor for the Phase-II ME3/1(ME4/1) integrated local trigger */
  bool runME3141ILT_;

  /** Default values of configuration parameters. */
  static const unsigned int def_fifo_tbins, def_fifo_pretrig;
  static const unsigned int def_drift_delay;
  static const unsigned int def_nplanes_hit_pretrig, def_nplanes_hit_pattern;
  static const unsigned int def_nplanes_hit_accel_pretrig;
  static const unsigned int def_nplanes_hit_accel_pattern;
  static const unsigned int def_trig_mode, def_accel_mode;
  static const unsigned int def_l1a_window_width;

  /** Chosen pattern mask. */
  int pattern_mask[CSCConstants::NUM_ALCT_PATTERNS][NUM_PATTERN_WIRES];

  /** Load pattern mask defined by configuration into pattern_mask */
  void loadPatternMask();

  /** Set default values for configuration parameters. */
  void setDefaultConfigParameters();

  /** Make sure that the parameter values are within the allowed range. */
  void checkConfigParameters();

  /** Clears the quality for a given wire and pattern if it is a ghost. */
  void clear(const int wire, const int pattern);

  /** ALCT algorithm methods. */
  void readWireDigis(std::vector<int> wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES]);
  bool pulseExtension(const std::vector<int> wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES]);
  bool preTrigger(const int key_wire, const int start_bx);
  bool patternDetection(const int key_wire);
  void ghostCancellationLogic();
  void ghostCancellationLogicSLHC();
  void lctSearch();
  void trigMode(const int key_wire);
  void accelMode(const int key_wire);

  std::vector<CSCALCTDigi>
    bestTrackSelector(const std::vector<CSCALCTDigi>& all_alcts);
  bool isBetterALCT(const CSCALCTDigi& lhsALCT, const CSCALCTDigi& rhsALCT);

  /** Dump ALCT configuration parameters. */
  void dumpConfigParams() const;

  /** Dump digis on wire groups. */
  void dumpDigis(const std::vector<int> wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES]) const;

  void showPatterns(const int key_wire);
};

#endif
