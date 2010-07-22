#ifndef CSCTriggerPrimitives_CSCAnodeLCTProcessor_h
#define CSCTriggerPrimitives_CSCAnodeLCTProcessor_h

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
 * $Date: 2009/03/27 17:04:52 $
 * $Revision: 1.18 $
 *
 */

#include <vector>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCALCTDigi.h>
#include <CondFormats/CSCObjects/interface/CSCL1TPParameters.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>

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
  void setConfigParameters(const CSCL1TPParameters* conf);

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

  /** Pre-defined patterns. */
  enum {NUM_PATTERN_WIRES = 14};
  static const int pattern_envelope[CSCConstants::NUM_ALCT_PATTERNS][NUM_PATTERN_WIRES];
  static const int pattern_mask_slim[CSCConstants::NUM_ALCT_PATTERNS][NUM_PATTERN_WIRES];
  static const int pattern_mask_open[CSCConstants::NUM_ALCT_PATTERNS][NUM_PATTERN_WIRES];

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

  int numWireGroups;
  int MESelection;

  int first_bx[CSCConstants::MAX_NUM_WIRES];
  int quality[CSCConstants::MAX_NUM_WIRES][3];
  std::vector<CSCWireDigi> digiV[CSCConstants::NUM_LAYERS];
  unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES];

  /** Flag for MTCC data (i.e., "open" patterns). */
  bool isMTCC;

  /** Use TMB07 flag for DAQ-2006 version (implemented in late 2007). */
  bool isTMB07;

  /** Configuration parameters. */
  unsigned int fifo_tbins, fifo_pretrig, drift_delay;
  unsigned int nplanes_hit_pretrig, nplanes_hit_accel_pretrig;
  unsigned int nplanes_hit_pattern, nplanes_hit_accel_pattern;
  unsigned int trig_mode, accel_mode, l1a_window_width;

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
