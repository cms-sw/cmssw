#ifndef L1Trigger_CSCTriggerPrimitives_CSCCathodeLCTProcessor_h
#define L1Trigger_CSCTriggerPrimitives_CSCCathodeLCTProcessor_h

/** \class CSCCathodeLCTProcessor
 *
 * This class simulates the functionality of the cathode LCT card. It is run by
 * the MotherBoard and returns up to two CathodeLCTs.  It can be run either in
 * a test mode, where it is passed arrays of halfstrip and distrip times,
 * or in normal mode where it determines the time and comparator
 * information from the comparator digis.
 *
 * The CathodeLCTs come in distrip and halfstrip flavors; they are sorted
 * (from best to worst) as follows: 6/6H, 5/6H, 6/6D, 4/6H, 5/6D, 4/6D.
 *
 * \date May 2001  Removed the card boundaries.  Changed the Pretrigger to
 * emulate the hardware electronic logic.  Also changed the keylayer to be
 * the 4th layer in a chamber instead of the 3rd layer from the interaction
 * region. The code is a more realistic simulation of hardware LCT logic now.
 * -Jason Mumford.
 *
 * \author Benn Tannenbaum  UCLA 13 July 1999 benn@physics.ucla.edu
 * Numerous later improvements by Jason Mumford and Slava Valuev (see cvs
 * in ORCA).
 * Porting from ORCA by S. Valuev (Slava.Valuev@cern.ch), May 2006.
 *
 * Updates for high pileup running by Vadim Khotilovich (TAMU), December 2012
 *
 * Updates for integrated local trigger with GEMs by
 * Sven Dildick (TAMU) and Tao Huang (TAMU), April 2015
 *
 * Removing usage of outdated class CSCTriggerGeometry by Sven Dildick (TAMU)
 */

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTPreTriggerDigi.h"
#include "CondFormats/CSCObjects/interface/CSCDBL1TPParameters.h"
#include "L1Trigger/CSCCommonTrigger/interface/CSCConstants.h"

#include <vector>

class CSCGeometry;

class CSCCathodeLCTProcessor
{
 public:
  /** Normal constructor. */
  CSCCathodeLCTProcessor(unsigned endcap, unsigned station, unsigned sector,
			 unsigned subsector, unsigned chamber,
			 const edm::ParameterSet& conf,
			 const edm::ParameterSet& comm,
			 const edm::ParameterSet& ctmb);

  /** Default constructor. Used for testing. */
  CSCCathodeLCTProcessor();

  /** Sets configuration parameters obtained via EventSetup mechanism. */
  void setConfigParameters(const CSCDBL1TPParameters* conf);

  /** Clears the LCT containers. */
  void clear();

  void setCSCGeometry(const CSCGeometry *g) { csc_g = g; }

  /** Runs the LCT processor code. Called in normal running -- gets info from
      a collection of comparator digis. */
  std::vector<CSCCLCTDigi> run(const CSCComparatorDigiCollection* compdc);

  /** Called in test mode and by the run(compdc) function; does the actual LCT
      finding. */
  void run(const std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
	   const std::vector<int> distrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]);

  /** Access routines to comparator digis. */
  bool getDigis(const CSCComparatorDigiCollection* compdc);
  void getDigis(const CSCComparatorDigiCollection* compdc, const CSCDetId& id);

  /** Best LCT in this chamber, as found by the processor. */
  CSCCLCTDigi bestCLCT[CSCConstants::MAX_CLCT_TBINS];

  /** Second best LCT in this chamber, as found by the processor. */
  CSCCLCTDigi secondCLCT[CSCConstants::MAX_CLCT_TBINS];

  /** Returns vector of CLCTs in the read-out time window, if any. */
  std::vector<CSCCLCTDigi> readoutCLCTs();

  /** Returns vector of all found CLCTs, if any. */
  std::vector<CSCCLCTDigi> getCLCTs();

  std::vector<int> preTriggerBXs() const {return thePreTriggerBXs;}

  std::vector<CSCCLCTPreTriggerDigi> preTriggerDigis() const {return thePreTriggerDigis; }

  static void distripStagger(int stag_triad[CSCConstants::MAX_NUM_STRIPS_7CFEBS],
			     int stag_time[CSCConstants::MAX_NUM_STRIPS_7CFEBS],
			     int stag_digi[CSCConstants::MAX_NUM_STRIPS_7CFEBS],
			     int i_distrip, bool debug = false);

  /** Set ring number
   * Has to be done for upgrade ME1a!
   **/
  void setRing(unsigned r) {theRing = r;}

  /** Pre-defined patterns. */
  static const int pre_hit_pattern[2][CSCConstants::MAX_STRIPS_IN_PATTERN];
  static const int pattern[CSCConstants::NUM_CLCT_PATTERNS_PRE_TMB07][CSCConstants::MAX_STRIPS_IN_PATTERN+1];

  static const int pattern2007_offset[CSCConstants::MAX_HALFSTRIPS_IN_PATTERN];
  static const int pattern2007[CSCConstants::NUM_CLCT_PATTERNS][CSCConstants::MAX_HALFSTRIPS_IN_PATTERN+2];

  // we use these next ones to address the various bits inside the array that's
  // used to make the cathode LCTs.
  enum CLCT_INDICES {CLCT_PATTERN,
                     CLCT_BEND,
                     CLCT_STRIP,
                     CLCT_BX,
                     CLCT_STRIP_TYPE,
                     CLCT_QUALITY,
                     CLCT_CFEB,
                     CLCT_NUM_QUANTITIES= 7};

 private:
  /** Verbosity level: 0: no print (default).
   *                   1: print only CLCTs found.
   *                   2: info at every step of the algorithm.
   *                   3: add special-purpose prints. */
  int infoV;

  /** Chamber id (trigger-type labels). */
  const unsigned theEndcap;
  const unsigned theStation;
  const unsigned theSector;
  const unsigned theSubsector;
  const unsigned theTrigChamber;

  const CSCGeometry* csc_g;

  // holders for easy access:
  unsigned int theRing;
  unsigned int theChamber;
  bool isME11;

  int numStrips;
  int stagger[CSCConstants::NUM_LAYERS];

  std::vector<CSCComparatorDigi> digiV[CSCConstants::NUM_LAYERS];
  std::vector<int> thePreTriggerBXs;
  std::vector<CSCCLCTPreTriggerDigi> thePreTriggerDigis;

  /** Flag for "real" - not idealized - version of the algorithm. */
  bool isMTCC;

  /** Flag for 2007 firmware version. */
  bool isTMB07;

  /** Flag for SLHC studies. */
  bool isSLHC;

  /** Offset between ALCT and CLCT in simulation.
   *  This is important when ALCTs (at BX0=8) and CLCTs (at BX0=7)
   *  are correlated in the trigger motherboard. */
  unsigned int alctClctOffset;

  /** Configuration parameters. */
  unsigned int fifo_tbins,  fifo_pretrig; // only for test beam mode.
  unsigned int hit_persist, drift_delay;
  unsigned int nplanes_hit_pretrig, nplanes_hit_pattern;
  unsigned int pid_thresh_pretrig,  min_separation;
  unsigned int tmb_l1a_window_size;

  /** VK: some quick and dirty fix to reduce CLCT deadtime */
  int start_bx_shift;

  /** VK: special configuration parameters for ME1a treatment */
  bool smartME1aME1b, disableME1a, gangedME1a;

  /** VK: separate handle for early time bins */
  int early_tbins;

  /** VK: use of localized dead-time zones */
  bool use_dead_time_zoning;
  unsigned int clct_state_machine_zone; // +- around a keystrip
  bool dynamic_state_machine_zone;  //use a pattern dependent zone

  /** VK: allow triggers only in +-pretrig_trig_zone around pretriggers */
  unsigned int pretrig_trig_zone;

  /** VK: whether to use corrected_bx instead of pretrigger BX */
  bool use_corrected_bx;

  /** VK: whether to readout only the earliest two LCTs in readout window */
  bool readout_earliest_2;

  /** Default values of configuration parameters. */
  static const unsigned int def_fifo_tbins,  def_fifo_pretrig;
  static const unsigned int def_hit_persist, def_drift_delay;
  static const unsigned int def_nplanes_hit_pretrig;
  static const unsigned int def_nplanes_hit_pattern;
  static const unsigned int def_pid_thresh_pretrig, def_min_separation;
  static const unsigned int def_tmb_l1a_window_size;

  /** Set default values for configuration parameters. */
  void setDefaultConfigParameters();

  /** Make sure that the parameter values are within the allowed range. */
  void checkConfigParameters();

  /** Number of di-strips/half-strips per CFEB. */
  static const int cfeb_strips[2];

  //---------------- Methods common to all firmware versions ------------------
  void readComparatorDigis(std::vector<int>halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
			   std::vector<int> distrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]);
  void readComparatorDigis(std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]);
  void pulseExtension(
 const std::vector<int> time[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
 const int nStrips,
 unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]);

  //--------------- Functions for 2007 version of the firmware ----------------
  std::vector<CSCCLCTDigi> findLCTs(
 const std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]);
  bool preTrigger(
      const unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
      const int start_bx, int& first_bx);
  bool ptnFinding(
      const unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
      const int nStrips, const unsigned int bx_time);
  void markBusyKeys(const int best_hstrip, const int best_patid,
		    int quality[CSCConstants::NUM_HALF_STRIPS_7CFEBS]);

  unsigned int best_pid[CSCConstants::NUM_HALF_STRIPS_7CFEBS];
  unsigned int nhits[CSCConstants::NUM_HALF_STRIPS_7CFEBS];
  int first_bx_corrected[CSCConstants::NUM_HALF_STRIPS_7CFEBS];

  //--------------- Functions for SLHC studies ----------------

  std::vector<CSCCLCTDigi> findLCTsSLHC(
    const std::vector<int>  halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]);

  bool ispretrig[CSCConstants::NUM_HALF_STRIPS_7CFEBS];

  //--------------------------- Auxiliary methods -----------------------------
  /** Dump CLCT configuration parameters. */
  void dumpConfigParams() const;

  /** Dump digis on half-strips and di-strips. */
  void dumpDigis(
      const std::vector<int> strip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
      const int stripType, const int nStrips) const;

  //--------------------------- Methods for tests -----------------------------
  void testDistripStagger();
  void testLCTs();
  int findNumLayersHit(std::vector<int> stripsHit[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]);
};

#endif
