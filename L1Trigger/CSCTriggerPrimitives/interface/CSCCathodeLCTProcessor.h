#ifndef L1Trigger_CSCTriggerPrimitives_CSCCathodeLCTProcessor_h
#define L1Trigger_CSCTriggerPrimitives_CSCCathodeLCTProcessor_h

/** \class CSCCathodeLCTProcessor
 *
 * This class simulates the functionality of the cathode LCT card. It is run by
 * the MotherBoard and returns up to two CathodeLCTs.  It can be run either in
 * a test mode, where it is passed arrays of halfstrip times,
 * or in normal mode where it determines the time and comparator
 * information from the comparator digis.
 *
 * The CathodeLCTs only come halfstrip flavors
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

#include "DataFormats/CSCDigi/interface/CSCComparatorDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCCLCTPreTriggerDigi.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCBaseboard.h"

#include <vector>

class CSCCathodeLCTProcessor : public CSCBaseboard {
public:
  /** Normal constructor. */
  CSCCathodeLCTProcessor(unsigned endcap,
                         unsigned station,
                         unsigned sector,
                         unsigned subsector,
                         unsigned chamber,
                         const edm::ParameterSet& conf);

  /** Default constructor. Used for testing. */
  CSCCathodeLCTProcessor();

  /** Sets configuration parameters obtained via EventSetup mechanism. */
  void setConfigParameters(const CSCDBL1TPParameters* conf);

  /** Clears the LCT containers. */
  void clear();

  /** Runs the LCT processor code. Called in normal running -- gets info from
      a collection of comparator digis. */
  std::vector<CSCCLCTDigi> run(const CSCComparatorDigiCollection* compdc);

  /** Called in test mode and by the run(compdc) function; does the actual LCT
      finding. */
  void run(const std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]);

  /** Returns vector of CLCTs in the read-out time window, if any. */
  std::vector<CSCCLCTDigi> readoutCLCTs() const;
  std::vector<CSCCLCTDigi> readoutCLCTsME1a() const;
  std::vector<CSCCLCTDigi> readoutCLCTsME1b() const;

  /** Returns vector of all found CLCTs, if any. */
  std::vector<CSCCLCTDigi> getCLCTs() const;

  /** get best/second best CLCT
   * Note: CLCT has BX shifted */
  CSCCLCTDigi getBestCLCT(int bx) const;
  CSCCLCTDigi getSecondCLCT(int bx) const;

  std::vector<int> preTriggerBXs() const { return thePreTriggerBXs; }

  /** read out CLCTs in ME1a , ME1b */
  std::vector<CSCCLCTPreTriggerDigi> preTriggerDigis() const { return thePreTriggerDigis; }
  std::vector<CSCCLCTPreTriggerDigi> preTriggerDigisME1a() const;
  std::vector<CSCCLCTPreTriggerDigi> preTriggerDigisME1b() const;

protected:
  /** Best LCT in this chamber, as found by the processor. */
  CSCCLCTDigi bestCLCT[CSCConstants::MAX_CLCT_TBINS];

  /** Second best LCT in this chamber, as found by the processor. */
  CSCCLCTDigi secondCLCT[CSCConstants::MAX_CLCT_TBINS];

  /** Access routines to comparator digis. */
  bool getDigis(const CSCComparatorDigiCollection* compdc);
  void getDigis(const CSCComparatorDigiCollection* compdc, const CSCDetId& id);

  /** Set default values for configuration parameters. */
  void setDefaultConfigParameters();

  /** Make sure that the parameter values are within the allowed range. */
  void checkConfigParameters();

  //---------------- Methods common to all firmware versions ------------------
  // Single-argument version for TMB07 (halfstrip-only) firmware.
  // Takes the comparator & time info and stuffs it into halfstrip vector.
  // Multiple hits on the same strip are allowed.
  void readComparatorDigis(std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]);
  void pulseExtension(const std::vector<int> time[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
                      const int nStrips,
                      unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]);

  //--------------- Functions for post-2007 version of the firmware -----------
  virtual std::vector<CSCCLCTDigi> findLCTs(
      const std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]);

  /* Check all half-strip pattern envelopes simultaneously, on every clock cycle, for a matching pattern */
  virtual bool preTrigger(const unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
                          const int start_bx,
                          int& first_bx);

  /* For a given clock cycle, check each half-strip if a pattern matches */
  bool patternFinding(const unsigned int pulse[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
                      const int nStrips,
                      const unsigned int bx_time);

  /* Mark the half-strips around the best half-strip as busy */
  void markBusyKeys(const int best_hstrip, const int best_patid, int quality[CSCConstants::NUM_HALF_STRIPS_7CFEBS]);

  //--------------------------- Auxiliary methods -----------------------------
  /** Dump CLCT configuration parameters. */
  void dumpConfigParams() const;

  /** Dump half-strip digis */
  void dumpDigis(const std::vector<int> strip[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
                 const int nStrips) const;

  //--------------------------- Member variables -----------------------------

  /* best pattern Id for a given half-strip */
  unsigned int best_pid[CSCConstants::NUM_HALF_STRIPS_7CFEBS];

  /* number of layers hit on a given half-strip */
  unsigned int nhits[CSCConstants::NUM_HALF_STRIPS_7CFEBS];

  int first_bx_corrected[CSCConstants::NUM_HALF_STRIPS_7CFEBS];

  /* does a given half-strip have a pre-trigger? */
  bool ispretrig[CSCConstants::NUM_HALF_STRIPS_7CFEBS];

  // we use these next ones to address the various bits inside the array that's
  // used to make the cathode LCTs.
  enum CLCT_INDICES {
    CLCT_PATTERN,
    CLCT_BEND,
    CLCT_STRIP,
    CLCT_BX,
    CLCT_STRIP_TYPE,
    CLCT_QUALITY,
    CLCT_CFEB,
    CLCT_NUM_QUANTITIES = 7
  };

  /* number of strips used in this processor */
  int numStrips;

  /* Is the layer in the chamber staggered? */
  int stagger[CSCConstants::NUM_LAYERS];

  std::vector<CSCComparatorDigi> digiV[CSCConstants::NUM_LAYERS];
  std::vector<int> thePreTriggerBXs;
  std::vector<CSCCLCTPreTriggerDigi> thePreTriggerDigis;

  /** Configuration parameters. */
  unsigned int fifo_tbins, fifo_pretrig;  // only for test beam mode.
  unsigned int hit_persist, drift_delay;
  unsigned int nplanes_hit_pretrig, nplanes_hit_pattern;
  unsigned int pid_thresh_pretrig, min_separation;
  unsigned int tmb_l1a_window_size;

  /** VK: some quick and dirty fix to reduce CLCT deadtime */
  int start_bx_shift;

  /** VK: separate handle for early time bins */
  int early_tbins;

  /** VK: whether to readout only the earliest two LCTs in readout window */
  bool readout_earliest_2;

  /** Default values of configuration parameters. */
  static const unsigned int def_fifo_tbins, def_fifo_pretrig;
  static const unsigned int def_hit_persist, def_drift_delay;
  static const unsigned int def_nplanes_hit_pretrig;
  static const unsigned int def_nplanes_hit_pattern;
  static const unsigned int def_pid_thresh_pretrig, def_min_separation;
  static const unsigned int def_tmb_l1a_window_size;
};

#endif
