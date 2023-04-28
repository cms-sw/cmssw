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
#include "DataFormats/CSCDigi/interface/CSCShowerDigi.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCBaseboard.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/LCTQualityControl.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/ComparatorCodeLUT.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/PulseArray.h"

#include <vector>

class CSCCathodeLCTProcessor : public CSCBaseboard {
public:
  /** Normal constructor. */
  CSCCathodeLCTProcessor(unsigned endcap,
                         unsigned station,
                         unsigned sector,
                         unsigned subsector,
                         unsigned chamber,
                         CSCBaseboard::Parameters& conf);

  /** Default destructor. */
  ~CSCCathodeLCTProcessor() override = default;

  /** Sets configuration parameters obtained via EventSetup mechanism. */
  void setConfigParameters(const CSCDBL1TPParameters* conf);

  void setESLookupTables(const CSCL1TPLookupTableCCLUT* conf);

  /** Clears the LCT containers. */
  void clear();

  /** Runs the LCT processor code. Called in normal running -- gets info from
      a collection of comparator digis. */
  std::vector<CSCCLCTDigi> run(const CSCComparatorDigiCollection* compdc);

  /** Called in test mode and by the run(compdc) function; does the actual LCT
      finding. */
  void run(const std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER]);

  /** Returns vector of CLCTs in the read-out time window, if any. */
  std::vector<CSCCLCTDigi> readoutCLCTs() const;

  /** Returns vector of all found CLCTs, if any. */
  std::vector<CSCCLCTDigi> getCLCTs() const;

  /** get best/second best CLCT
   * Note: CLCT has BX shifted */
  CSCCLCTDigi getBestCLCT(int bx) const;
  CSCCLCTDigi getSecondCLCT(int bx) const;

  /* get the flag of local shower around best CLCT at bx */
  bool getLocalShowerFlag(int bx) const;

  std::vector<int> preTriggerBXs() const { return thePreTriggerBXs; }

  /** read out CLCTs in ME1a , ME1b */
  std::vector<CSCCLCTPreTriggerDigi> preTriggerDigis() const { return thePreTriggerDigis; }

  /* get special bits for high multiplicity triggers */
  //unsigned getInTimeHMT() const { return inTimeHMT_; }
  //unsigned getOutTimeHMT() const { return outTimeHMT_; }
  /* get array of high multiplicity triggers */
  std::vector<CSCShowerDigi> getAllShower() const;

  /** Returns shower bits */
  std::vector<CSCShowerDigi> readoutShower() const;

protected:
  /** Best LCT in this chamber, as found by the processor. */
  CSCCLCTDigi bestCLCT[CSCConstants::MAX_CLCT_TBINS];

  /** Second best LCT in this chamber, as found by the processor. */
  CSCCLCTDigi secondCLCT[CSCConstants::MAX_CLCT_TBINS];

  CSCShowerDigi cathode_showers_[CSCConstants::MAX_CLCT_TBINS];
  //CSCShowerDigi shower_;

  /* flag of shower around best CLCT in each BX */
  bool localShowerFlag[CSCConstants::MAX_CLCT_TBINS];

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
  void readComparatorDigis(
      std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER]);
  void pulseExtension(
      const std::vector<int> time[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER]);

  //--------------- Functions for post-2007 version of the firmware -----------
  virtual std::vector<CSCCLCTDigi> findLCTs(
      const std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER]);

  /* Check all half-strip pattern envelopes simultaneously, on every clock cycle, for a matching pattern
     Returns true if a pretrigger was found, and the first BX of the pretrigger */
  virtual bool preTrigger(const int start_bx, int& first_bx);

  /* For a given clock cycle, check each half-strip if a pattern matches
     This function determines best_pid_ and nhits_ for each half-strip */
  bool patternFinding(const unsigned int bx_time,
                      std::map<int, std::map<int, CSCCLCTDigi::ComparatorContainer> >& hits_in_patterns);

  void cleanComparatorContainer(CSCCLCTDigi& lct) const;

  /* Mark the half-strips around the best half-strip as busy */
  void markBusyKeys(const int best_hstrip,
                    const int best_patid,
                    int quality[CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER]);

  // build a new CLCT trigger
  CSCCLCTDigi constructCLCT(const int bx,
                            const unsigned halfstrip_withstagger,
                            const CSCCLCTDigi::ComparatorContainer& hits);

  // build a new CLCT pretrigger
  CSCCLCTPreTriggerDigi constructPreCLCT(const int bx, const unsigned halfstrip, const unsigned index) const;

  // resets ispretrig_
  void clearPreTriggers();
  //--------------------------- Auxiliary methods -----------------------------
  /** Dump CLCT configuration parameters. */
  void dumpConfigParams() const;

  /** Dump half-strip digis */
  void dumpDigis(
      const std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER]) const;

  /* check whether there is a shower around best CLCT */
  void checkLocalShower(
      int zone,
      const std::vector<int> halfstrip[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER]);

  void encodeHighMultiplicityBits();

  //--------------------------- Member variables -----------------------------

  PulseArray pulse_;

  /* best pattern Id for a given half-strip */
  unsigned int best_pid[CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER];

  /* number of layers hit on a given half-strip */
  unsigned int nhits[CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER];

  /* does a given half-strip have a pre-trigger? */
  bool ispretrig_[CSCConstants::MAX_NUM_HALF_STRIPS_RUN2_TRIGGER];

  // actual LUT used
  CSCPatternBank::LCTPatterns clct_pattern_ = {};

  /* number of strips used in this processor */
  int numStrips_;
  int numCFEBs_;
  int numHalfStrips_;

  /* Is the layer in the chamber staggered? */
  int stagger[CSCConstants::NUM_LAYERS];

  std::vector<CSCComparatorDigi> digiV[CSCConstants::NUM_LAYERS];
  std::vector<int> thePreTriggerBXs;
  std::vector<CSCCLCTPreTriggerDigi> thePreTriggerDigis;

  /* data members for high multiplicity triggers */
  std::vector<unsigned> thresholds_;
  unsigned showerNumTBins_;
  unsigned minLayersCentralTBin_;
  unsigned minbx_readout_;
  unsigned maxbx_readout_;
  /** check the peak of total hits and single bx hits for cathode HMT */
  bool peakCheck_;

  /** Configuration parameters. */
  unsigned int fifo_tbins, fifo_pretrig;  // only for test beam mode.
  unsigned int hit_persist, drift_delay;
  unsigned int nplanes_hit_pretrig, nplanes_hit_pattern;
  unsigned int pid_thresh_pretrig, min_separation;
  unsigned int tmb_l1a_window_size;

  /** VK: some quick and dirty fix to reduce CLCT deadtime */
  int start_bx_shift;

  /* define the region around best CLCT to check local shower */
  int localShowerZone;

  /* threshold of total hits for local shower */
  int localShowerThresh;

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

  /* quality control */
  std::unique_ptr<LCTQualityControl> qualityControl_;

  /* comparator-code lookup table algorithm */
  std::unique_ptr<ComparatorCodeLUT> cclut_;
};

#endif
