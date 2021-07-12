#ifndef L1Trigger_CSCTriggerPrimitives_CSCAnodeLCTProcessor_h
#define L1Trigger_CSCTriggerPrimitives_CSCAnodeLCTProcessor_h

/** \class CSCAnodeLCTProcessor
 *
 * This class simulates the functionality of the anode LCT card. It is run by
 * the MotherBoard and returns up to two AnodeLCTs.  It can be run either in a
 * test mode, where it is passed an array of wire times, or in normal mode
 * where it determines the wire times from the wire digis.
 *
 *    This is the simulation for the Anode LCT Processor for the Level-1
 *    Trigger.  This processor consists of several stages:
 *
 *      1. Pulse extension of signals coming from wires.
 *      2. Pretrigger for each key-wire.
 *      3. Pattern detector if a pretrigger is found for the given key-wire.
 *      4. Ghost Cancellation Logic (GCL).
 *      5. Best track search and promotion.
 *      6. Second best track search and promotion.
 *
 *    The inputs to the ALCT Processor are wire digis.
 *     The output is up to two ALCT digi words.
 *
 * \author Benn Tannenbaum  benn@physics.ucla.edu 13 July 1999
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

#include "DataFormats/CSCDigi/interface/CSCWireDigiCollection.h"
#include "DataFormats/CSCDigi/interface/CSCALCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCALCTPreTriggerDigi.h"
#include "DataFormats/CSCDigi/interface/CSCShowerDigi.h"
#include "CondFormats/CSCObjects/interface/CSCDBL1TPParameters.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCBaseboard.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/LCTQualityControl.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/PulseArray.h"

#include <vector>

class CSCAnodeLCTProcessor : public CSCBaseboard {
public:
  /** Normal constructor. */
  CSCAnodeLCTProcessor(unsigned endcap,
                       unsigned station,
                       unsigned sector,
                       unsigned subsector,
                       unsigned chamber,
                       const edm::ParameterSet& conf);

  /** Default destructor. */
  ~CSCAnodeLCTProcessor() override = default;

  /** Sets configuration parameters obtained via EventSetup mechanism. */
  void setConfigParameters(const CSCDBL1TPParameters* conf);

  /** Clears the LCT containers. */
  void clear();

  // This is the main routine for normal running.  It gets wire times
  // from the wire digis and then passes them on to another run() function.
  std::vector<CSCALCTDigi> run(const CSCWireDigiCollection* wiredc);

  // This version of the run() function can either be called in a standalone
  // test, being passed the time array, or called by the run() function above.
  // It gets wire times from an input array and then loops over the keywires.
  // All found LCT candidates are sorted and the best two are retained.
  void run(const std::vector<int> wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIREGROUPS]);

  /** Returns vector of ALCTs in the read-out time window, if any. */
  std::vector<CSCALCTDigi> readoutALCTs() const;

  /** Returns vector of all found ALCTs, if any. */
  std::vector<CSCALCTDigi> getALCTs() const;

  /** read out pre-ALCTs */
  std::vector<CSCALCTPreTriggerDigi> preTriggerDigis() const { return thePreTriggerDigis; }

  /** Return best/second best ALCTs */
  CSCALCTDigi getBestALCT(int bx) const;
  CSCALCTDigi getSecondALCT(int bx) const;

  /* get special bits for high multiplicity triggers */
  unsigned getInTimeHMT() const { return inTimeHMT_; }
  unsigned getOutTimeHMT() const { return outTimeHMT_; }

  /** Returns shower bits */
  CSCShowerDigi readoutShower() const;

protected:
  /** Best LCTs in this chamber, as found by the processor.
      In old ALCT algorithms, up to two best ALCT per Level-1 accept window
      had been reported.
      In the ALCT-2006 algorithms, up to two best ALCTs PER EVERY TIME BIN in
      Level-1 accept window are reported. */
  CSCALCTDigi bestALCT[CSCConstants::MAX_ALCT_TBINS];

  /** Second best LCTs in this chamber, as found by the processor. */
  CSCALCTDigi secondALCT[CSCConstants::MAX_ALCT_TBINS];

  PulseArray pulse_;

  CSCShowerDigi shower_;

  /** Access routines to wire digis. */
  bool getDigis(const CSCWireDigiCollection* wiredc);
  void getDigis(const CSCWireDigiCollection* wiredc, const CSCDetId& id);

  int numWireGroups;
  int MESelection;

  int first_bx[CSCConstants::MAX_NUM_WIREGROUPS];
  int first_bx_corrected[CSCConstants::MAX_NUM_WIREGROUPS];

  int quality[CSCConstants::MAX_NUM_WIREGROUPS][CSCConstants::NUM_ALCT_PATTERNS];
  std::vector<CSCWireDigi> digiV[CSCConstants::NUM_LAYERS];

  std::vector<CSCALCTDigi> lct_list;

  std::vector<CSCALCTPreTriggerDigi> thePreTriggerDigis;

  /* data members for high multiplicity triggers */
  void encodeHighMultiplicityBits(
      const std::vector<int> wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIREGROUPS]);
  unsigned inTimeHMT_;
  unsigned outTimeHMT_;
  std::vector<unsigned> thresholds_;
  unsigned showerMinInTBin_;
  unsigned showerMaxInTBin_;
  unsigned showerMinOutTBin_;
  unsigned showerMaxOutTBin_;

  /** Configuration parameters. */
  unsigned int fifo_tbins, fifo_pretrig, drift_delay;
  unsigned int nplanes_hit_pretrig, nplanes_hit_accel_pretrig;
  unsigned int nplanes_hit_pattern, nplanes_hit_accel_pattern;
  unsigned int trig_mode, accel_mode, l1a_window_width;

  /** Phase2: hit persistency length */
  unsigned int hit_persist;

  /** Phase2: separate handle for early time bins */
  int early_tbins;

  /** Phase2: delta BX time depth for ghostCancellationLogic */
  int ghost_cancellation_bx_depth;

  /** Phase2: whether to consider ALCT candidates' qualities
      while doing ghostCancellationLogic on +-1 wire groups */
  bool ghost_cancellation_side_quality;

  /** Phase2: deadtime clocks after pretrigger (extra in addition to drift_delay) */
  unsigned int pretrig_extra_deadtime;

  /** Phase2: whether to use corrected_bx instead of pretrigger BX */
  bool use_corrected_bx;

  /** Phase2: whether to use narrow pattern mask for the rings close to the beam */
  bool narrow_mask_r1;

  /** Default values of configuration parameters. */
  static const unsigned int def_fifo_tbins, def_fifo_pretrig;
  static const unsigned int def_drift_delay;
  static const unsigned int def_nplanes_hit_pretrig, def_nplanes_hit_pattern;
  static const unsigned int def_nplanes_hit_accel_pretrig;
  static const unsigned int def_nplanes_hit_accel_pattern;
  static const unsigned int def_trig_mode, def_accel_mode;
  static const unsigned int def_l1a_window_width;

  /* quality control */
  std::unique_ptr<LCTQualityControl> qualityControl_;

  /** Chosen pattern mask. */
  CSCPatternBank::LCTPatterns alct_pattern_ = {};

  /** Load pattern mask defined by configuration into pattern_mask */
  void loadPatternMask();

  /** Set default values for configuration parameters. */
  void setDefaultConfigParameters();

  /** Make sure that the parameter values are within the allowed range. */
  void checkConfigParameters();

  /** Clears the quality for a given wire and pattern if it is a ghost. */
  void clear(const int wire, const int pattern);

  /* Gets wire times from the wire digis and fills wire[][] vector */
  void readWireDigis(std::vector<int> wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIREGROUPS]);

  /* A pulse array will be used as a bit representation of hit times.
     For example: if a keywire has a bx_time of 3, then 1 shifted
     left 3 will be bit pattern 0000000000001000.  Bits are then added to
     signify the duration of a signal (hit_persist, formerly bx_width).  So
     for the pulse with a hit_persist of 6 will look like 0000000111111000. */
  bool pulseExtension(const std::vector<int> wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIREGROUPS]);

  /* Check that there are nplanes_hit_pretrig or more layers hit in collision
     or accelerator patterns for a particular key_wire.  If so, return
     true and the PatternDetection process will start. */
  bool preTrigger(const int key_wire, const int start_bx);

  /* See if there is a pattern that satisfies nplanes_hit_pattern number of
     layers hit for either the accelerator or collision patterns.  Use
     the pattern with the best quality. */
  bool patternDetection(const int key_wire,
                        std::map<int, std::map<int, CSCALCTDigi::WireContainer> >& hits_in_patterns);

  // remove the invalid wires from the container
  void cleanWireContainer(CSCALCTDigi::WireContainer& wireHits) const;

  //  set the wire hit container
  void setWireContainer(CSCALCTDigi&, CSCALCTDigi::WireContainer& wireHits) const;

  /* In older versions of the ALCT emulation, the ghost cancellation was performed after
     the ALCTs were found. In December 2018 it became clear that during the study of data
     and emulation comparison on 2018 data, a small disagreement between data and emulation
     was found. The changes we implemented then allow re-triggering on one wiregroup after
     some dead time once an earlier ALCT was constructed built on this wiregroup. Before this
     commit the ALCT processor would prohibit the wiregroup from triggering in one event after
     an ALCT was found on that wiregroup. In the firwmare, the wiregroup with ALCT is only dead
     for a few BX before it can be triggered by next muon. The implementation of ghost cancellation
     logic wqas changed to accommodate the re-triggering change while the idea of the ghost
     cancellation logic is kept the same.
  */
  virtual void ghostCancellationLogicOneWire(const int key_wire, int* ghost_cleared);

  virtual int getTempALCTQuality(int temp_quality) const;

  void lctSearch();
  /* Function which enables/disables either collision or accelerator tracks.
     The function uses the trig_mode parameter to decide. */
  void trigMode(const int key_wire);

  /* Function which gives a preference either to the collision patterns
     or accelerator patterns.  The function uses the accel_mode parameter
     to decide. */
  void accelMode(const int key_wire);

  /* Selects two collision and two accelerator ALCTs per time bin with
     the best quality. */
  std::vector<CSCALCTDigi> bestTrackSelector(const std::vector<CSCALCTDigi>& all_alcts);

  /* This method should have been an overloaded > operator, but we
     have to keep it here since need to check values in quality[][]
     array modified according to accel_mode parameter. */
  bool isBetterALCT(const CSCALCTDigi& lhsALCT, const CSCALCTDigi& rhsALCT) const;

  /** Dump ALCT configuration parameters. */
  void dumpConfigParams() const;

  /** Dump digis on wire groups. */
  void dumpDigis(const std::vector<int> wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIREGROUPS]) const;

  void showPatterns(const int key_wire);
};

#endif
