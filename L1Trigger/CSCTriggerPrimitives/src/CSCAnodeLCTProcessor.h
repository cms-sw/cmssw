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
 * $Date: 2006/09/22 09:19:11 $
 * $Revision: 1.3 $
 *
 */

#include <vector>
#include <FWCore/ParameterSet/interface/ParameterSet.h>
#include <DataFormats/CSCDigi/interface/CSCWireDigiCollection.h>
#include <DataFormats/CSCDigi/interface/CSCALCTDigi.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCConstants.h>

class CSCAnodeLCTProcessor
{
 public:
  /** Normal constructor. */
  CSCAnodeLCTProcessor(unsigned endcap, unsigned station, unsigned sector,
		       unsigned subsector, unsigned chamber,
		       const edm::ParameterSet& conf);

  /** Default constructor. Used for testing. */
  CSCAnodeLCTProcessor();

  /** Clears the LCT containers. */
  void clear();

  /** Runs the LCT processor code. Called in normal running -- gets info from
      a collection of wire digis. */
  std::vector<CSCALCTDigi> run(const CSCWireDigiCollection* wiredc);

  /** Runs the LCT processor code. Called in normal running or in testing
      mode. */
  void run(const int wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES]);

  /** Access routines to wire digis. */
  void getDigis(const CSCWireDigiCollection* wiredc);
  void getDigis(const std::vector<std::vector<CSCWireDigi> > digis);

  /** Best LCT in this chamber, as found by the processor. */
  CSCALCTDigi bestALCT;

  /** Second best LCT in this chamber, as found by the processor. */
  CSCALCTDigi secondALCT;

  /** Returns vector of found ALCTs, if any. */
  std::vector<CSCALCTDigi> getALCTs();

  /** Access to times on wires on any layer. */
  std::vector<int> wireHits(const int layer) const;

  /** Access to time on single wire on any layer. */
  int wireHit(const int layer, const int wire) const;

  /** Pre-defined patterns. */
  enum {NUM_PATTERN_WIRES = 14};
  static const int pattern_envelope[CSCConstants::NUM_ALCT_PATTERNS][NUM_PATTERN_WIRES];
  static const int pattern_mask[CSCConstants::NUM_ALCT_PATTERNS][NUM_PATTERN_WIRES];

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

  std::vector<int> theWireHits[CSCConstants::NUM_LAYERS];

  static const int bx_min;
  static const int bx_max;

  /** Configuration parameters. */
  int bx_width, nph_thresh, nph_pattern, drift_delay, fifo_pretrig;
  int trig_mode, alct_amode;
  int fifo_tbins, l1a_window; // only for test beam mode.

  /** Clears the quality for a given wire and pattern if it is a ghost. */
  void clear(const int wire, const int pattern);

  /** ALCT algorithm methods. */
  void readWireDigis(int wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES]);
  bool pulseExtension(const int wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES]);
  bool preTrigger(const int key_wire);
  void patternDetection(const int key_wire);
  void ghostCancellationLogic();
  void lctSearch();
  void trigMode(const int key_wire);
  void alctAmode(const int key_wire);

  /** Dump ALCT configuration parameters. */
  void dumpConfigParams() const;

  /** Dump digis on wire groups. */
  void dumpDigis(const int wire[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES]) const;

  /** Set times on all layers for all wires. */
  void saveAllHits(const int wires[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES]);

  /** Set times on wires on any layer. */
  void setWireHits(const int layer, const std::vector<int>& wireHits);

  /** Set time on single wire on any layer. */
  void setWireHit(const int layer, const int wire, const int hit);

  void showPatterns(const int key_wire);
};

#endif
