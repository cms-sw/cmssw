#ifndef CSCTriggerPrimitives_CSCMotherboard_h
#define CSCTriggerPrimitives_CSCMotherboard_h

/** \class CSCMotherboard
 *
 * Correlates anode and cathode LCTs from the same chamber.
 *
 * When the Trigger MotherBoard (TMB) is instantiated it instantiates an ALCT
 * and CLCT board.  The MotherBoard takes up to two LCTs from each anode
 * and cathode LCT card and combines them into a single Correlated LCT.
 * The output is up to two Correlated LCTs.
 *
 * It can be run in either a test mode, where the arguments are a collection
 * of wire times and arrays of halfstrip and distrip times, or
 * for general use, with wire digi and comparator digi collections as
 * arguments.  In the latter mode, the wire & strip info is passed on the
 * LCTProcessors, where it is decoded and converted into a convenient form.
 * After running the anode and cathode LCTProcessors, TMB correlates the
 * anode and cathode LCTs.  At present, it simply matches the best CLCT
 * with the best ALCT; perhaps a better algorithm will be determined in
 * the future.  The MotherBoard then determines a few more numbers (such as
 * quality and pattern) from the ALCT and CLCT information, and constructs
 * two correlated LCT "digis".
 *
 * \author Benn Tannenbaum 28 August 1999 benn@physics.ucla.edu
 *
 * Based on code by Nick Wisniewski (nw@its.caltech.edu) and a framework
 * by Darin Acosta (acosta@phys.ufl.edu).
 *
 * Numerous later improvements by Jason Mumford and Slava Valuev (see cvs
 * in ORCA).
 * Porting from ORCA by S. Valuev (Slava.Valuev@cern.ch), May 2006.
 *
 *
 */

#include <L1Trigger/CSCTriggerPrimitives/src/CSCAnodeLCTProcessor.h>
#include <L1Trigger/CSCTriggerPrimitives/src/CSCCathodeLCTProcessor.h>
#include <DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h>

class CSCMotherboard
{
 public:
  /** Normal constructor. */
  CSCMotherboard(unsigned endcap, unsigned station, unsigned sector, 
		 unsigned subsector, unsigned chamber,
		 const edm::ParameterSet& conf);

  /** Constructor for use during testing. */
  CSCMotherboard();

  /** Default destructor. */
  ~CSCMotherboard();

  /** Test version of run function. */
  void run(const std::vector<int> w_time[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES],
	   const std::vector<int> hs_times[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS],
	   const std::vector<int> ds_times[CSCConstants::NUM_LAYERS][CSCConstants::NUM_HALF_STRIPS_7CFEBS]);

  /** Run function for normal usage.  Runs cathode and anode LCT processors,
      takes results and correlates into CorrelatedLCT. */
  void run(const CSCWireDigiCollection* wiredc, const CSCComparatorDigiCollection* compdc);

  /** Returns vector of correlated LCTs in the read-out time window, if any. */
  std::vector<CSCCorrelatedLCTDigi> readoutLCTs();

  /** Returns vector of all found correlated LCTs, if any. */
  std::vector<CSCCorrelatedLCTDigi> getLCTs();

  /** Clears correlated LCT and passes clear signal on to cathode and anode
      LCT processors. */
  void clear();

  /** Set configuration parameters obtained via EventSetup mechanism. */
  void setConfigParameters(const CSCDBL1TPParameters* conf);

  /** Anode LCT processor. */
  CSCAnodeLCTProcessor* alct;

  /** Cathode LCT processor. */
  CSCCathodeLCTProcessor* clct;

 // VK: change to protected, to allow inheritance
 protected:

  /** Verbosity level: 0: no print (default).
   *                   1: print LCTs found. */
  int infoV;

  /** Chamber id (trigger-type labels). */
  const unsigned theEndcap;
  const unsigned theStation;
  const unsigned theSector;
  const unsigned theSubsector;
  const unsigned theTrigChamber;

  /** Flag for MTCC data. */
  bool isMTCC;

  /** Flag for new (2007) version of TMB firmware. */
  bool isTMB07;

  /** Flag for SLHC studies. */
  bool isSLHC;

  /** Configuration parameters. */
  unsigned int mpc_block_me1a;
  unsigned int alct_trig_enable, clct_trig_enable, match_trig_enable;
  unsigned int match_trig_window_size, tmb_l1a_window_size;

  /** SLHC: whether to not reuse ALCTs that were used by previous matching CLCTs */
  bool drop_used_alcts;

  /** SLHC: separate handle for early time bins */
  int early_tbins;
  
  /** SLHC: whether to readout only the earliest two LCTs in readout window */
  bool readout_earliest_2;

  /** Default values of configuration parameters. */
  static const unsigned int def_mpc_block_me1a;
  static const unsigned int def_alct_trig_enable, def_clct_trig_enable;
  static const unsigned int def_match_trig_enable, def_match_trig_window_size;
  static const unsigned int def_tmb_l1a_window_size;

  /** Maximum number of time bins. */
  enum {MAX_LCT_BINS = 16};

  /** Container for first correlated LCT. */
  CSCCorrelatedLCTDigi firstLCT[MAX_LCT_BINS];

  /** Container for second correlated LCT. */
  CSCCorrelatedLCTDigi secondLCT[MAX_LCT_BINS];

  /** Make sure that the parameter values are within the allowed range. */
  void checkConfigParameters();

  void correlateLCTs(CSCALCTDigi bestALCT, CSCALCTDigi secondALCT,
		     CSCCLCTDigi bestCLCT, CSCCLCTDigi secondCLCT);
  CSCCorrelatedLCTDigi constructLCTs(const CSCALCTDigi& aLCT,
				     const CSCCLCTDigi& cLCT);
  unsigned int encodePattern(const int ptn, const int highPt);
  unsigned int findQuality(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT);

  /** Dump TMB/MPC configuration parameters. */
  void dumpConfigParams() const;

  // Method for tests
  void testLCT();
};
#endif
