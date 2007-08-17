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
 * of wire times and arrays of comparator times & comparator results, or
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
 * $Date: 2007/04/18 16:08:55 $
 * $Revision: 1.4 $
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
  void run(int time1[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_WIRES],
	   int time2[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_STRIPS],
	   int triad[CSCConstants::NUM_LAYERS][CSCConstants::MAX_NUM_STRIPS]);

  /** Run function for normal usage.  Runs cathode and anode LCT processors,
      takes results and correlates into CorrelatedLCT. */
  std::vector<CSCCorrelatedLCTDigi> run(const CSCWireDigiCollection* wiredc,
				    const CSCComparatorDigiCollection* compdc);

  /** Returns vector of found correlated LCTs, if any. */
  std::vector<CSCCorrelatedLCTDigi> getLCTs();

  /** Clears correlated LCT and passes clear signal on to cathode and anode
      LCT processors. */
  void clear();

  /** Set configuration parameters obtained via EventSetup mechanism. */
  void setConfigParameters(const L1CSCTPParameters* conf);

  /** Anode LCT processor. */
  CSCAnodeLCTProcessor* alct;

  /** Cathode LCT processor. */
  CSCCathodeLCTProcessor* clct;

 private:
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

  /** Container for first correlated LCT. */
  CSCCorrelatedLCTDigi firstLCT;

  /** Container for second correlated LCT. */
  CSCCorrelatedLCTDigi secondLCT;

  void correlateLCTs(CSCALCTDigi bestALCT, CSCALCTDigi secondALCT,
		     CSCCLCTDigi bestCLCT, CSCCLCTDigi secondCLCT);
  CSCCorrelatedLCTDigi constructLCTs(const CSCALCTDigi& aLCT,
				     const CSCCLCTDigi& cLCT);
  unsigned int encodePattern(const int ptn, const int highPt);
  unsigned int findQuality(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT);

  // Obsolete methods
  int findSTA(const bool, const bool, const bool, const bool);
  int findBxnMatch(const int aBxn, const int cBxn);

  // Method for tests
  void testLCT();
};
#endif
