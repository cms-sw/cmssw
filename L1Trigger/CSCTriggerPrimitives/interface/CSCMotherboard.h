#ifndef L1Trigger_CSCTriggerPrimitives_CSCMotherboard_h
#define L1Trigger_CSCTriggerPrimitives_CSCMotherboard_h

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
 * of wire times and arrays of halfstrip times, or
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
 * Extended for Run-3 and Phase-2 by Vadim Khotilovich, Tao Huang, Sven Dildick and Giovanni Mocellin
 */

#include "L1Trigger/CSCTriggerPrimitives/interface/CSCAnodeLCTProcessor.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCCathodeLCTProcessor.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/LCTContainer.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCALCTCrossCLCT.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCUpgradeAnodeLCTProcessor.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCUpgradeCathodeLCTProcessor.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/LCTQualityAssignment.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"
#include "DataFormats/CSCDigi/interface/CSCShowerDigi.h"
#include "CondFormats/CSCObjects/interface/CSCL1TPLookupTableCCLUT.h"
#include "CondFormats/CSCObjects/interface/CSCL1TPLookupTableME21ILT.h"
#include "CondFormats/CSCObjects/interface/CSCL1TPLookupTableME11ILT.h"

class CSCMotherboard : public CSCBaseboard {
public:
  /** Normal constructor. */
  CSCMotherboard(unsigned endcap,
                 unsigned station,
                 unsigned sector,
                 unsigned subsector,
                 unsigned chamber,
                 CSCBaseboard::Parameters& conf);

  /** Default destructor. */
  ~CSCMotherboard() override = default;

  struct RunContext {
    const CSCGeometry* cscGeometry_;
    // access to lookup tables via eventsetup
    const CSCL1TPLookupTableCCLUT* lookupTableCCLUT_;
    const CSCL1TPLookupTableME11ILT* lookupTableME11ILT_;
    const CSCL1TPLookupTableME21ILT* lookupTableME21ILT_;
    /** Set configuration parameters obtained via EventSetup mechanism. */
    const CSCDBL1TPParameters* parameters_;
  };

  /** Run function for normal usage.  Runs cathode and anode LCT processors,
      takes results and correlates into CorrelatedLCT. */
  void run(const CSCWireDigiCollection* wiredc, const CSCComparatorDigiCollection* compdc, const RunContext&);

  /*
    Returns vector of good correlated LCTs in the read-out time window.
    LCTs in the BX window [early_tbins,...,late_tbins] are considered
    good for physics. The central LCT BX is time bin 8.
    - tmb_l1a_window_size = 7 (Run-1, Run-2) -> [5, 6, 7, 8, 9, 10, 11]
    - tmb_l1a_window_size = 5 (Run-3)        ->    [6, 7, 8, 9, 10]
    - tmb_l1a_window_size = 3 (Run-4?)       ->       [7, 8, 9]

    Note, this function does not have an exact counterpart in the
    firmware. The reason is that the DAQ of LCTs is not correctly
    simulated in CMSSW - at least the simulation of the L1-accept.
    So, this function corresponds to both the trigger path and the
    DAQ path in the firmware. In general, the function will return
    LCTs that would not be used in the OMTF or EMTF emulator,
    because they are out-of-time relative for tracking purposes. For
    instance an LCT with BX5 would be read out by the DAQ, but would
    likely not be used by the EMTF.
  */
  virtual std::vector<CSCCorrelatedLCTDigi> readoutLCTs() const;

  // LCT selection: at most 2 in each BX
  void selectLCTs();

  /** Returns shower bits */
  std::vector<CSCShowerDigi> readoutShower() const;

  /** Anode LCT processor. */
  std::unique_ptr<CSCAnodeLCTProcessor> alctProc;

  /** Cathode LCT processor. */
  std::unique_ptr<CSCCathodeLCTProcessor> clctProc;

  // VK: change to protected, to allow inheritance
protected:
  std::tuple<std::vector<CSCALCTDigi>, std::vector<CSCCLCTDigi>> runCommon(const CSCWireDigiCollection* wiredc,
                                                                           const CSCComparatorDigiCollection* compdc,
                                                                           const RunContext& context);

  // helper function to return ALCT/CLCT with correct central BX
  CSCALCTDigi getBXShiftedALCT(const CSCALCTDigi&) const;
  CSCCLCTDigi getBXShiftedCLCT(const CSCCLCTDigi&) const;

  /** Configuration parameters. */
  unsigned int match_trig_window_size() const { return match_trig_window_size_; }
  unsigned int match_trig_enable() const { return match_trig_enable_; }

  int preferred_bx_match(unsigned int index) const { return preferred_bx_match_[index]; }
  bool sort_clct_bx() const { return sort_clct_bx_; }

  /*sort CLCT by quality+bending and if CLCTs from different BX have
    same quality+bending, then rank CLCT by timing
   */
  void sortCLCTByQualBend(int alct_bx, std::vector<unsigned>& clctBxVector);

  bool doesALCTCrossCLCT(const CSCALCTDigi&, const CSCCLCTDigi&) const;

  // CLCT pattern number: encodes the pattern number itself
  unsigned int encodePattern(const int clctPattern) const;

  /** Container with all LCTs prior to sorting and selecting. */
  LCTContainer allLCTs_;

  /* quality assignment */
  std::unique_ptr<LCTQualityAssignment> qualityAssignment_;

private:
  /** Clears correlated LCT and passes clear signal on to cathode and anode
      LCT processors. */
  void clear();

  /** Make sure that the parameter values are within the allowed range. */
  void checkConfigParameters();

  /*
     For valid ALCTs in the trigger time window, look for CLCTs within the
     match-time window. Valid CLCTs are matched in-time. If a match was found
     for the best ALCT and best CLCT, also the second best ALCT and second
     best CLCT are sent to a correlation function "correlateLCTs" that will
     make the best-best pair and second-second pair (if applicable).
  */
  void matchALCTCLCT();

  /*
    This function matches maximum two ALCTs with maximum two CLCTs in
    a bunch crossing. The best ALCT is considered the one with the highest
    quality in a BX. Similarly for the best CLCT. If there is just one
    ALCT and just one CLCT, the correlated LCT is made from those two
    components. If there are exactly two ALCTs and two CLCTs, the best
    LCT and second best LCT are formed from the best ALCT-CLCT combination
    and the second best ALCT-CLCT combination. In case there is missing
    information (e.g. second best ALCT, but no second best CLCT), information
    is copied over.
   */
  void correlateLCTs(const CSCALCTDigi& bestALCT,
                     const CSCALCTDigi& secondALCT,
                     const CSCCLCTDigi& bestCLCT,
                     const CSCCLCTDigi& secondCLCT,
                     CSCCorrelatedLCTDigi& bLCT,
                     CSCCorrelatedLCTDigi& sLCT,
                     int type) const;

  /*
     This method calculates all the TMB words and then passes them to the
     constructor of correlated LCTs. The LCT data members are filled with
     information from the ALCT-CLCT combination.
  */
  void constructLCTs(
      const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT, int type, int trknmb, CSCCorrelatedLCTDigi& lct) const;

  /*
    These functions copy valid ALCT/CLCT information to invalid the ALCT/CLCT
    if present, so that we always construct the maximum number of valid LCTs
  */
  void copyValidToInValidALCT(CSCALCTDigi&, CSCALCTDigi&) const;
  void copyValidToInValidCLCT(CSCCLCTDigi&, CSCCLCTDigi&) const;

  /** Dump TMB/MPC configuration parameters. */
  void dumpConfigParams() const;

  /* match cathode shower and anode shower with and/or logic */
  void matchShowers(CSCShowerDigi* anode_showers, CSCShowerDigi* cathode_showers, bool andlogic);

  /* encode high multiplicity bits for Run-3 exotic triggers */
  void encodeHighMultiplicityBits();

  void setConfigParameters(const CSCDBL1TPParameters* conf);

  /* Container with sorted and selected LCTs */
  std::vector<CSCCorrelatedLCTDigi> lctV;

  /*
     Preferential index array in matching window, relative to the ALCT BX.
     Where the central match BX goes first,
     then the closest early, the closest late, etc.
  */
  std::vector<int> preferred_bx_match_;
  // encode special bits for high-multiplicity triggers
  std::vector<unsigned> showerSource_;

  /* quality control */
  std::unique_ptr<LCTQualityControl> qualityControl_;

  /*
    Helper class to check if an ALCT intersects with a CLCT. Normally
    this class should not be used. It is left in the code as a potential
    improvement for ME1/1 when unphysical LCTs are not desired. This
    function is not implemented in the firmware.
  */
  std::unique_ptr<CSCALCTCrossCLCT> cscOverlap_;

  CSCShowerDigi showers_[CSCConstants::MAX_LCT_TBINS];

  unsigned int mpc_block_me1a_;
  unsigned int alct_trig_enable_, clct_trig_enable_, match_trig_enable_;
  unsigned int match_trig_window_size_, tmb_l1a_window_size_;

  /** Phase2: separate handle for early time bins */
  int early_tbins;

  // encode special bits for high-multiplicity triggers
  unsigned thisShowerSource_;

  unsigned minbx_readout_;
  unsigned maxbx_readout_;

  /** Phase2: whether to not reuse CLCTs that were used by previous matching ALCTs */
  bool drop_used_clcts;

  /** Phase2: whether to readout only the earliest two LCTs in readout window */
  bool readout_earliest_2;

  // when set to true, ignore CLCTs found in later BX's
  bool match_earliest_clct_only_;

  bool ignoreAlctCrossClct_;

  /* sort CLCT by bx if true, otherwise sort CLCT by quality+bending */
  bool sort_clct_bx_;

  /** Default values of configuration parameters. */
  static const unsigned int def_mpc_block_me1a;
  static const unsigned int def_alct_trig_enable, def_clct_trig_enable;
  static const unsigned int def_match_trig_enable, def_match_trig_window_size;
  static const unsigned int def_tmb_l1a_window_size;
};
#endif
