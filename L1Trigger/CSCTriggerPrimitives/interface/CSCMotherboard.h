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
 *
 */

#include "L1Trigger/CSCTriggerPrimitives/interface/CSCAnodeLCTProcessor.h"
#include "L1Trigger/CSCTriggerPrimitives/interface/CSCCathodeLCTProcessor.h"
#include "DataFormats/CSCDigi/interface/CSCCorrelatedLCTDigi.h"

class CSCMotherboard : public CSCBaseboard {
public:
  /** Normal constructor. */
  CSCMotherboard(unsigned endcap,
                 unsigned station,
                 unsigned sector,
                 unsigned subsector,
                 unsigned chamber,
                 const edm::ParameterSet& conf);

  /** Constructor for use during testing. */
  CSCMotherboard();

  /** Default destructor. */
  ~CSCMotherboard() override = default;

  /** Run function for normal usage.  Runs cathode and anode LCT processors,
      takes results and correlates into CorrelatedLCT. */
  virtual void run(const CSCWireDigiCollection* wiredc, const CSCComparatorDigiCollection* compdc);

  /** Returns vector of correlated LCTs in the read-out time window, if any. */
  virtual std::vector<CSCCorrelatedLCTDigi> readoutLCTs() const;

  /** Returns vector of all found correlated LCTs, if any. */
  std::vector<CSCCorrelatedLCTDigi> getLCTs() const;

  /** Clears correlated LCT and passes clear signal on to cathode and anode
      LCT processors. */
  void clear();

  /** Set configuration parameters obtained via EventSetup mechanism. */
  void setConfigParameters(const CSCDBL1TPParameters* conf);

  /** Anode LCT processor. */
  std::unique_ptr<CSCAnodeLCTProcessor> alctProc;

  /** Cathode LCT processor. */
  std::unique_ptr<CSCCathodeLCTProcessor> clctProc;

  // VK: change to protected, to allow inheritance
protected:
  /* Containers for reconstructed ALCTs and CLCTs */
  std::vector<CSCALCTDigi> alctV;
  std::vector<CSCCLCTDigi> clctV;

  /** Container for first correlated LCT. */
  CSCCorrelatedLCTDigi firstLCT[CSCConstants::MAX_LCT_TBINS];

  /** Container for second correlated LCT. */
  CSCCorrelatedLCTDigi secondLCT[CSCConstants::MAX_LCT_TBINS];

  // helper function to return ALCT/CLCT with correct central BX
  CSCALCTDigi getBXShiftedALCT(const CSCALCTDigi&) const;
  CSCCLCTDigi getBXShiftedCLCT(const CSCCLCTDigi&) const;

  /** Configuration parameters. */
  unsigned int mpc_block_me1a;
  unsigned int alct_trig_enable, clct_trig_enable, match_trig_enable;
  unsigned int match_trig_window_size, tmb_l1a_window_size;

  /** SLHC: whether to not reuse ALCTs that were used by previous matching CLCTs */
  bool drop_used_alcts;

  /** SLHC: whether to not reuse CLCTs that were used by previous matching ALCTs */
  bool drop_used_clcts;

  /** SLHC: separate handle for early time bins */
  int early_tbins;

  /** SLHC: whether to readout only the earliest two LCTs in readout window */
  bool readout_earliest_2;

  /** if true: use regular CLCT-to-ALCT matching in TMB
      if false: do ALCT-to-CLCT matching */
  bool clct_to_alct;

  // encode special bits for high-multiplicity triggers
  unsigned int highMultiplicityBits_;
  bool useHighMultiplicityBits_;

  /** Default values of configuration parameters. */
  static const unsigned int def_mpc_block_me1a;
  static const unsigned int def_alct_trig_enable, def_clct_trig_enable;
  static const unsigned int def_match_trig_enable, def_match_trig_window_size;
  static const unsigned int def_tmb_l1a_window_size;

  /* quality control */
  std::unique_ptr<LCTQualityControl> qualityControl_;

  /** Make sure that the parameter values are within the allowed range. */
  void checkConfigParameters();

  void correlateLCTs(const CSCALCTDigi& bestALCT,
                     const CSCALCTDigi& secondALCT,
                     const CSCCLCTDigi& bestCLCT,
                     const CSCCLCTDigi& secondCLCT,
                     int type);

  // This method calculates all the TMB words and then passes them to the
  // constructor of correlated LCTs.
  CSCCorrelatedLCTDigi constructLCTs(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT, int type, int trknmb) const;

  // CLCT pattern number: encodes the pattern number itself
  unsigned int encodePattern(const int clctPattern) const;

  // 4-bit LCT quality number.Made by TMB lookup tables and used for MPC sorting.
  enum class LCT_Quality : unsigned int {
    INVALID = 0,
    NO_CLCT = 1,
    NO_ALCT = 2,
    CLCT_LAYER_TRIGGER = 3,
    LOW_QUALITY = 4,
    MARGINAL_ANODE_CATHODE = 5,
    HQ_ANODE_MARGINAL_CATHODE = 6,
    HQ_CATHODE_MARGINAL_ANODE = 7,
    HQ_ACCEL_ALCT = 8,
    HQ_RESERVED_1 = 9,
    HQ_RESERVED_2 = 10,
    HQ_PATTERN_2_3 = 11,
    HQ_PATTERN_4_5 = 12,
    HQ_PATTERN_6_7 = 13,
    HQ_PATTERN_8_9 = 14,
    HQ_PATTERN_10 = 15
  };

  enum class LCT_QualityRun3 : unsigned int {
    INVALID = 0,
    LowQ = 1,
    MedQ = 2,
    HighQ = 3,
  };

  LCT_Quality findQuality(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT) const;

  LCT_QualityRun3 findQualityRun3(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT) const;

  /** Dump TMB/MPC configuration parameters. */
  void dumpConfigParams() const;

  /* encode high multiplicity bits for Run-3 exotic triggers */
  void encodeHighMultiplicityBits(unsigned alctBits);
};
#endif
