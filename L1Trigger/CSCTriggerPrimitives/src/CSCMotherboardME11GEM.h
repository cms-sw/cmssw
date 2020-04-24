#ifndef L1Trigger_CSCTriggerPrimitives_CSCMotherboardME11GEM_h
#define L1Trigger_CSCTriggerPrimitives_CSCMotherboardME11GEM_h

/** \class CSCMotherboardME11GEM
 *
 * Extended CSCMotherboard for ME11 TMB upgrade
 *
 * \author Sven Dildick March 2014
 *
 * Based on CSCMotherboard code
 *
 */

#include "L1Trigger/CSCTriggerPrimitives/src/CSCMotherboard.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigiCollection.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigiCollection.h"
#include "L1Trigger/CSCTriggerPrimitives/src/GEMCoPadProcessor.h"

class CSCGeometry;
class CSCChamber;
class GEMGeometry;
class GEMSuperChamber;

class CSCMotherboardME11GEM : public CSCMotherboard
{
  typedef std::pair<unsigned int, GEMPadDigi> GEMPadBX;
  typedef std::vector<GEMPadBX> GEMPadsBX;
  typedef std::map<int, GEMPadsBX> GEMPads;

 public:
  /** Normal constructor. */
  CSCMotherboardME11GEM(unsigned endcap, unsigned station, unsigned sector,
		 unsigned subsector, unsigned chamber,
		 const edm::ParameterSet& conf);

  /** Constructor for use during testing. */
  CSCMotherboardME11GEM();

  /** Default destructor. */
  ~CSCMotherboardME11GEM() override;

  /** Run function for normal usage.  Runs cathode and anode LCT processors,
      takes results and correlates into CorrelatedLCT. */
  void run(const CSCWireDigiCollection* wiredc,
	   const CSCComparatorDigiCollection* compdc,
	   const GEMPadDigiCollection* gemPads);

  /** Returns vectors of found correlated LCTs in ME1a and ME1b, if any. */
  std::vector<CSCCorrelatedLCTDigi> getLCTs1a();
  std::vector<CSCCorrelatedLCTDigi> getLCTs1b();

  /** labels for ME1a and ME1B */
  enum ME11Part {ME1B = 1, ME1A=4};

  /** Methods to sort the LCTs */
  std::vector<CSCCorrelatedLCTDigi> sortLCTsByQuality(int bx, enum ME11Part = ME1B);
  std::vector<CSCCorrelatedLCTDigi> sortLCTsByQuality(enum ME11Part = ME1B);
  std::vector<CSCCorrelatedLCTDigi> sortLCTsByQuality(std::vector<CSCCorrelatedLCTDigi>);
  std::vector<CSCCorrelatedLCTDigi> sortLCTsByGEMDPhi(int bx, enum ME11Part = ME1B);
  std::vector<CSCCorrelatedLCTDigi> sortLCTsByGEMDPhi(enum ME11Part = ME1B);
  std::vector<CSCCorrelatedLCTDigi> sortLCTsByGEMDPhi(std::vector<CSCCorrelatedLCTDigi>);

  /** Returns vectors of found ALCTs in ME1a and ME1b, if any. */
  std::vector<CSCALCTDigi> getALCTs1b() {return alctV;}

  /** Returns vectors of found CLCTs in ME1a and ME1b, if any. */
  std::vector<CSCCLCTDigi> getCLCTs1a() {return clctV1a;}
  std::vector<CSCCLCTDigi> getCLCTs1b() {return clctV1b;}

  /** Clears correlated LCT and passes clear signal on to cathode and anode
      LCT processors. */
  void clear();

  /** Set configuration parameters obtained via EventSetup mechanism. */
  void setConfigParameters(const CSCDBL1TPParameters* conf);

  /** additional Cathode LCT processor for ME1a */
  std::unique_ptr<CSCCathodeLCTProcessor> clct1a;

  std::vector<CSCCorrelatedLCTDigi> readoutLCTs1a();
  std::vector<CSCCorrelatedLCTDigi> readoutLCTs1b();
  std::vector<CSCCorrelatedLCTDigi> readoutLCTs(enum ME11Part me1ab);
  std::vector<GEMCoPadDigi> readoutCoPads();

  /** additional processor for GEMs */
  std::unique_ptr<GEMCoPadProcessor> coPadProcessor;

  /// set CSC and GEM geometries for the matching needs
  void setCSCGeometry(const CSCGeometry *g) { csc_g = g; }
  void setGEMGeometry(const GEMGeometry *g) { gem_g = g; }

 private:

  static const int lut_wg_vs_hs_me1b[48][2];
  static const int lut_wg_vs_hs_me1a[48][2];
  static const int lut_wg_vs_hs_me1ag[48][2];
  static const double lut_pt_vs_dphi_gemcsc[8][3];
  static const double lut_wg_etaMin_etaMax_odd[48][3];
  static const double lut_wg_etaMin_etaMax_even[48][3];

  /** SLHC: special configuration parameters for ME11 treatment. */
  bool smartME1aME1b, disableME1a, gangedME1a;

  bool doesALCTCrossCLCT(CSCALCTDigi &a, CSCCLCTDigi &c, int me);

  /** Container for first correlated LCT in ME1a. */
  //CSCCorrelatedLCTDigi firstLCT1a[MAX_LCT_BINS];

  /** Container for second correlated LCT in ME1a. */
  //CSCCorrelatedLCTDigi secondLCT1a[MAX_LCT_BINS];

  /** for the case when more than 2 LCTs/BX are allowed;
      maximum match window = 15 */
  CSCCorrelatedLCTDigi allLCTs1b[MAX_LCT_BINS][15][2];
  CSCCorrelatedLCTDigi allLCTs1a[MAX_LCT_BINS][15][2];

  void correlateLCTsGEM(CSCALCTDigi bestALCT, CSCALCTDigi secondALCT,
			CSCCLCTDigi bestCLCT, CSCCLCTDigi secondCLCT,
			CSCCorrelatedLCTDigi& lct1, CSCCorrelatedLCTDigi& lct2, int me,
			const GEMPadsBX& pads = GEMPadsBX(), const GEMPadsBX& copads = GEMPadsBX());

  void correlateLCTsGEM(CSCALCTDigi bestALCT, CSCALCTDigi secondALCT, GEMPadDigi gemPad,
			CSCCorrelatedLCTDigi& lct1, CSCCorrelatedLCTDigi& lct2, int me);

  void correlateLCTsGEM(CSCCLCTDigi bestCLCT, CSCCLCTDigi secondCLCT, GEMPadDigi gemPad, int roll,
			CSCCorrelatedLCTDigi& lct1, CSCCorrelatedLCTDigi& lct2, int me);

  void retrieveGEMPads(const GEMPadDigiCollection* pads, unsigned id);
  void retrieveGEMCoPads();

  void createGEMRollEtaLUT(bool isEven);

  int assignGEMRoll(double eta);
  int deltaRoll(int wg, int roll);
  int deltaPad(int hs, int pad);

  CSCCorrelatedLCTDigi constructLCTsGEM(const CSCALCTDigi& alct, const GEMPadDigi& gem,
                                        int me, bool oldDataFormat = false);
  CSCCorrelatedLCTDigi constructLCTsGEM(const CSCCLCTDigi& clct, const GEMPadDigi& gem, int roll,
                                        int me, bool oldDataFormat = true);
  CSCCorrelatedLCTDigi constructLCTsGEM(const CSCALCTDigi& alct, const CSCCLCTDigi& clct,
					bool hasPad, bool hasCoPad);

  unsigned int encodePatternGEM(const int ptn, const int highPt);
  unsigned int findQualityGEM(const CSCALCTDigi& aLCT, const GEMPadDigi& gem);
  unsigned int findQualityGEM(const CSCCLCTDigi& cLCT, const GEMPadDigi& gem);
  unsigned int findQualityGEM(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT,
			      bool hasPad, bool hasCoPad);

  void printGEMTriggerPads(int minBX, int maxBx, bool iscopad = false);

  bool isPadInOverlap(int roll);

  GEMPadsBX matchingGEMPads(const CSCCLCTDigi& cLCT, const GEMPadsBX& pads = GEMPadsBX(),
                            enum ME11Part = ME1B, bool isCopad = false, bool first = true);
  GEMPadsBX matchingGEMPads(const CSCALCTDigi& aLCT, const GEMPadsBX& pads = GEMPadsBX(),
                            enum ME11Part = ME1B, bool isCopad = false, bool first = true);
  GEMPadsBX matchingGEMPads(const CSCCLCTDigi& cLCT, const CSCALCTDigi& aLCT, const GEMPadsBX& pads = GEMPadsBX(),
                            enum ME11Part = ME1B, bool isCopad = false, bool first = true);

  std::vector<CSCALCTDigi> alctV;
  std::vector<CSCCLCTDigi> clctV1b;
  std::vector<CSCCLCTDigi> clctV1a;
  std::vector<GEMCoPadDigi> gemCoPadV;

  /** "preferential" index array in matching window for cross-BX sorting */
  int pref[MAX_LCT_BINS];

  bool match_earliest_alct_me11_only;
  bool match_earliest_clct_me11_only;

  /** if true: use regular CLCT-to-ALCT matching in TMB
      if false: do ALCT-to-CLCT matching */
  bool clct_to_alct;

  /** whether to not reuse CLCTs that were used by previous matching ALCTs
      in ALCT-to-CLCT algorithm */
  bool drop_used_clcts;

  unsigned int tmb_cross_bx_algo;

  /** maximum lcts per BX in ME11: 2, 3, 4 or 999 */
  unsigned int max_me11_lcts;

  /// GEM-CSC integrated local algorithm
  bool runME11ILT_;

  const CSCGeometry* csc_g;
  const GEMGeometry* gem_g;

  // debug gem matching
  bool debug_gem_matching;
  bool debug_luts;

  //  deltas used to match to GEM pads
  int maxDeltaBXPad_;
  int maxDeltaPadPad_;
  int maxDeltaBXPadEven_;
  int maxDeltaPadPadEven_;
  int maxDeltaBXPadOdd_;
  int maxDeltaPadPadOdd_;

  //  deltas used to match to GEM coincidence pads
  int maxDeltaBXCoPad_;
  int maxDeltaPadCoPad_;
  int maxDeltaBXCoPadEven_;
  int maxDeltaPadCoPadEven_;
  int maxDeltaBXCoPadOdd_;
  int maxDeltaPadCoPadOdd_;

  // Drop low quality stubs if they don't have GEMs
  bool dropLowQualityCLCTsNoGEMs_ME1a_;
  bool dropLowQualityCLCTsNoGEMs_ME1b_;
  bool dropLowQualityALCTsNoGEMs_ME1a_;
  bool dropLowQualityALCTsNoGEMs_ME1b_;

  // build LCT from ALCT and GEM
  bool buildLCTfromALCTandGEM_ME1a_;
  bool buildLCTfromALCTandGEM_ME1b_;
  bool buildLCTfromCLCTandGEM_ME1a_;
  bool buildLCTfromCLCTandGEM_ME1b_;

  // LCT ghostbusting
  bool doLCTGhostBustingWithGEMs_;

  // correct LCT timing with GEMs
  bool correctLCTtimingWithGEM_;

  // send LCT old dataformat
  bool useOldLCTDataFormat_;

  // promote ALCT-GEM pattern
  bool promoteALCTGEMpattern_;

  // promote ALCT-GEM quality
  bool promoteALCTGEMquality_;
  bool promoteCLCTGEMquality_ME1a_;
  bool promoteCLCTGEMquality_ME1b_;

  // map of roll N to min and max eta
  std::map<int,std::pair<double,double> > gemRollToEtaLimits_;
  std::map<int,std::pair<int,int>> cscWgToGemRoll_;

  // map of pad to HS
  std::map<int,int> gemPadToCscHsME1a_;
  std::map<int,int> gemPadToCscHsME1b_;
  std::map<int,std::pair<int,int>> cscHsToGemPadME1a_;
  std::map<int,std::pair<int,int>> cscHsToGemPadME1b_;

  // map< bx , vector<gemid, pad> >
  GEMPads pads_;
  GEMPads coPads_;
};
#endif
