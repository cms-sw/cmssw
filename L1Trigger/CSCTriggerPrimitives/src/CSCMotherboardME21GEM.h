#ifndef L1Trigger_CSCTriggerPrimitives_CSCMotherboardME21GEM_h
#define L1Trigger_CSCTriggerPrimitives_CSCMotherboardME21GEM_h

/** \class CSCMotherboardME21GEM
 *
 * Extended CSCMotherboard for ME21 TMB upgrade
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

class CSCMotherboardME21GEM : public CSCMotherboard
{
  typedef std::pair<unsigned int, GEMPadDigi> GEMPadBX;
  typedef std::vector<GEMPadBX> GEMPadsBX;
  typedef std::map<int, GEMPadsBX> GEMPads;

 public:
  /** Normal constructor. */
  CSCMotherboardME21GEM(unsigned endcap, unsigned station, unsigned sector,
		 unsigned subsector, unsigned chamber,
		 const edm::ParameterSet& conf);

  /** Default destructor. */
  ~CSCMotherboardME21GEM() override;

  void clear();

  /** Run function for normal usage.  Runs cathode and anode LCT processors,
      takes results and correlates into CorrelatedLCT. */
  void run(const CSCWireDigiCollection* wiredc,
           const CSCComparatorDigiCollection* compdc,
           const GEMPadDigiCollection* gemPads);

  /// set CSC and GEM geometries for the matching needs
  void setCSCGeometry(const CSCGeometry *g) { csc_g = g; }
  void setGEMGeometry(const GEMGeometry *g) { gem_g = g; }

  void retrieveGEMPads(const GEMPadDigiCollection* pads, unsigned id);
  void retrieveGEMCoPads();

  std::map<int,std::pair<double,double> > createGEMRollEtaLUT();

  int assignGEMRoll(double eta);
  int deltaRoll(int wg, int roll);
  int deltaPad(int hs, int pad);

  void printGEMTriggerPads(int minBX, int maxBx, bool iscopad = false);

  GEMPadsBX matchingGEMPads(const CSCCLCTDigi& cLCT, const GEMPadsBX& pads = GEMPadsBX(),
                            bool isCopad = false, bool first = true);
  GEMPadsBX matchingGEMPads(const CSCALCTDigi& aLCT, const GEMPadsBX& pads = GEMPadsBX(),
                            bool isCopad = false, bool first = true);
  GEMPadsBX matchingGEMPads(const CSCCLCTDigi& cLCT, const CSCALCTDigi& aLCT, const GEMPadsBX& pads = GEMPadsBX(),
                            bool isCopad = false, bool first = true);
  GEMPadsBX matchingGEMPads(const CSCCLCTDigi& cLCT, const CSCCLCTDigi& aLCT, const GEMPadsBX& pads = GEMPadsBX(),
                            bool isCopad = false);
  GEMPadsBX matchingGEMPads(const CSCALCTDigi& cLCT, const CSCALCTDigi& aLCT, const GEMPadsBX& pads = GEMPadsBX(),
                            bool isCopad = false);
  GEMPadsBX matchingGEMPads(const CSCCLCTDigi& cLCT, const CSCCLCTDigi&,
                            const CSCALCTDigi& aLCT, const CSCALCTDigi&,
                            const GEMPadsBX& pads = GEMPadsBX(),
                            bool isCopad = false, bool first = true);

  unsigned int findQualityGEM(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT,
			      bool hasPad, bool hasCoPad);

  void correlateLCTs(CSCALCTDigi bestALCT, CSCALCTDigi secondALCT,
		     CSCCLCTDigi bestCLCT, CSCCLCTDigi secondCLCT,
                     CSCCorrelatedLCTDigi& lct1, CSCCorrelatedLCTDigi& lct2,
                     const GEMPadsBX& pads = GEMPadsBX(), const GEMPadsBX& copads = GEMPadsBX());

  void correlateLCTsGEM(CSCALCTDigi bestALCT, CSCALCTDigi secondALCT,
			GEMPadDigi gemPad,
			CSCCorrelatedLCTDigi& lct1, CSCCorrelatedLCTDigi& lct2);

  void correlateLCTsGEM(CSCCLCTDigi bestCLCT, CSCCLCTDigi secondCLCT,
                        GEMPadDigi gemPad, int roll,
			CSCCorrelatedLCTDigi& lct1, CSCCorrelatedLCTDigi& lct2);

  void correlateLCTsGEM(CSCALCTDigi bestALCT, CSCALCTDigi secondALCT,
			CSCCLCTDigi bestCLCT, CSCCLCTDigi secondCLCT,
			CSCCorrelatedLCTDigi& lct1, CSCCorrelatedLCTDigi& lct2,
			const GEMPadsBX& pads = GEMPadsBX(), const GEMPadsBX& copads = GEMPadsBX());

  CSCCorrelatedLCTDigi constructLCTsGEM(const CSCALCTDigi& alct, const GEMPadDigi& gem,
                                        bool oldDataFormat = false);
  CSCCorrelatedLCTDigi constructLCTsGEM(const CSCCLCTDigi& clct, const GEMPadDigi& gem, int roll,
                                        bool oldDataFormat = true);
  CSCCorrelatedLCTDigi constructLCTsGEM(const CSCALCTDigi& alct, const CSCCLCTDigi& clct,
					bool hasPad, bool hasCoPad);

  /** additional processor for GEMs */
  std::unique_ptr<GEMCoPadProcessor> coPadProcessor;

  /** Methods to sort the LCTs */
  std::vector<CSCCorrelatedLCTDigi> sortLCTsByQuality(int bx);
  std::vector<CSCCorrelatedLCTDigi> sortLCTsByQuality(std::vector<CSCCorrelatedLCTDigi>);
  std::vector<CSCCorrelatedLCTDigi> sortLCTsByGEMDPhi(int bx);
  std::vector<CSCCorrelatedLCTDigi> sortLCTsByGEMDPhi(std::vector<CSCCorrelatedLCTDigi>);

  std::vector<CSCCorrelatedLCTDigi> getLCTs();
  std::vector<CSCCorrelatedLCTDigi> readoutLCTs();
  std::vector<GEMCoPadDigi> readoutCoPads();

 private:

  /** for the case when more than 2 LCTs/BX are allowed;
      maximum match window = 15 */
  CSCCorrelatedLCTDigi allLCTs[MAX_LCT_BINS][15][2];

  static const double lut_pt_vs_dphi_gemcsc[8][3];
  static const double lut_wg_eta_odd[112][2];
  static const double lut_wg_eta_even[112][2];

  const CSCGeometry* csc_g;
  const GEMGeometry* gem_g;


  std::vector<CSCALCTDigi> alctV;
  std::vector<CSCCLCTDigi> clctV;
  std::vector<GEMCoPadDigi> gemCoPadV;

  /** "preferential" index array in matching window for cross-BX sorting */
  int pref[MAX_LCT_BINS];

  bool match_earliest_clct_me21_only;

  /** whether to not reuse CLCTs that were used by previous matching ALCTs
      in ALCT-to-CLCT algorithm */
  bool drop_used_clcts;

  unsigned int tmb_cross_bx_algo;

  /** maximum lcts per BX in ME2 */
  unsigned int max_me21_lcts;

  // masterswitch
  bool runME21ILT_;

  // debug gem matching
  bool debug_gem_matching;
  bool debug_luts;

  //  deltas used to match to GEM pads
  int maxDeltaBXPad_;
  int maxDeltaPadPad_;
  int maxDeltaPadPadEven_;
  int maxDeltaPadPadOdd_;

  //  deltas used to match to GEM coincidence pads
  int maxDeltaBXCoPad_;
  int maxDeltaPadCoPad_;
  int maxDeltaPadCoPadEven_;
  int maxDeltaPadCoPadOdd_;

  bool doLCTGhostBustingWithGEMs_;

  // drop low quality stubs if they don't have GEMs
  bool dropLowQualityCLCTsNoGEMs_;
  bool dropLowQualityALCTsNoGEMs_;

  // correct LCT timing with GEMs
  bool correctLCTtimingWithGEM_;

  // build LCT from ALCT and GEM
  bool buildLCTfromALCTandGEM_;
  bool buildLCTfromCLCTandGEM_;

  bool useOldLCTDataFormat_;

  // promote ALCT-GEM pattern
  bool promoteALCTGEMpattern_;

  // promote ALCT-GEM quality
  bool promoteALCTGEMquality_;
  bool promoteCLCTGEMquality_;

  std::map<int,std::pair<double,double> > gemRollToEtaLimits_;
  std::map<int,int> cscWgToGemRoll_;

  // map of pad to HS
  std::map<int,int> gemPadToCscHs_;
  std::map<int,std::pair<int,int>> cscHsToGemPad_;

  // map< bx , vector<gemid, pad> >
  GEMPads pads_;
  GEMPads coPads_;
};
#endif
