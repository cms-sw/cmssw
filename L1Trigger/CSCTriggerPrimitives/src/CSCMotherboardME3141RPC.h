#ifndef CSCTriggerPrimitives_CSCMotherboardME3141RPC_h
#define CSCTriggerPrimitives_CSCMotherboardME3141RPC_h

/** \class CSCMotherboardME3141R0C
 *
 * Extended CSCMotherboard for ME3141 TMB upgrade
 *
 * \author Sven Dildick March 2014
 *
 * Based on CSCMotherboard code
 *
 */

#include <L1Trigger/CSCTriggerPrimitives/src/CSCMotherboard.h>
#include <DataFormats/RPCDigi/interface/RPCDigiCollection.h>

class CSCGeometry;
class CSCChamber;
class RPCGeometry;

class CSCMotherboardME3141RPC : public CSCMotherboard
{
  typedef std::pair<unsigned int, const RPCDigi*> RPCDigiBX;
  typedef std::vector<RPCDigiBX> RPCDigisBX;
  typedef std::map<int, RPCDigisBX> RPCDigis;

 public:
  /** Normal constructor. */
  CSCMotherboardME3141RPC(unsigned endcap, unsigned station, unsigned sector, 
		 unsigned subsector, unsigned chamber,
		 const edm::ParameterSet& conf);

  /** Default destructor. */
  ~CSCMotherboardME3141RPC();

  /** Run function for normal usage.  Runs cathode and anode LCT processors,
      takes results and correlates into CorrelatedLCT. */
  void run(const CSCWireDigiCollection* wiredc, 
           const CSCComparatorDigiCollection* compdc, 
           const RPCDigiCollection* rpcDigis);

  /** labels for ME1a and ME1B */
  enum MEX1Station {ME31 = 1, ME41};

  /** Clears correlated LCT and passes clear signal on to cathode and anode
      LCT processors. */
  void clear();

  /// set CSC and RPC geometries for the matching needs
  void setCSCGeometry(const CSCGeometry *g) { csc_g = g; }
  void setRPCGeometry(const RPCGeometry *g) { rpc_g = g; }

  // check that the RE31 and RE41 chambers are really there
  bool hasRE31andRE41();

  void retrieveRPCDigis(const RPCDigiCollection* digis, unsigned id);
  std::map<int,std::pair<double,double> > createRPCRollLUT(RPCDetId id);
  int assignRPCRoll(double eta);
  void printRPCTriggerDigis(int minBX, int maxBx);
  
  RPCDigisBX matchingRPCDigis(const CSCCLCTDigi& cLCT, const RPCDigisBX& digis = RPCDigisBX(), bool first = true);  
  RPCDigisBX matchingRPCDigis(const CSCALCTDigi& aLCT, const RPCDigisBX& digis = RPCDigisBX(), bool first = true);  
  RPCDigisBX matchingRPCDigis(const CSCCLCTDigi& cLCT, const CSCALCTDigi& aLCT, const RPCDigisBX& digis = RPCDigisBX(), 
			     bool first = true);  

  unsigned int findQualityRPC(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT, bool hasRPC);

  void matchRPCDigis(enum MEX1Station st);

  void correlateLCTsRPC(CSCALCTDigi bestALCT, CSCALCTDigi secondALCT,
			CSCCLCTDigi bestCLCT, CSCCLCTDigi secondCLCT,
			CSCCorrelatedLCTDigi& lct1, CSCCorrelatedLCTDigi& lct2,
			const RPCDigisBX& digis = RPCDigisBX());

  void correlateLCTsRPC(CSCCLCTDigi bestCLCT,CSCCLCTDigi secondCLCT,
                        RPCDigi rpcDigi, int roll,
                        CSCCorrelatedLCTDigi& lct1,CSCCorrelatedLCTDigi& lct2);
 
  void correlateLCTsRPC(CSCALCTDigi bestALCT, CSCALCTDigi secondALCT,
                        RPCDigi gemPad,
                        CSCCorrelatedLCTDigi& lct1, CSCCorrelatedLCTDigi& lct2);

  CSCCorrelatedLCTDigi constructLCTsRPC(const CSCALCTDigi& alct, const CSCCLCTDigi& clct, const RPCDigisBX& digis = RPCDigisBX()); 
  CSCCorrelatedLCTDigi constructLCTsRPC(const CSCCLCTDigi& clct, const RPCDigi& rpc, int roll, 
                                        bool oldDataFormat);
  CSCCorrelatedLCTDigi constructLCTsRPC(const CSCALCTDigi& alct, const RPCDigi& rpc, bool oldDataFormat);

  /** Methods to sort the LCTs */
  std::vector<CSCCorrelatedLCTDigi> sortLCTsByQuality(int bx);
  std::vector<CSCCorrelatedLCTDigi> sortLCTsByQuality(std::vector<CSCCorrelatedLCTDigi>);
  std::vector<CSCCorrelatedLCTDigi> sortLCTsByGEMDPhi(int bx);
  std::vector<CSCCorrelatedLCTDigi> sortLCTsByGEMDPhi(std::vector<CSCCorrelatedLCTDigi>);

  std::vector<CSCCorrelatedLCTDigi> getLCTs();
  std::vector<CSCCorrelatedLCTDigi> readoutLCTs();

 private: 

  /** for the case when more than 2 LCTs/BX are allowed;
      maximum match window = 15 */
  CSCCorrelatedLCTDigi allLCTs[MAX_LCT_BINS][15][2];

  static const double lut_pt_vs_dphi_rpccsc_me31[8][3];
  static const double lut_pt_vs_dphi_rpccsc_me41[8][3];
  static const double lut_wg_me31_eta_odd[96][2];
  static const double lut_wg_me31_eta_even[96][2];
  static const double lut_wg_me41_eta_odd[96][2];
  static const double lut_wg_me41_eta_even[96][2];

  const CSCGeometry* csc_g;
  const RPCGeometry* rpc_g;

  std::vector<CSCALCTDigi> alctV;
  std::vector<CSCCLCTDigi> clctV;

  /** "preferential" index array in matching window for cross-BX sorting */
  int pref[MAX_LCT_BINS];

  bool match_earliest_clct_me3141_only;

  bool drop_used_clcts;

  unsigned int tmb_cross_bx_algo;

  /** maximum lcts per BX in ME2 */
  unsigned int max_me3141_lcts;

  // masterswitch
  bool runME3141ILT_;

  /// Do RPC matching?
  bool do_rpc_matching;

  /// RPC matching dphi and deta
  double rpc_match_delta_phi_odd;
  double rpc_match_delta_phi_even;
  double rpc_match_delta_eta;

  /// delta BX for RPC pads matching
  int rpc_match_delta_bx;

  /// min eta of LCT for which we require RPC match (we don't throw out LCTs below this min eta) 
  double rpc_match_min_eta;

  /// whether to throw out RPC-fiducial LCTs that have no rpc match
  bool rpc_clear_nomatch_lcts;

  // debug 
  bool debug_rpc_matching_;
  bool debug_luts_;
  bool debug_rpc_dphi;

  //  deltas used to match to RPC digis
  int maxDeltaBXRPC_;
  int maxDeltaStripRPC_;
  int maxDeltaStripRPCOdd_;
  int maxDeltaStripRPCEven_;
  int maxDeltaWg_;

  bool useOldLCTDataFormat_;
  
  // drop low quality stubs if they don't have RPCs
  bool dropLowQualityCLCTsNoRPCs_;

  bool buildLCTfromALCTandRPC_;
  bool buildLCTfromCLCTandRPC_;
  bool buildLCTfromLowQstubandRPC_;

  bool promoteCLCTRPCquality_;
  bool promoteALCTRPCquality_;
  bool promoteALCTRPCpattern_;

  std::map<int,std::pair<double,double> > rpcRollToEtaLimits_;
  std::map<int,int> cscWgToRpcRoll_;

  // map of RPC strip to CSC HS
  std::map<int,int> rpcStripToCscHs_;
  std::map<int,std::pair<int,int>> cscHsToRpcStrip_;

  // map< bx , vector<rpcid, pad> >
  RPCDigis rpcDigis_;
};
#endif
 
