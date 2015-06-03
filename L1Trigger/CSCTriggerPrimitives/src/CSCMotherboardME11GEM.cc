#include <L1Trigger/CSCTriggerPrimitives/src/CSCMotherboardME11GEM.h>
#include <FWCore/MessageLogger/interface/MessageLogger.h>
#include <DataFormats/MuonDetId/interface/CSCTriggerNumbering.h>
#include <Geometry/GEMGeometry/interface/GEMGeometry.h>
#include <Geometry/GEMGeometry/interface/GEMEtaPartitionSpecs.h>
#include <L1Trigger/CSCCommonTrigger/interface/CSCTriggerGeometry.h>
#include <DataFormats/Math/interface/deltaPhi.h>
#include <DataFormats/Math/interface/normalizedPhi.h>
#include <cmath>
#include <tuple>
#include <set>
#include "boost/container/flat_set.hpp"

// LUT for which ME1/1 wire group can cross which ME1/a halfstrip
// 1st index: WG number
// 2nd index: inclusive HS range
const int CSCMotherboardME11GEM::lut_wg_vs_hs_me1a[48][2] = {
{0, 95},{0, 95},{0, 95},{0, 95},{0, 95},
{0, 95},{0, 95},{0, 95},{0, 95},{0, 95},
{0, 95},{0, 95},{0, 77},{0, 61},{0, 39},
{0, 22},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1} };
// a modified LUT for ganged ME1a
const int CSCMotherboardME11GEM::lut_wg_vs_hs_me1ag[48][2] = {
{0, 31},{0, 31},{0, 31},{0, 31},{0, 31},
{0, 31},{0, 31},{0, 31},{0, 31},{0, 31},
{0, 31},{0, 31},{0, 31},{0, 31},{0, 31},
{0, 22},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1} };

// LUT for which ME1/1 wire group can cross which ME1/b halfstrip
// 1st index: WG number
// 2nd index: inclusive HS range
const int CSCMotherboardME11GEM::lut_wg_vs_hs_me1b[48][2] = {
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{-1,-1},{-1,-1},{-1,-1},{-1,-1},{-1,-1},
{100, 127},{73, 127},{47, 127},{22, 127},{0, 127},
{0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
{0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
{0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
{0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
{0, 127},{0, 127},{0, 127},{0, 127},{0, 127},
{0, 127},{0, 127},{0, 127},{0, 127},{0, 105},
{0, 93},{0, 78},{0, 63} };

// LUT with bending angles of the GEM-CSC high efficiency patterns (98%)
// 1st index: pt value = {5,10,15,20,30,40}
// 2nd index: bending angle for odd numbered chambers
// 3rd index: bending angle for even numbered chambers
/*
const double CSCMotherboardME11GEM::lut_pt_vs_dphi_gemcsc[7][3] = {
  {5.,  0.02203511, 0.00930056},
  {6 ,  0.0182579 , 0.00790009},
  {10., 0.01066000, 0.00483286},
  {15., 0.00722795, 0.00363230},
  {20., 0.00562598, 0.00304878},
  {30., 0.00416544, 0.00253782},
  {40., 0.00342827, 0.00230833} };
*/

const double CSCMotherboardME11GEM::lut_pt_vs_dphi_gemcsc[8][3] = {
  {3, 0.03971647, 0.01710244},                                    
  {5, 0.02123785, 0.00928431}, 
  {7, 0.01475524, 0.00650928},                                                                                                  
  {10, 0.01023299, 0.00458796},                                                                                                 
  {15, 0.00689220, 0.00331313},                                                                  
  {20, 0.00535176, 0.00276152},                                                                                                                        
  {30, 0.00389050, 0.00224959},               
  {40, 0.00329539, 0.00204670}};

const double CSCMotherboardME11GEM::lut_wg_etaMin_etaMax_odd[48][3] = {
{0, 2.44005, 2.44688},
{1, 2.38863, 2.45035},
{2, 2.32742, 2.43077},
{3, 2.30064, 2.40389},
{4, 2.2746, 2.37775},
{5, 2.24925, 2.35231},
{6, 2.22458, 2.32754},
{7, 2.20054, 2.30339},
{8, 2.1771, 2.27985},
{9, 2.15425, 2.25689},
{10, 2.13194, 2.23447},
{11, 2.11016, 2.21258},
{12, 2.08889, 2.19119},
{13, 2.06809, 2.17028},
{14, 2.04777, 2.14984},
{15, 2.02788, 2.12983},
{16, 2.00843, 2.11025},
{17, 1.98938, 2.09108},
{18, 1.97073, 2.0723},
{19, 1.95246, 2.0539},
{20, 1.93456, 2.03587},
{21, 1.91701, 2.01818},
{22, 1.8998, 2.00084},
{23, 1.88293, 1.98382},
{24, 1.86637, 1.96712},
{25, 1.85012, 1.95073},
{26, 1.83417, 1.93463},
{27, 1.8185, 1.91882},
{28, 1.80312, 1.90329},
{29, 1.788, 1.88803},
{30, 1.77315, 1.87302},
{31, 1.75855, 1.85827},
{32, 1.74421, 1.84377},
{33, 1.7301, 1.8295},
{34, 1.71622, 1.81547},
{35, 1.70257, 1.80166},
{36, 1.68914, 1.78807},
{37, 1.67592, 1.77469},
{38, 1.66292, 1.76151},
{39, 1.65011, 1.74854},
{40, 1.63751, 1.73577},
{41, 1.62509, 1.72319},
{42, 1.61287, 1.71079},
{43, 1.60082, 1.69857},
{44, 1.59924, 1.68654},
{45, 1.6006, 1.67467},
{46, 1.60151, 1.66297},
{47, 1.60198, 1.65144} };

const double CSCMotherboardME11GEM::lut_wg_etaMin_etaMax_even[48][3] = {
{0, 2.3917, 2.39853},
{1, 2.34037, 2.40199},
{2, 2.27928, 2.38244},
{3, 2.25254, 2.35561},
{4, 2.22655, 2.32951},
{5, 2.20127, 2.30412},
{6, 2.17665, 2.27939},
{7, 2.15267, 2.25529},
{8, 2.12929, 2.2318},
{9, 2.1065, 2.20889},
{10, 2.08425, 2.18652},
{11, 2.06253, 2.16468},
{12, 2.04132, 2.14334},
{13, 2.0206, 2.12249},
{14, 2.00033, 2.1021},
{15, 1.98052, 2.08215},
{16, 1.96113, 2.06262},
{17, 1.94215, 2.04351},
{18, 1.92357, 2.02479},
{19, 1.90538, 2.00645},
{20, 1.88755, 1.98847},
{21, 1.87007, 1.97085},
{22, 1.85294, 1.95357},
{23, 1.83614, 1.93662},
{24, 1.81965, 1.91998},
{25, 1.80348, 1.90365},
{26, 1.78761, 1.88762},
{27, 1.77202, 1.87187},
{28, 1.75672, 1.85641},
{29, 1.74168, 1.84121},
{30, 1.72691, 1.82628},
{31, 1.7124, 1.8116},
{32, 1.69813, 1.79716},
{33, 1.68411, 1.78297},
{34, 1.67032, 1.769},
{35, 1.65675, 1.75526},
{36, 1.64341, 1.74174},
{37, 1.63028, 1.72844},
{38, 1.61736, 1.71534},
{39, 1.60465, 1.70245},
{40, 1.59213, 1.68975},
{41, 1.57981, 1.67724},
{42, 1.56767, 1.66492},
{43, 1.55572, 1.65278},
{44, 1.55414, 1.64082},
{45, 1.55549, 1.62903},
{46, 1.5564, 1.61742},
{47, 1.55686, 1.60596} };

CSCMotherboardME11GEM::CSCMotherboardME11GEM(unsigned endcap, unsigned station,
			       unsigned sector, unsigned subsector,
			       unsigned chamber,
			       const edm::ParameterSet& conf) :
		CSCMotherboard(endcap, station, sector, subsector, chamber, conf)
{
  const edm::ParameterSet commonParams(conf.getParameter<edm::ParameterSet>("commonParam"));

  // special configuration parameters for ME11 treatment
  smartME1aME1b = commonParams.getParameter<bool>("smartME1aME1b");
  disableME1a = commonParams.getParameter<bool>("disableME1a");
  gangedME1a = commonParams.getParameter<bool>("gangedME1a");
  runME11ILT_ = commonParams.getParameter<bool>("runME11ILT");

  if (!isSLHC) edm::LogError("L1CSCTPEmulatorConfigError")
    << "+++ Upgrade CSCMotherboardME11GEM constructed while isSLHC is not set! +++\n";
  if (!smartME1aME1b) edm::LogError("L1CSCTPEmulatorConfigError")
    << "+++ Upgrade CSCMotherboardME11GEM constructed while smartME1aME1b is not set! +++\n";

  const edm::ParameterSet alctParams(conf.getParameter<edm::ParameterSet>("alctSLHC"));
  const edm::ParameterSet clctParams(conf.getParameter<edm::ParameterSet>("clctSLHC"));
  const edm::ParameterSet me11tmbParams(conf.getParameter<edm::ParameterSet>("me11tmbSLHCGEM"));

  clct1a.reset( new CSCCathodeLCTProcessor(endcap, station, sector, subsector, chamber, clctParams, commonParams, me11tmbParams) );
  clct1a->setRing(4);

  match_earliest_alct_me11_only = me11tmbParams.getParameter<bool>("matchEarliestAlctME11Only");
  match_earliest_clct_me11_only = me11tmbParams.getParameter<bool>("matchEarliestClctME11Only");

  // if true: use regular CLCT-to-ALCT matching in TMB
  // if false: do ALCT-to-CLCT matching
  clct_to_alct = me11tmbParams.getParameter<bool>("clctToAlct");

  // whether to not reuse CLCTs that were used by previous matching ALCTs
  // in ALCT-to-CLCT algorithm
  drop_used_clcts = me11tmbParams.getParameter<bool>("tmbDropUsedClcts");

  tmb_cross_bx_algo = me11tmbParams.getParameter<unsigned int>("tmbCrossBxAlgorithm");

  // maximum lcts per BX in ME11: 2, 3, 4 or 999
  max_me11_lcts = me11tmbParams.getParameter<unsigned int>("maxME11LCTs");

  pref[0] = match_trig_window_size/2;
  for (unsigned int m=2; m<match_trig_window_size; m+=2)
  {
    pref[m-1] = pref[0] - m/2;
    pref[m]   = pref[0] + m/2;
  }

  //----------------------------------------------------------------------------------------//

  //       G E M  -  C S C   I N T E G R A T E D   L O C A L   A L G O R I T H M

  //----------------------------------------------------------------------------------------//

  /// Do GEM matching?
  do_gem_matching = me11tmbParams.getParameter<bool>("doGemMatching");
  
  /// GEM matching dphi and deta
  gem_match_delta_phi_odd = me11tmbParams.getParameter<double>("gemMatchDeltaPhiOdd");
  gem_match_delta_phi_even = me11tmbParams.getParameter<double>("gemMatchDeltaPhiEven");
  gem_match_delta_eta = me11tmbParams.getParameter<double>("gemMatchDeltaEta");

  /// delta BX for GEM pads matching
  gem_match_delta_bx = me11tmbParams.getParameter<int>("gemMatchDeltaBX");

  /// min eta of LCT for which we require GEM match (we don't throw out LCTs below this min eta)
  gem_match_min_eta = me11tmbParams.getParameter<double>("gemMatchMinEta");
  gem_match_max_eta = me11tmbParams.getParameter<double>("gemMatchMaxEta");

  /// whether to throw out GEM-fiducial LCTs that have no gem match
  gem_clear_nomatch_lcts = me11tmbParams.getParameter<bool>("gemClearNomatchLCTs");

  // debug gem matching
  debug_gem_matching = me11tmbParams.getParameter<bool>("debugMatching");
  debug_luts = me11tmbParams.getParameter<bool>("debugLUTs");
  debug_gem_dphi = me11tmbParams.getParameter<bool>("debugGEMDphi");

  //  deltas used to construct GEM coincidence pads
  maxDeltaBXInCoPad_ = me11tmbParams.getParameter<int>("maxDeltaBXInCoPad");
  maxDeltaPadInCoPad_ = me11tmbParams.getParameter<int>("maxDeltaPadInCoPad");

  //  deltas used to match to GEM pads
  maxDeltaBXPadEven_ = me11tmbParams.getParameter<int>("maxDeltaBXPadEven");
  maxDeltaPadPadEven_ = me11tmbParams.getParameter<int>("maxDeltaPadPadEven");
  maxDeltaBXPadOdd_ = me11tmbParams.getParameter<int>("maxDeltaBXPadOdd");
  maxDeltaPadPadOdd_ = me11tmbParams.getParameter<int>("maxDeltaPadPadOdd");

  //  deltas used to match to GEM coincidence pads
  maxDeltaBXCoPadEven_ = me11tmbParams.getParameter<int>("maxDeltaBXCoPadEven");
  maxDeltaPadCoPadEven_ = me11tmbParams.getParameter<int>("maxDeltaPadCoPadEven");
  maxDeltaBXCoPadOdd_ = me11tmbParams.getParameter<int>("maxDeltaBXCoPadOdd");
  maxDeltaPadCoPadOdd_ = me11tmbParams.getParameter<int>("maxDeltaPadCoPadOdd");

  // drop low quality stubs if they don't have GEMs
  dropLowQualityCLCTsNoGEMs_ME1a_ = me11tmbParams.getParameter<bool>("dropLowQualityCLCTsNoGEMs_ME1a");
  dropLowQualityCLCTsNoGEMs_ME1b_ = me11tmbParams.getParameter<bool>("dropLowQualityCLCTsNoGEMs_ME1b");
  dropLowQualityALCTsNoGEMs_ME1a_ = me11tmbParams.getParameter<bool>("dropLowQualityALCTsNoGEMs_ME1a");
  dropLowQualityALCTsNoGEMs_ME1b_ = me11tmbParams.getParameter<bool>("dropLowQualityALCTsNoGEMs_ME1b");

  // build LCT from ALCT and GEM
  buildLCTfromALCTandGEM_ME1a_ = me11tmbParams.getParameter<bool>("buildLCTfromALCTandGEM_ME1a");
  buildLCTfromALCTandGEM_ME1b_ = me11tmbParams.getParameter<bool>("buildLCTfromALCTandGEM_ME1b");
  buildLCTfromCLCTandGEM_ME1a_ = me11tmbParams.getParameter<bool>("buildLCTfromCLCTandGEM_ME1a");
  buildLCTfromCLCTandGEM_ME1b_ = me11tmbParams.getParameter<bool>("buildLCTfromCLCTandGEM_ME1b");

  // LCT ghostbusting
  doLCTGhostBustingWithGEMs_ = me11tmbParams.getParameter<bool>("doLCTGhostBustingWithGEMs");

  // correct LCT timing with GEMs
  correctLCTtimingWithGEM_ = me11tmbParams.getParameter<bool>("correctLCTtimingWithGEM");

  // use "old" or "new" dataformat for integrated LCTs?
  useOldLCTDataFormat_ = me11tmbParams.getParameter<bool>("useOldLCTDataFormat");

  // promote ALCT-GEM pattern
  promoteALCTGEMpattern_ = me11tmbParams.getParameter<bool>("promoteALCTGEMpattern");

  // promote ALCT-GEM quality
  promoteALCTGEMquality_ = me11tmbParams.getParameter<bool>("promoteALCTGEMquality");
  promoteCLCTGEMquality_ME1a_ = me11tmbParams.getParameter<bool>("promoteCLCTGEMquality_ME1a");
  promoteCLCTGEMquality_ME1b_ = me11tmbParams.getParameter<bool>("promoteCLCTGEMquality_ME1b");

  // send first 2 LCTs
  firstTwoLCTsInChamber_ = me11tmbParams.getParameter<bool>("firstTwoLCTsInChamber");
}


CSCMotherboardME11GEM::CSCMotherboardME11GEM() : CSCMotherboard()
{
  // Constructor used only for testing.

  clct1a.reset( new CSCCathodeLCTProcessor() );
  clct1a->setRing(4);

  pref[0] = match_trig_window_size/2;
  for (unsigned int m=2; m<match_trig_window_size; m+=2)
  {
    pref[m-1] = pref[0] - m/2;
    pref[m]   = pref[0] + m/2;
  }
}


CSCMotherboardME11GEM::~CSCMotherboardME11GEM()
{
}


void CSCMotherboardME11GEM::clear()
{
  CSCMotherboard::clear();
  if (clct1a) clct1a->clear();
  for (int bx = 0; bx < MAX_LCT_BINS; bx++)
  {
    //firstLCT1a[bx].clear();
    //secondLCT1a[bx].clear();
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
      for (int i=0;i<2;i++)
      {
        allLCTs1b[bx][mbx][i].clear();
        allLCTs1a[bx][mbx][i].clear();
      }
  }
  gemRollToEtaLimits_.clear();
  cscWgToGemRoll_.clear();

  gemPadToCscHsME1a_.clear();
  gemPadToCscHsME1b_.clear();

  cscHsToGemPadME1a_.clear();
  cscHsToGemPadME1b_.clear();

  pads_.clear();
  coPads_.clear();
}

// Set configuration parameters obtained via EventSetup mechanism.
void CSCMotherboardME11GEM::setConfigParameters(const CSCDBL1TPParameters* conf)
{
  alct->setConfigParameters(conf);
  clct->setConfigParameters(conf);
  clct1a->setConfigParameters(conf);
  // No config. parameters in DB for the TMB itself yet.
}


void CSCMotherboardME11GEM::run(const CSCWireDigiCollection* wiredc,
                             const CSCComparatorDigiCollection* compdc,
                             const GEMPadDigiCollection* gemPads)
{
  clear();
  
  if (!( alct and clct and  clct1a and smartME1aME1b))
  {
    if (infoV >= 0) edm::LogError("L1CSCTPEmulatorSetupError")
      << "+++ run() called for non-existing ALCT/CLCT processor! +++ \n";
    return;
  }

  alctV = alct->run(wiredc); // run anodeLCT
  clctV1b = clct->run(compdc); // run cathodeLCT in ME1/b
  clctV1a = clct1a->run(compdc); // run cathodeLCT in ME1/a

  bool gemGeometryAvailable(false);
  if (gem_g != nullptr) {
    if (infoV >= 0) edm::LogInfo("L1CSCTPEmulatorSetupInfo")
      << "+++ run() called for GEM-CSC integrated trigger! +++ \n";
    gemGeometryAvailable = true;
  }
  
  int used_clct_mask[20], used_clct_mask_1a[20];
  for (int b=0;b<20;b++)
    used_clct_mask[b] = used_clct_mask_1a[b] = 0;

  // retrieve CSCChamber geometry                                                                                                                                       
  CSCTriggerGeomManager* geo_manager(CSCTriggerGeometry::get());
  const CSCChamber* cscChamberME1b(geo_manager->chamber(theEndcap, theStation, theSector, theSubsector, theTrigChamber));
  const CSCDetId me1bId(cscChamberME1b->id());
  const CSCDetId me1aId(me1bId.endcap(), 1, 4, me1bId.chamber());
  const CSCChamber* cscChamberME1a(csc_g->chamber(me1aId));

  if (runME11ILT_){
      
    // check for GEM geometry
    if (not gemGeometryAvailable){
      if (infoV >= 0) edm::LogError("L1CSCTPEmulatorSetupError")
	<< "+++ run() called for GEM-CSC integrated trigger without valid GEM geometry! +++ \n";
      return;
    }

    // trigger geometry
    const CSCLayer* keyLayerME1b(cscChamberME1b->layer(3));
    const CSCLayerGeometry* keyLayerGeometryME1b(keyLayerME1b->geometry());
    const CSCLayer* keyLayerME1a(cscChamberME1a->layer(3));
    const CSCLayerGeometry* keyLayerGeometryME1a(keyLayerME1a->geometry());

    const bool isEven(me1bId.chamber()%2==0);
    const int region((theEndcap == 1) ? 1: -1);
    const GEMDetId gem_id(region, 1, theStation, 1, me1bId.chamber(), 0);
    const GEMChamber* gemChamber(gem_g->chamber(gem_id));

    // initialize depending on whether even or odd     
    maxDeltaBXPad_ = isEven ? maxDeltaBXPadEven_ : maxDeltaBXPadOdd_;
    maxDeltaPadPad_ = isEven ? maxDeltaPadPadEven_ : maxDeltaPadPadOdd_;
    maxDeltaBXCoPad_ = isEven ? maxDeltaBXCoPadEven_ : maxDeltaBXCoPadOdd_;
    maxDeltaPadCoPad_ = isEven ? maxDeltaPadCoPadEven_ : maxDeltaPadCoPadOdd_;
    
    // LUT<roll,<etaMin,etaMax> >    
    createGEMRollEtaLUT(isEven);
    if (debug_luts){
      LogDebug("CSCMotherboardME11GEM") << "me1b Det "<< me1bId<<" "<< me1bId.rawId() <<" " 
					<< (isEven ? "Even":"odd") <<" chamber "<< me1bId.chamber()<<std::endl;
      if (gemRollToEtaLimits_.size())
        for(auto p : gemRollToEtaLimits_) 
	  LogDebug("CSCMotherboardME11GEM") << "pad "<< p.first << " min eta " << (p.second).first << " max eta " << (p.second).second << std::endl;
    }
    
    // loop on all wiregroups to create a LUT <WG,rollMin,rollMax>
    const int numberOfWG(keyLayerGeometryME1b->numberOfWireGroups());
    for (int i = 0; i< numberOfWG; ++i){
      auto etaMin(isEven ? lut_wg_etaMin_etaMax_even[i][1] : lut_wg_etaMin_etaMax_odd[i][1]); 
      auto etaMax(isEven ? lut_wg_etaMin_etaMax_even[i][2] : lut_wg_etaMin_etaMax_odd[i][2]); 
      cscWgToGemRoll_[i] = std::make_pair(assignGEMRoll(etaMin), assignGEMRoll(etaMax));
    }
    if (debug_luts){
      for(auto p : cscWgToGemRoll_) {
	LogDebug("CSCMotherboardME11GEM") << "WG "<< p.first << " GEM pads " << (p.second).first << " " << (p.second).second << std::endl;
      }
    }

    // pick any roll
    auto randRoll(gemChamber->etaPartition(2));

    // ME1a
    auto nStripsME1a(keyLayerGeometryME1a->numberOfStrips());
    for (float i = 0; i< nStripsME1a; i = i+0.5){
      const LocalPoint lpCSC(keyLayerGeometryME1a->topology()->localPosition(i));
      const GlobalPoint gp(keyLayerME1a->toGlobal(lpCSC));
      const LocalPoint lpGEM(randRoll->toLocal(gp));
      const int HS(i/0.5);
      const bool edge(HS < 4 or HS > 93);
      const float pad(edge ? -99 : randRoll->pad(lpGEM));
      // HS are wrapped-around
      cscHsToGemPadME1a_[HS] = std::make_pair(std::floor(pad),std::ceil(pad));
    }
    // ME1b
    auto nStripsME1b(keyLayerGeometryME1b->numberOfStrips());
    for (float i = 0; i< nStripsME1b; i = i+0.5){
      const LocalPoint lpCSC(keyLayerGeometryME1b->topology()->localPosition(i));
      const GlobalPoint gp(keyLayerME1b->toGlobal(lpCSC));
      const LocalPoint lpGEM(randRoll->toLocal(gp));
      const int HS(i/0.5);
      const bool edge(HS < 5 or HS > 124);
      const float pad(edge ? -99 : randRoll->pad(lpGEM));
      // HS are wrapped-around
      cscHsToGemPadME1b_[HS] = std::make_pair(std::floor(pad),std::ceil(pad));
    }
    if (debug_luts){
      LogDebug("CSCMotherboardME11GEM") << "detId " << me1bId;
      LogDebug("CSCMotherboardME11GEM") << "CSCHSToGEMPad LUT in ME1a";
      for(auto p : cscHsToGemPadME1a_) {
        LogDebug("CSCMotherboardME11GEM") << "CSC HS "<< p.first << " GEM Pad low " << (p.second).first << " GEM Pad high " << (p.second).second;
      }
      LogDebug("CSCMotherboardME11GEM") << "CSCHSToGEMPad LUT in ME1b";
      for(auto p : cscHsToGemPadME1b_) {
        LogDebug("CSCMotherboardME11GEM") << "CSC HS "<< p.first << " GEM Pad low " << (p.second).first << " GEM Pad high " << (p.second).second;
      }
    }

    const int nGEMPads(randRoll->npads());
    for (int i = 0; i< nGEMPads; ++i){
      const LocalPoint lpGEM(randRoll->centreOfPad(i));
      const GlobalPoint gp(randRoll->toGlobal(lpGEM));
      const LocalPoint lpCSCME1a(keyLayerME1a->toLocal(gp));
      const LocalPoint lpCSCME1b(keyLayerME1b->toLocal(gp));
      const float stripME1a(keyLayerGeometryME1a->strip(lpCSCME1a));
      const float stripME1b(keyLayerGeometryME1b->strip(lpCSCME1b));
      // HS are wrapped-around
      gemPadToCscHsME1a_[i] = (int) (stripME1a - 0.25)/0.5;
      gemPadToCscHsME1b_[i] = (int) (stripME1b - 0.25)/0.5;
    }
    if (debug_luts){
      LogDebug("CSCMotherboardME11GEM") << "detId " << me1bId;
      LogDebug("CSCMotherboardME11GEM") << "GEMPadToCSCHs LUT in ME1a";
      for(auto p : gemPadToCscHsME1a_) {
        LogDebug("CSCMotherboardME11GEM") << "GEM Pad "<< p.first << " CSC HS: " << p.second;
      }
      LogDebug("CSCMotherboardME11GEM") << "GEMPadToCSCHs LUT in ME1b";
      for(auto p : gemPadToCscHsME1b_) {
        LogDebug("CSCMotherboardME11GEM") << "GEM Pad "<< p.first << " CSC HS: " << p.second;
      }
    }

    // build coincidence pads
    std::auto_ptr<GEMCoPadDigiCollection> pCoPads(new GEMCoPadDigiCollection());
    buildCoincidencePads(gemPads, *pCoPads, me1bId);
    
    // retrieve pads and copads in a certain BX window for this CSC 
    pads_.clear();
    coPads_.clear();
    retrieveGEMPads(gemPads, gem_id);
    retrieveGEMCoPads(pCoPads.get(), gem_id);
  }

  const bool hasPads(pads_.size()!=0);
  const bool hasCoPads(hasPads and coPads_.size()!=0);
  bool hasLCTs(false);

  // ALCT-centric matching
  for (int bx_alct = 0; bx_alct < CSCAnodeLCTProcessor::MAX_ALCT_BINS; bx_alct++)
  {
    if (alct->bestALCT[bx_alct].isValid())
    {
      const int bx_clct_start(bx_alct - match_trig_window_size/2);
      const int bx_clct_stop(bx_alct + match_trig_window_size/2);
      const int bx_copad_start(bx_alct - maxDeltaBXCoPad_);
      const int bx_copad_stop(bx_alct + maxDeltaBXCoPad_);

      if (debug_gem_matching){ 
        std::cout << "========================================================================" << std::endl;
        std::cout << "ALCT-CLCT matching in ME1/1 chamber: " << cscChamberME1b->id() << std::endl;
        std::cout << "------------------------------------------------------------------------" << std::endl;
        std::cout << "+++ Best ALCT Details: ";
        alct->bestALCT[bx_alct].print();
        std::cout << "+++ Second ALCT Details: ";
        alct->secondALCT[bx_alct].print();
        
        printGEMTriggerPads(bx_clct_start, bx_clct_stop);      
        printGEMTriggerPads(bx_clct_start, bx_clct_stop, true);      
        
        std::cout << "------------------------------------------------------------------------" << std::endl;
        std::cout << "Attempt ALCT-CLCT matching in ME1/b in bx range: [" << bx_clct_start << "," << bx_clct_stop << "]" << std::endl;
      }
      
      // ALCT-to-CLCT matching in ME1b
      int nSuccesFulMatches = 0;
      for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++)
      {
        if (bx_clct < 0 or bx_clct >= CSCCathodeLCTProcessor::MAX_CLCT_BINS) continue;
        if (drop_used_clcts and used_clct_mask[bx_clct]) continue;
        if (clct->bestCLCT[bx_clct].isValid())
        {
          const int quality(clct->bestCLCT[bx_clct].getQuality());
          if (debug_gem_matching) std::cout << "++Valid ME1b CLCT: " << clct->bestCLCT[bx_clct] << std::endl;

	  // pick the pad that corresponds 
	  auto matchingPads(matchingGEMPads(clct->bestCLCT[bx_clct], alct->bestALCT[bx_alct], pads_[bx_clct], ME1B, false));
	  auto matchingCoPads(matchingGEMPads(clct->bestCLCT[bx_clct], alct->bestALCT[bx_alct], coPads_[bx_clct], ME1B, true));
          if (runME11ILT_ and dropLowQualityCLCTsNoGEMs_ME1b_ and quality < 4 and hasPads){
            int nFound(matchingPads.size());
            const bool clctInEdge(clct->bestCLCT[bx_clct].getKeyStrip() < 5 or clct->bestCLCT[bx_clct].getKeyStrip() > 124);
            if (clctInEdge){
              if (debug_gem_matching) std::cout << "\tInfo: low quality CLCT in CSC chamber edge, don't care about GEM pads" << std::endl;
            }
            else {
              if (nFound != 0){
                if (debug_gem_matching) std::cout << "\tInfo: low quality CLCT with " << nFound << " matching GEM trigger pads" << std::endl;
              }
              else {
                if (debug_gem_matching) std::cout << "\tWarning: low quality CLCT without matching GEM trigger pad" << std::endl;
                continue;
              }
            }
          }

          // check timing
          if (runME11ILT_ and correctLCTtimingWithGEM_){
            int nFound(matchingCoPads.size());
            if (nFound != 0 and bx_alct == 6 and bx_clct != 6){
              if (debug_gem_matching) std::cout << "\tInfo: CLCT with incorrect timing" << std::endl;
              continue;
            }
          }
          
          ++nSuccesFulMatches;
      
          hasLCTs = true;
          //	    if (infoV > 1) LogTrace("CSCMotherboard")
          int mbx = bx_clct-bx_clct_start;
          correlateLCTsGEM(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
			   clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct],
			   allLCTs1b[bx_alct][mbx][0], allLCTs1b[bx_alct][mbx][1], ME1B, matchingPads, matchingCoPads);
          if (debug_gem_matching) {
            std::cout << "Successful ALCT-CLCT match in ME1b: bx_alct = " << bx_alct
                      << "; match window: [" << bx_clct_start << "; " << bx_clct_stop
                      << "]; bx_clct = " << bx_clct << std::endl;
            std::cout << "+++ Best CLCT Details: ";
            clct->bestCLCT[bx_clct].print();
            std::cout << "+++ Second CLCT Details: ";
            clct->secondCLCT[bx_clct].print();
          }
          
          if (allLCTs1b[bx_alct][mbx][0].isValid()) {
            used_clct_mask[bx_clct] += 1;
            if (match_earliest_clct_me11_only) break;
          }
        }
      }

      // ALCT-to-GEM matching in ME1b
      int nSuccesFulGEMMatches = 0;
      if (runME11ILT_ and nSuccesFulMatches==0 and buildLCTfromALCTandGEM_ME1b_){
        if (debug_gem_matching) std::cout << "++No valid ALCT-CLCT matches in ME1b" << std::endl;
        for (int bx_gem = bx_copad_start; bx_gem <= bx_copad_stop; bx_gem++) {
          if (not hasCoPads) {
            continue;
          }
          
          // find the best matching copad - first one 
          auto copads(matchingGEMPads(alct->bestALCT[bx_alct], coPads_[bx_gem], ME1B, true));             
          if (debug_gem_matching) std::cout << "\t++Number of matching GEM CoPads in BX " << bx_alct << " : "<< copads.size() << std::endl;
          if (copads.size()==0) {
            continue;
          }
          
          correlateLCTsGEM(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                           *(copads.at(0)).second, allLCTs1b[bx_alct][0][0], allLCTs1b[bx_alct][0][1], ME1B);
          if (allLCTs1b[bx_alct][0][0].isValid()) {
            ++nSuccesFulGEMMatches;            
            if (match_earliest_clct_me11_only) break;
          }
          if (debug_gem_matching) {
            std::cout << "Successful ALCT-GEM CoPad match in ME1b: bx_alct = " << bx_alct << std::endl << std::endl;
            std::cout << "------------------------------------------------------------------------" << std::endl << std::endl;
          }
        }
      }

      if (debug_gem_matching) {
        std::cout << "========================================================================" << std::endl;
        std::cout << "Summary: " << std::endl;
        if (nSuccesFulMatches>1)
          std::cout << "Too many successful ALCT-CLCT matches in ME1b: " << nSuccesFulMatches
                    << ", CSCDetId " << cscChamberME1b->id()
                    << ", bx_alct = " << bx_alct
                    << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
        else if (nSuccesFulMatches==1)
          std::cout << "1 successful ALCT-CLCT match in ME1b: " 
                    << " CSCDetId " << cscChamberME1b->id()
                    << ", bx_alct = " << bx_alct
                    << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
        else if (nSuccesFulGEMMatches==1)
          std::cout << "1 successful ALCT-GEM match in ME1b: " 
                    << " CSCDetId " << cscChamberME1b->id()
                    << ", bx_alct = " << bx_alct
                    << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
        else 
          std::cout << "Unsuccessful ALCT-CLCT match in ME1b: " 
                    << "CSCDetId " << cscChamberME1b->id()
                    << ", bx_alct = " << bx_alct
                    << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;

        std::cout << "------------------------------------------------------------------------" << std::endl;
        std::cout << "Attempt ALCT-CLCT matching in ME1/a in bx range: [" << bx_clct_start << "," << bx_clct_stop << "]" << std::endl;
      }

      // ALCT-to-CLCT matching in ME1a
      nSuccesFulMatches = 0;
      for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++)
      {
        if (bx_clct < 0 or bx_clct >= CSCCathodeLCTProcessor::MAX_CLCT_BINS) continue;
        if (drop_used_clcts and used_clct_mask_1a[bx_clct]) continue;
        if (clct1a->bestCLCT[bx_clct].isValid())
        {
          const int quality(clct1a->bestCLCT[bx_clct].getQuality());
          if (debug_gem_matching) std::cout << "++Valid ME1a CLCT: " << clct1a->bestCLCT[bx_clct] << std::endl;

	  // pick the pad that corresponds 
	  auto matchingPads(matchingGEMPads(clct1a->bestCLCT[bx_clct], alct->bestALCT[bx_alct], pads_[bx_clct], ME1A, false));
	  auto matchingCoPads(matchingGEMPads(clct1a->bestCLCT[bx_clct], alct->bestALCT[bx_alct], coPads_[bx_clct], ME1A, true));
          if (runME11ILT_ and dropLowQualityCLCTsNoGEMs_ME1a_ and quality < 4 and hasPads){
            int nFound(matchingPads.size());
            const bool clctInEdge(clct1a->bestCLCT[bx_clct].getKeyStrip() < 4 or clct1a->bestCLCT[bx_clct].getKeyStrip() > 93);
            if (clctInEdge){
              if (debug_gem_matching) std::cout << "\tInfo: low quality CLCT in CSC chamber edge, don't care about GEM pads" << std::endl;
            }
            else {
              if (nFound != 0){
                if (debug_gem_matching) std::cout << "\tInfo: low quality CLCT with " << nFound << " matching GEM trigger pads" << std::endl;
              }
              else {
                if (debug_gem_matching) std::cout << "\tWarning: low quality CLCT without matching GEM trigger pad" << std::endl;
                continue;
              }
            }
          }
          ++nSuccesFulMatches;
          int mbx = bx_clct-bx_clct_start;
          correlateLCTsGEM(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
			   clct1a->bestCLCT[bx_clct], clct1a->secondCLCT[bx_clct],
			   allLCTs1a[bx_alct][mbx][0], allLCTs1a[bx_alct][mbx][1], ME1A, matchingPads, matchingCoPads);
          if (debug_gem_matching) {
            std::cout << "Successful ALCT-CLCT match in ME1a: bx_alct = " << bx_alct
                      << "; match window: [" << bx_clct_start << "; " << bx_clct_stop
                      << "]; bx_clct = " << bx_clct << std::endl;
            std::cout << "+++ Best CLCT Details: ";
            clct1a->bestCLCT[bx_clct].print();
            std::cout << "+++ Second CLCT Details: ";
            clct1a->secondCLCT[bx_clct].print();
          }
          if (allLCTs1a[bx_alct][mbx][0].isValid()){
            used_clct_mask_1a[bx_clct] += 1;
            if (match_earliest_clct_me11_only) break;
          }
        }
      }

      // ALCT-to-GEM matching in ME1a
      nSuccesFulGEMMatches = 0;
      if (runME11ILT_ and nSuccesFulMatches==0 and buildLCTfromALCTandGEM_ME1a_){
        if (debug_gem_matching) std::cout << "++No valid ALCT-CLCT matches in ME1a" << std::endl;
        for (int bx_gem = bx_copad_start; bx_gem <= bx_copad_stop; bx_gem++) {
          if (not hasCoPads) {
            continue;
          }

          // find the best matching copad - first one 
          auto copads(matchingGEMPads(alct->bestALCT[bx_alct], coPads_[bx_gem], ME1A, true));             
          if (debug_gem_matching) std::cout << "\t++Number of matching GEM CoPads in BX " << bx_alct << " : "<< copads.size() << std::endl;
          if (copads.size()==0) {
            continue;
          }
          
          correlateLCTsGEM(alct->bestALCT[bx_alct], alct->secondALCT[bx_alct],
                           *(copads.at(0)).second, allLCTs1a[bx_alct][0][0], allLCTs1a[bx_alct][0][1], ME1A);
          if (allLCTs1a[bx_alct][0][0].isValid()) {
             ++nSuccesFulGEMMatches;            
            if (match_earliest_clct_me11_only) break;
          }
          if (debug_gem_matching) {
            std::cout << "Successful ALCT-GEM CoPad match in ME1a: bx_alct = " << bx_alct << std::endl << std::endl;
            std::cout << "------------------------------------------------------------------------" << std::endl << std::endl;
          }
        }
      }
            
      if (debug_gem_matching) {
        std::cout << "========================================================================" << std::endl;
        std::cout << "Summary: " << std::endl;
        if (nSuccesFulMatches>1)
          std::cout << "Too many successful ALCT-CLCT matches in ME1a: " << nSuccesFulMatches
                    << ", CSCDetId " << cscChamberME1a->id()
                    << ", bx_alct = " << bx_alct
                    << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
        else if (nSuccesFulMatches==1)
          std::cout << "1 successful ALCT-CLCT match in ME1a: " 
                    << " CSCDetId " << cscChamberME1a->id()
                    << ", bx_alct = " << bx_alct
                    << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
        else if (nSuccesFulGEMMatches==1)
          std::cout << "1 successful ALCT-GEM match in ME1a: " 
                    << " CSCDetId " << cscChamberME1a->id()
                    << ", bx_alct = " << bx_alct
                    << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
        else 
          std::cout << "Unsuccessful ALCT-CLCT match in ME1a: " 
                    << "CSCDetId " << cscChamberME1a->id()
                    << ", bx_alct = " << bx_alct
                    << "; match window: [" << bx_clct_start << "; " << bx_clct_stop << "]" << std::endl;
      }

    } // end of ALCT valid block 
    else {
      auto coPads(coPads_[bx_alct]);
      if (runME11ILT_ and coPads.size()!=0) {
        // keep it simple for the time being, only consider the first copad
        const int bx_clct_start(bx_alct - match_trig_window_size/2);
        const int bx_clct_stop(bx_alct + match_trig_window_size/2);
        
        // matching in ME1b
        if (buildLCTfromCLCTandGEM_ME1b_) {
          for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++) {
            if (bx_clct < 0 or bx_clct >= CSCCathodeLCTProcessor::MAX_CLCT_BINS) continue;
            if (drop_used_clcts and used_clct_mask[bx_clct]) continue;
            if (clct->bestCLCT[bx_clct].isValid()) {
              const int quality(clct->bestCLCT[bx_clct].getQuality());
              // only use high-Q stubs for the time being
              if (quality < 4) continue;
              int mbx = bx_clct-bx_clct_start;            
              correlateLCTsGEM(clct->bestCLCT[bx_clct], clct->secondCLCT[bx_clct], *(coPads[0].second), GEMDetId(coPads[0].first).roll(),
                               allLCTs1b[bx_alct][mbx][0], allLCTs1b[bx_alct][mbx][1], ME1B);
              if (debug_gem_matching) {
                //	    if (infoV > 1) LogTrace("CSCMotherboard")
                std::cout << "Successful GEM-CLCT match in ME1b: bx_alct = " << bx_alct
                          << "; match window: [" << bx_clct_start << "; " << bx_clct_stop
                          << "]; bx_clct = " << bx_clct << std::endl;
                std::cout << "+++ Best CLCT Details: ";
                clct->bestCLCT[bx_clct].print();
                std::cout << "+++ Second CLCT Details: ";
                clct->secondCLCT[bx_clct].print();
              }
              if (allLCTs1b[bx_alct][mbx][0].isValid()) {
                used_clct_mask[bx_clct] += 1;
                if (match_earliest_clct_me11_only) break;
              }
            }
          }
        }

        // matching in ME1a
        if (buildLCTfromCLCTandGEM_ME1a_) {
          for (int bx_clct = bx_clct_start; bx_clct <= bx_clct_stop; bx_clct++) {
            if (bx_clct < 0 || bx_clct >= CSCCathodeLCTProcessor::MAX_CLCT_BINS) continue;
            if (drop_used_clcts && used_clct_mask_1a[bx_clct]) continue;
            if (clct1a->bestCLCT[bx_clct].isValid()){
              const int quality(clct1a->bestCLCT[bx_clct].getQuality());
              // only use high-Q stubs for the time being
              if (quality < 4) continue;
              int mbx = bx_clct-bx_clct_start;            
              correlateLCTsGEM(clct1a->bestCLCT[bx_clct], clct1a->secondCLCT[bx_clct], *(coPads[0].second), GEMDetId(coPads[0].first).roll(),
                               allLCTs1a[bx_alct][mbx][0], allLCTs1a[bx_alct][mbx][1], ME1A);
              if (debug_gem_matching) {
                //	    if (infoV > 1) LogTrace("CSCMotherboard")
                std::cout << "Successful GEM-CLCT match in ME1a: bx_alct = " << bx_alct
                          << "; match window: [" << bx_clct_start << "; " << bx_clct_stop
                          << "]; bx_clct = " << bx_clct << std::endl;
                std::cout << "+++ Best CLCT Details: ";
                clct1a->bestCLCT[bx_clct].print();
                std::cout << "+++ Second CLCT Details: ";
                clct1a->secondCLCT[bx_clct].print();
              }
              if (allLCTs1a[bx_alct][mbx][0].isValid()){
                used_clct_mask_1a[bx_clct] += 1;
                if (match_earliest_clct_me11_only) break;
              }
            }
          }
        }
      }        
    }
  } // end of ALCT-centric matching
  
    // possibly use some discrimination from GEMs
  if (gemGeometryAvailable and runME11ILT_ and do_gem_matching) {
    matchGEMPads(ME1A);
    matchGEMPads(ME1B);
  }
    
  if (hasLCTs and debug_gem_matching){
    std::cout << "========================================================================" << std::endl;
    std::cout << "Counting the LCTs" << std::endl;
    std::cout << "========================================================================" << std::endl;
  }
  
  // reduction of nLCTs per each BX
  for (int bx = 0; bx < MAX_LCT_BINS; bx++)
  {
    // counting
    unsigned int n1a=0, n1b=0;
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
      for (int i=0;i<2;i++)
      {
        int cbx = bx + mbx - match_trig_window_size/2;
        if (allLCTs1b[bx][mbx][i].isValid())
        {
          n1b++;
	  if (infoV > 0) LogDebug("CSCMotherboard") 
	    << "1b LCT"<<i+1<<" "<<bx<<"/"<<cbx<<": "<<allLCTs1b[bx][mbx][i]<<std::endl;
        }
        if (allLCTs1a[bx][mbx][i].isValid())
        {
          n1a++;
	  if (infoV > 0) LogDebug("CSCMotherboard") 
	    << "1a LCT"<<i+1<<" "<<bx<<"/"<<cbx<<": "<<allLCTs1a[bx][mbx][i]<<std::endl;
        }
      }
    if (infoV > 0 and n1a+n1b>0) LogDebug("CSCMotherboard") 
      <<"bx "<<bx<<" nLCT:"<<n1a<<" "<<n1b<<" "<<n1a+n1b<<std::endl;
    
    // some simple cross-bx sorting algorithms
    if (tmb_cross_bx_algo == 1 and (n1a>2 or n1b>2) )
    {
      n1a=0, n1b=0;
      for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
        for (int i=0;i<2;i++)
        {
          if (allLCTs1b[bx][pref[mbx]][i].isValid())
          {
            n1b++;
            if (n1b>2) allLCTs1b[bx][pref[mbx]][i].clear();
          }
          if (allLCTs1a[bx][pref[mbx]][i].isValid())
          {
            n1a++;
            if (n1a>2) allLCTs1a[bx][pref[mbx]][i].clear();
          }
        }

      n1a=0, n1b=0;
      for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
        for (int i=0;i<2;i++)
        {
          int cbx = bx + mbx - match_trig_window_size/2;
          if (allLCTs1b[bx][mbx][i].isValid())
          {
            n1b++;
           if (infoV > 0) LogDebug("CSCMotherboard") 
             << "1b LCT"<<i+1<<" "<<bx<<"/"<<cbx<<": "<<allLCTs1b[bx][mbx][i]<<std::cout;
          }
          if (allLCTs1a[bx][mbx][i].isValid())
          {
            n1a++;
            if (infoV > 0) LogDebug("CSCMotherboard") 
              << "1a LCT"<<i+1<<" "<<bx<<"/"<<cbx<<": "<<allLCTs1a[bx][mbx][i]<<std::cout;
          }
        }
      if (infoV > 0 and n1a+n1b>0) LogDebug("CSCMotherboard") 
        <<"bx "<<bx<<" nnLCT:"<<n1a<<" "<<n1b<<" "<<n1a+n1b<<std::endl;
    } // x-bx sorting

    // Maximum 2 per whole ME11 per BX case:
    // (supposedly, now we should have max 2 per bx in each 1a and 1b)
    if (n1a+n1b > max_me11_lcts and tmb_cross_bx_algo == 1)
    {
      // do it simple so far: take all low eta 1/b stubs
      unsigned int nLCT=n1b;
      n1a=0;
      // right now nLCT<=2; cut 1a if necessary
      for (unsigned int mbx=0; mbx<match_trig_window_size; mbx++)
        for (int i=0;i<2;i++)
          if (allLCTs1a[bx][mbx][i].isValid()) {
            nLCT++;
            if (nLCT>max_me11_lcts) allLCTs1a[bx][mbx][i].clear();
            else n1a++;
          }
      // if (infoV > 0 and nLCT>0) LogDebug("CSCMotherboard") 
//       std::cout <<"bx "<<bx<<" nnnLCT: "<<n1a<<" "<<n1b<<" "<<n1a+n1b<<std::cout;
    }
  }// reduction per bx
  
  bool first = true;
  unsigned int n1b=0, n1a=0;
  for (auto p : readoutLCTs1b())
    {
      if (debug_gem_matching and first){
        std::cout << "========================================================================" << std::endl;
        std::cout << "Counting the final LCTs" << std::endl;
        std::cout << "========================================================================" << std::endl;
        first = false;
        std::cout << "tmb_cross_bx_algo: " << tmb_cross_bx_algo << std::endl;
        
      }
      n1b++;
      if (debug_gem_matching)
        std::cout << "1b LCT "<<n1b<<"  " << p <<std::endl;
    }
  
  for (auto p : readoutLCTs1a())
    {
      if (debug_gem_matching and first){
        std::cout << "========================================================================" << std::endl;
        std::cout << "Counting the final LCTs" << std::endl;
        std::cout << "========================================================================" << std::endl;
        first = false;
        std::cout << "tmb_cross_bx_algo: " << tmb_cross_bx_algo << std::endl;
      }
      n1a++;
      if (debug_gem_matching)
        std::cout << "1a LCT "<<n1a<<"  " << p <<std::endl;
    }
  
  //   if (infoV > 1) LogTrace("CSCMotherboardME11GEM")<<"clct_count E:"<<theEndcap<<"S:"<<theStation<<"R:"<<1<<"C:"
  // 					       <<CSCTriggerNumbering::chamberFromTriggerLabels(theSector,theSubsector, theStation, theTrigChamber)
  // 					       <<"  a "<<n_clct_a<<"  b "<<n_clct_b<<"  ab "<<n_clct_a+n_clct_b;
}

std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11GEM::readoutLCTs1a()
{
  return readoutLCTs(ME1A);
}


std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11GEM::readoutLCTs1b()
{
  return readoutLCTs(ME1B);
}


// Returns vector of read-out correlated LCTs, if any.  Starts with
// the vector of all found LCTs and selects the ones in the read-out
// time window.
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11GEM::readoutLCTs(enum ME11Part me1ab)
{
  std::vector<CSCCorrelatedLCTDigi> tmpV;

  // The start time of the L1A*LCT coincidence window should be related
  // to the fifo_pretrig parameter, but I am not completely sure how.
  // Just choose it such that the window is centered at bx=7.  This may
  // need further tweaking if the value of tmb_l1a_window_size changes.
  //static int early_tbins = 4;
  // The number of LCT bins in the read-out is given by the
  // tmb_l1a_window_size parameter, forced to be odd
  static int lct_bins   = 
    (tmb_l1a_window_size % 2 == 0) ? tmb_l1a_window_size + 1 : tmb_l1a_window_size;
  static int late_tbins = early_tbins + lct_bins;


  // Start from the vector of all found correlated LCTs and select
  // those within the LCT*L1A coincidence window.
  int bx_readout = -1;
  std::vector<CSCCorrelatedLCTDigi> tmp_lcts;
  std::vector<CSCCorrelatedLCTDigi> all_lcts;
  if (me1ab == ME1A) tmp_lcts = getLCTs1a();
  if (me1ab == ME1B) tmp_lcts = getLCTs1b();
  switch(tmb_cross_bx_algo){
  case 0: all_lcts = tmp_lcts;
    break;
  case 1: all_lcts = tmp_lcts; 
    break;
  case 2: all_lcts = sortLCTsByQuality(me1ab); 
    break;
  case 3: all_lcts = sortLCTsByGEMDPhi(me1ab);
    break;
  default: std::cout<<"tmb_cross_bx_algo error" <<std::endl;
    break;
  }
  std::vector <CSCCorrelatedLCTDigi>::const_iterator plct = all_lcts.begin();
  for (; plct != all_lcts.end(); plct++)
  {
    if (!plct->isValid()) continue;

    int bx = (*plct).getBX();
    // Skip LCTs found too early relative to L1Accept.
    if (bx <= early_tbins) continue;

    // Skip LCTs found too late relative to L1Accept.
    if (bx > late_tbins) continue;

    // If (readout_earliest_2) take only LCTs in the earliest bx in the read-out window:
    // in digi->raw step, LCTs have to be packed into the TMB header, and
    // currently there is room just for two.
    if (readout_earliest_2 and (bx_readout == -1 or bx == bx_readout) )
    {
      tmpV.push_back(*plct);
      if (bx_readout == -1) bx_readout = bx;
    }
    else tmpV.push_back(*plct);
  }
  return tmpV;
}


// Returns vector of found correlated LCTs, if any.
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11GEM::getLCTs1b()
{
  std::vector<CSCCorrelatedLCTDigi> tmpV;

  for (int bx = 0; bx < MAX_LCT_BINS; bx++) 
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
      for (int i=0;i<2;i++)
        if (allLCTs1b[bx][mbx][i].isValid()) tmpV.push_back(allLCTs1b[bx][mbx][i]);
  return tmpV;
}


// Returns vector of found correlated LCTs, if any.
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11GEM::getLCTs1a()
{
  std::vector<CSCCorrelatedLCTDigi> tmpV;
  
  // disabled ME1a
  if (mpc_block_me1a or disableME1a) return tmpV;

  // Report all LCTs found.
  for (int bx = 0; bx < MAX_LCT_BINS; bx++)
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++) 
      for (int i=0;i<2;i++)
        if (allLCTs1a[bx][mbx][i].isValid())  tmpV.push_back(allLCTs1a[bx][mbx][i]);
  return tmpV;
}


//sort LCTs by Quality in each BX
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11GEM::sortLCTsByQuality(int bx, enum ME11Part me)
{
  auto allLCTs(me==ME1A ? allLCTs1a : allLCTs1b); 
  std::vector<CSCCorrelatedLCTDigi> LCTs;
  std::vector<CSCCorrelatedLCTDigi> tmpV;
  tmpV.clear();
  LCTs.clear();
  for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++) 
    for (int i=0;i<2;i++)
      if (allLCTs[bx][mbx][i].isValid())  
        LCTs.push_back(allLCTs[bx][mbx][i]);

  std::sort(LCTs.begin(), LCTs.end(), CSCMotherboard::sortByQuality);
  tmpV = LCTs;
  if (tmpV.size()> max_me11_lcts) tmpV.erase(tmpV.begin()+max_me11_lcts, tmpV.end());
  return  tmpV;
}

std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11GEM::sortLCTsByQuality(std::vector<CSCCorrelatedLCTDigi> LCTs)
{
  std::vector<CSCCorrelatedLCTDigi> tmpV;
  tmpV.clear();
  std::sort(LCTs.begin(), LCTs.end(), CSCMotherboard::sortByQuality);
  tmpV = LCTs;
  if (tmpV.size()> max_me11_lcts) tmpV.erase(tmpV.begin()+max_me11_lcts, tmpV.end());
  return  tmpV;
}


//sort LCTs in whole LCTs BX window
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11GEM::sortLCTsByQuality(enum ME11Part me)
{
  std::vector<CSCCorrelatedLCTDigi> LCTs_final;
  LCTs_final.clear();
  for (int bx = 0; bx < MAX_LCT_BINS; bx++)
    {
      std::vector<CSCCorrelatedLCTDigi> LCTs1a;
      std::vector<CSCCorrelatedLCTDigi> LCTs1b;
      std::vector<CSCCorrelatedLCTDigi> LCTs_tmp;
      std::vector<CSCCorrelatedLCTDigi> LCTs_tmp1;
      LCTs1a = sortLCTsByQuality(bx, ME1A);
      LCTs1b = sortLCTsByQuality(bx, ME1B);
      std::vector<CSCCorrelatedLCTDigi>::iterator it1a = LCTs1a.begin();
      std::vector<CSCCorrelatedLCTDigi>::iterator it1b = LCTs1b.begin();
      LCTs_tmp.insert(LCTs_tmp.begin(), LCTs1b.begin(), LCTs1b.end());
      LCTs_tmp.insert(LCTs_tmp.end(), LCTs1a.begin(), LCTs1a.end());
      LCTs_tmp1 = sortLCTsByQuality(LCTs_tmp);//LCTs reduction per BX
      if (firstTwoLCTsInChamber_)
        {
          std::vector<CSCCorrelatedLCTDigi>::iterator itp = LCTs_tmp1.begin();
          for ( ; itp != LCTs_tmp1.end(); itp++)
            {
              if (me==ME1A and it1a != LCTs1a.end() and *itp==*it1a ) 
                {
                  LCTs_final.push_back(*it1a);
                  it1a++;
                }
              if (me==ME1B and it1b != LCTs1b.end() and *itp==*it1b)
                {
                  LCTs_final.push_back(*it1b);
                  it1b++;
                }
            }
        }   
      else {
        if (LCTs1a.size() and LCTs1b.size() and me==ME1A)
          LCTs_final.push_back(*LCTs1a.begin());
        else if (LCTs1a.size() and LCTs1b.size() and me==ME1B)
          LCTs_final.push_back(*LCTs1b.begin());
        else if (LCTs1a.size() and LCTs1b.size()==0 and me==ME1A)
          LCTs_final.insert(LCTs_final.end(), LCTs1a.begin(), LCTs1a.end());
        else if (LCTs1b.size() and LCTs1a.size()==0 and me==ME1B)
          LCTs_final.insert(LCTs_final.end(), LCTs1b.begin(), LCTs1b.end());
      }
    }
  return LCTs_final;
}


//sort LCTs by GEMDPhi in each BX
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11GEM::sortLCTsByGEMDPhi(int bx, enum ME11Part me)
{
   
  auto allLCTs(me==ME1A ? allLCTs1a : allLCTs1b); 
  std::vector<CSCCorrelatedLCTDigi> LCTs;
  std::vector<CSCCorrelatedLCTDigi> tmpV;
  tmpV.clear();
  LCTs.clear();
  for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++) 
    for (int i=0;i<2;i++)
      if (allLCTs[bx][mbx][i].isValid())  
        LCTs.push_back(allLCTs[bx][mbx][i]);

  std::sort(LCTs.begin(), LCTs.end(), CSCMotherboard::sortByGEMDphi);
  tmpV = LCTs;
  if (tmpV.size() > max_me11_lcts) tmpV.erase(tmpV.begin()+max_me11_lcts, tmpV.end());
  return tmpV;
}

std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11GEM::sortLCTsByGEMDPhi(std::vector<CSCCorrelatedLCTDigi> LCTs)
{
  std::vector<CSCCorrelatedLCTDigi> tmpV;
  tmpV.clear();
  std::sort(LCTs.begin(), LCTs.end(), CSCMotherboard::sortByGEMDphi);
  tmpV = LCTs;
  if (tmpV.size() > max_me11_lcts) tmpV.erase(tmpV.begin()+max_me11_lcts, tmpV.end());
  return tmpV;
}


//sort LCTs in whole LCTs BX window
std::vector<CSCCorrelatedLCTDigi> CSCMotherboardME11GEM::sortLCTsByGEMDPhi(enum ME11Part me)
{
  std::vector<CSCCorrelatedLCTDigi> LCTs_final;
  LCTs_final.clear();
  for (int bx = 0; bx < MAX_LCT_BINS; bx++)
    {
      std::vector<CSCCorrelatedLCTDigi> LCTs1a;
      std::vector<CSCCorrelatedLCTDigi> LCTs1b;
      std::vector<CSCCorrelatedLCTDigi> LCTs_tmp;
      std::vector<CSCCorrelatedLCTDigi> LCTs_tmp1;
      LCTs1a = sortLCTsByGEMDPhi(bx, ME1A);
      LCTs1b = sortLCTsByGEMDPhi(bx, ME1B);
      std::vector<CSCCorrelatedLCTDigi>::iterator it1a = LCTs1a.begin();
      std::vector<CSCCorrelatedLCTDigi>::iterator it1b = LCTs1b.begin();
      LCTs_tmp.insert(LCTs_tmp.begin(), LCTs1b.begin(), LCTs1b.end());
      LCTs_tmp.insert(LCTs_tmp.end(), LCTs1a.begin(), LCTs1a.end());
      LCTs_tmp1 = sortLCTsByGEMDPhi(LCTs_tmp);//LCTs reduction per BX
      if (firstTwoLCTsInChamber_)
        {
          std::vector<CSCCorrelatedLCTDigi>::iterator itp = LCTs_tmp1.begin();
          while (itp != LCTs_tmp1.end())
            {
              if (me==ME1B and it1b != LCTs1b.end() and *itp==*it1b)
                {
                  LCTs_final.push_back(*it1b);
                  it1b++;
                }
              if (me==ME1A and it1a != LCTs1a.end() and *itp==*it1a) 
                {
                  LCTs_final.push_back(*it1a);
                  it1a++;
                }
              itp++;
            }
        }
      else {
        if (LCTs1a.size() and LCTs1b.size() and me==ME1A)
          LCTs_final.push_back(*LCTs1a.begin());
        else if (LCTs1a.size() and LCTs1b.size() and me==ME1B)
          LCTs_final.push_back(*LCTs1b.begin());
        else if (LCTs1a.size() and LCTs1b.size()==0 and me==ME1A)
          LCTs_final.insert(LCTs_final.end(), LCTs1a.begin(), LCTs1a.end());
        else if (LCTs1b.size() and LCTs1a.size()==0 and me==ME1B)
          LCTs_final.insert(LCTs_final.end(), LCTs1b.begin(), LCTs1b.end());
      }
    }
  return LCTs_final;
}


bool CSCMotherboardME11GEM::doesALCTCrossCLCT(CSCALCTDigi &a, CSCCLCTDigi &c, int me)
{
  if ( !c.isValid() or !a.isValid() ) return false;
  int key_hs = c.getKeyStrip();
  int key_wg = a.getKeyWG();
  if ( me == ME1A )
  {
    if ( !gangedME1a )
    {
      // wrap around ME11 HS number for -z endcap
      if (theEndcap==2) key_hs = 95 - key_hs;
      if ( key_hs >= lut_wg_vs_hs_me1a[key_wg][0] and 
           key_hs <= lut_wg_vs_hs_me1a[key_wg][1]    ) return true;
      return false;
    }
    else
    {
      if (theEndcap==2) key_hs = 31 - key_hs;
      if ( key_hs >= lut_wg_vs_hs_me1ag[key_wg][0] and
           key_hs <= lut_wg_vs_hs_me1ag[key_wg][1]    ) return true;
      return false;
    }
  }
  if ( me == ME1B)
  {
    if (theEndcap==2) key_hs = 127 - key_hs;
    if ( key_hs >= lut_wg_vs_hs_me1b[key_wg][0] and 
         key_hs <= lut_wg_vs_hs_me1b[key_wg][1]      ) return true;
  }
  return false;
}

void CSCMotherboardME11GEM::correlateLCTs(CSCALCTDigi bestALCT,
				   CSCALCTDigi secondALCT,
				   CSCCLCTDigi bestCLCT,
				   CSCCLCTDigi secondCLCT,
				   CSCCorrelatedLCTDigi& lct1,
				   CSCCorrelatedLCTDigi& lct2)
{
  bool anodeBestValid     = bestALCT.isValid();
  bool anodeSecondValid   = secondALCT.isValid();
  bool cathodeBestValid   = bestCLCT.isValid();
  bool cathodeSecondValid = secondCLCT.isValid();

  if (anodeBestValid and !anodeSecondValid)     secondALCT = bestALCT;
  if (!anodeBestValid and anodeSecondValid)     bestALCT   = secondALCT;
  if (cathodeBestValid and !cathodeSecondValid) secondCLCT = bestCLCT;
  if (!cathodeBestValid and cathodeSecondValid) bestCLCT   = secondCLCT;

  // ALCT-CLCT matching conditions are defined by "trig_enable" configuration
  // parameters.
  if ((alct_trig_enable  and bestALCT.isValid()) or
      (clct_trig_enable  and bestCLCT.isValid()) or
      (match_trig_enable and bestALCT.isValid() and bestCLCT.isValid()))
  {
    lct1 = constructLCTs(bestALCT, bestCLCT);
    lct1.setTrknmb(1);
  }

  if (((secondALCT != bestALCT) or (secondCLCT != bestCLCT)) and
      ((alct_trig_enable  and secondALCT.isValid()) or
       (clct_trig_enable  and secondCLCT.isValid()) or
       (match_trig_enable and secondALCT.isValid() and secondCLCT.isValid())))
  {
    lct2 = constructLCTs(secondALCT, secondCLCT);
    lct2.setTrknmb(2);
  }
}


void CSCMotherboardME11GEM::correlateLCTsGEM(CSCALCTDigi bestALCT,
					  CSCALCTDigi secondALCT,
					  GEMPadDigi gemPad,
					  CSCCorrelatedLCTDigi& lct1,
					  CSCCorrelatedLCTDigi& lct2, int ME)
{
  bool anodeBestValid     = bestALCT.isValid();
  bool anodeSecondValid   = secondALCT.isValid();

  if (anodeBestValid and !anodeSecondValid)     secondALCT = bestALCT;
  if (!anodeBestValid and anodeSecondValid)     bestALCT   = secondALCT;

  if ((alct_trig_enable  and bestALCT.isValid()) or
      (match_trig_enable and bestALCT.isValid()))
  {
    lct1 = constructLCTsGEM(bestALCT, gemPad, ME, useOldLCTDataFormat_);
    lct1.setTrknmb(1);
    //    lct1.setGEMDPhi(0.0);
  }

  if ((alct_trig_enable  and secondALCT.isValid()) or
      (match_trig_enable and secondALCT.isValid() and secondALCT != bestALCT))
  {
    lct2 = constructLCTsGEM(secondALCT, gemPad, ME, useOldLCTDataFormat_);
    lct2.setTrknmb(2);
    //    lct2.setGEMDPhi(0.0);
  }
}


void CSCMotherboardME11GEM::correlateLCTsGEM(CSCCLCTDigi bestCLCT,
                                             CSCCLCTDigi secondCLCT,
                                             GEMPadDigi gemPad, int roll,
                                             CSCCorrelatedLCTDigi& lct1,
                                             CSCCorrelatedLCTDigi& lct2, int ME)
{
  bool cathodeBestValid     = bestCLCT.isValid();
  bool cathodeSecondValid   = secondCLCT.isValid();

  if (cathodeBestValid and !cathodeSecondValid)     secondCLCT = bestCLCT;
  if (!cathodeBestValid and cathodeSecondValid)     bestCLCT   = secondCLCT;

  if ((clct_trig_enable  and bestCLCT.isValid()) or
      (match_trig_enable and bestCLCT.isValid()))
  {
    lct1 = constructLCTsGEM(bestCLCT, gemPad, roll, ME, useOldLCTDataFormat_);
    lct1.setTrknmb(1);
  }

  if ((clct_trig_enable  and secondCLCT.isValid()) or
       (match_trig_enable and secondCLCT.isValid() and secondCLCT != bestCLCT))
  {
    lct2 = constructLCTsGEM(secondCLCT, gemPad, roll, ME, useOldLCTDataFormat_);
    lct2.setTrknmb(2);
  }
}

void CSCMotherboardME11GEM::correlateLCTsGEM(CSCALCTDigi bestALCT,
					  CSCALCTDigi secondALCT,
					  CSCCLCTDigi bestCLCT,
					  CSCCLCTDigi secondCLCT,
					  CSCCorrelatedLCTDigi& lct1,
					  CSCCorrelatedLCTDigi& lct2,
					  int me, 
					  const GEMPadsBX& pads, 
					  const GEMPadsBX& copads)
{
  // assume that always anodeBestValid and cathodeBestValid
  
  if (secondALCT == bestALCT) secondALCT.clear();
  if (secondCLCT == bestCLCT) secondCLCT.clear();

  int ok11 = doesALCTCrossCLCT( bestALCT, bestCLCT, me);
  int ok12 = doesALCTCrossCLCT( bestALCT, secondCLCT, me);
  int ok21 = doesALCTCrossCLCT( secondALCT, bestCLCT, me);
  int ok22 = doesALCTCrossCLCT( secondALCT, secondCLCT, me);
  int code = (ok11<<3) | (ok12<<2) | (ok21<<1) | (ok22);

  int dbg=0;
  int ring = me;
  int chamb= CSCTriggerNumbering::chamberFromTriggerLabels(theSector,theSubsector, theStation, theTrigChamber);
  CSCDetId did(theEndcap, theStation, ring, chamb, 0);
  if (dbg) LogTrace("CSCMotherboardME11GEM")<<"debug correlateLCTs in "<<did<<std::endl
	   <<"ALCT1: "<<bestALCT<<std::endl
	   <<"ALCT2: "<<secondALCT<<std::endl
	   <<"CLCT1: "<<bestCLCT<<std::endl
	   <<"CLCT2: "<<secondCLCT<<std::endl
	   <<"ok 11 12 21 22 code = "<<ok11<<" "<<ok12<<" "<<ok21<<" "<<ok22<<" "<<code<<std::endl;

  if ( code==0 ) return;

  // LUT defines correspondence between possible ok## combinations
  // and resulting lct1 and lct2
  int lut[16][2] = {
          //ok: 11 12 21 22
    {0 ,0 }, // 0  0  0  0
    {22,0 }, // 0  0  0  1
    {21,0 }, // 0  0  1  0
    {21,22}, // 0  0  1  1
    {12,0 }, // 0  1  0  0
    {12,22}, // 0  1  0  1
    {12,21}, // 0  1  1  0
    {12,21}, // 0  1  1  1
    {11,0 }, // 1  0  0  0
    {11,22}, // 1  0  0  1
    {11,21}, // 1  0  1  0
    {11,22}, // 1  0  1  1
    {11,12}, // 1  1  0  0
    {11,22}, // 1  1  0  1
    {11,12}, // 1  1  1  0
    {11,22}, // 1  1  1  1
  };

  if (dbg) LogTrace("CSCMotherboardME11GEM")<<"lut 0 1 = "<<lut[code][0]<<" "<<lut[code][1]<<std::endl;

  // first check the special case (11,22) where we have an ambiguity
  const int nPads(pads.size());
  const int nCoPads(copads.size());
  const bool hasPads(nPads!=0);
  const bool hasCoPads(nCoPads!=0);
  
  if (doLCTGhostBustingWithGEMs_ and (lut[code][0] == 11) and (lut[code][0] == 22) and hasPads and (me==ME1B)){

    if (debug_gem_matching) std::cout << "++Info: 2 valid ALCTs-CLCTs pairs with trigger pads. Call the GHOSTBUSTERS!!!" << std::endl;    
    // first check if there are any copads
    typedef std::pair<int,int> mypair;
    // for each trigger pad, store (deltaRoll,deltaHS) for 11,22,12 and 21.
    std::vector<std::tuple<mypair,mypair,mypair,mypair>> deltas;
    deltas.clear();

    if (hasCoPads){
      for (auto p : copads) {
        const GEMDetId detId(p.first);
        const int rollN(detId.roll());
        const int padN((p.second)->pad());
        
        auto t11(std::make_pair(deltaRoll(  bestALCT.getKeyWG(), rollN), deltaPad(  bestCLCT.getKeyStrip(), padN)));
        auto t22(std::make_pair(deltaRoll(secondALCT.getKeyWG(), rollN), deltaPad(secondCLCT.getKeyStrip(), padN)));
        auto t12(std::make_pair(deltaRoll(  bestALCT.getKeyWG(), rollN), deltaPad(secondCLCT.getKeyStrip(), padN)));
        auto t21(std::make_pair(deltaRoll(secondALCT.getKeyWG(), rollN), deltaPad(  bestCLCT.getKeyStrip(), padN)));

        deltas.push_back(std::make_tuple(t11,t22,t12,t21));
      }
      if (debug_gem_matching){
        std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - " << std::endl;
        std::cout << "Printing (deltaRoll, deltaPad) for each (ALCT,CLCT) pair and for each trigger copad" << std::endl;
        for (unsigned i =0; i < deltas.size(); ++i){
          auto c(deltas.at(i));
          std::cout << "\tCoPad " << i << std::endl;
          std::cout << "\t11: " << "(" << std::get<0>(c).first << "," << std::get<0>(c).second << "); "
                    << "22: "   << "(" << std::get<1>(c).first << "," << std::get<1>(c).second << "); " 
                    << "12: "   << "(" << std::get<2>(c).first << "," << std::get<2>(c).second << "); "
                    << "21: "   << "(" << std::get<3>(c).first << "," << std::get<3>(c).second << ")" << std::endl << std::endl;
        }
        std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - " << std::endl;
      }
      
      
//      lct1 = constructLCTs(bestALCT, bestCLCT);
//      lct1.setTrknmb(1);
//      lct2 = constructLCTs(secondALCT, secondCLCT);
//      lct2.setTrknmb(2);
  
//      lct1 = constructLCTs(bestALCT, secondCLCT);
//      lct1.setTrknmb(1);
//      lct2 = constructLCTs(secondLCT, bestCLCT);
//      lct2.setTrknmb(2);
      return;
    }

    // if no copads were found, do the same with pads...
    if (hasPads){
      for (auto p : pads) {
        const GEMDetId detId(p.first);
        const int rollN(detId.roll());
        const int padN((p.second)->pad());
        
        auto t11(std::make_pair(deltaRoll(  bestALCT.getKeyWG(), rollN), deltaPad(  bestCLCT.getKeyStrip(), padN)));
        auto t22(std::make_pair(deltaRoll(secondALCT.getKeyWG(), rollN), deltaPad(secondCLCT.getKeyStrip(), padN)));
        auto t12(std::make_pair(deltaRoll(  bestALCT.getKeyWG(), rollN), deltaPad(secondCLCT.getKeyStrip(), padN)));
        auto t21(std::make_pair(deltaRoll(secondALCT.getKeyWG(), rollN), deltaPad(  bestCLCT.getKeyStrip(), padN)));

        deltas.push_back(std::make_tuple(t11,t22,t12,t21));
      }
      if (debug_gem_matching){
        std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - " << std::endl;
        std::cout << "Printing (deltaRoll, deltaPad) for each (ALCT,CLCT) pair and for each trigger pad" << std::endl;
        for (unsigned i =0; i < deltas.size(); ++i){
          auto c(deltas.at(i));
          std::cout << "\tPad " << i << std::endl;
          std::cout << "\t11: " << "(" << std::get<0>(c).first << "," << std::get<0>(c).second << "); "
                    << "22: "   << "(" << std::get<1>(c).first << "," << std::get<1>(c).second << "); " 
                    << "12: "   << "(" << std::get<2>(c).first << "," << std::get<2>(c).second << "); "
                    << "21: "   << "(" << std::get<3>(c).first << "," << std::get<3>(c).second << ")" << std::endl << std::endl;
        }
        std::cout << "- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - " << std::endl;
      }

      return;
    }
  }

  switch (lut[code][0]) {
    case 11:
      lct1 = constructLCTsGEM(bestALCT, bestCLCT, hasPads, hasCoPads);
      break;
    case 12:
      lct1 = constructLCTsGEM(bestALCT, secondCLCT, hasPads, hasCoPads);
      break;
    case 21:
      lct1 = constructLCTsGEM(secondALCT, bestCLCT, hasPads, hasCoPads);
      break;
    case 22:
      lct1 = constructLCTsGEM(secondALCT, secondCLCT, hasPads, hasCoPads);
      break;
    default: 
      return;  
  }
  lct1.setTrknmb(1);

  if (dbg) LogTrace("CSCMotherboardME11GEM")<<"lct1: "<<lct1<<std::endl;
  
  switch (lut[code][1])
  {
    case 12:
      lct2 = constructLCTsGEM(bestALCT, secondCLCT, hasPads, hasCoPads);
      lct2.setTrknmb(2);
      if (dbg) LogTrace("CSCMotherboardME11GEM")<<"lct2: "<<lct2<<std::endl;
      return;
    case 21:
      lct2 = constructLCTsGEM(secondALCT, bestCLCT, hasPads, hasCoPads);
      lct2.setTrknmb(2);
      if (dbg) LogTrace("CSCMotherboardME11GEM")<<"lct2: "<<lct2<<std::endl;
      return;
    case 22:
      lct2 = constructLCTsGEM(secondALCT, secondCLCT, hasPads, hasCoPads);
      lct2.setTrknmb(2);
      if (dbg) LogTrace("CSCMotherboardME11GEM")<<"lct2: "<<lct2<<std::endl;
      return;
    default: 
      return;
  }
  if (dbg) LogTrace("CSCMotherboardME11GEM")<<"out of correlateLCTs"<<std::endl;

  return;
}


void CSCMotherboardME11GEM::matchGEMPads(enum ME11Part ME)
{
  /*
  using namespace std;

  auto allLCTs(ME==ME1A ? allLCTs1a : allLCTs1b);
  // check if we have any LCTs at all
  int nlct = 0;
  for (int bx = 0; bx < MAX_LCT_BINS; bx++)
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
      for (int i=0;i<2;i++)
      {
        CSCCorrelatedLCTDigi& lct = allLCTs[bx][mbx][i];
        if (lct.isValid()) nlct++;
      }
  if (nlct == 0) return;

  // retrieve CSCChamber geometry
  CSCTriggerGeomManager* geo_manager = CSCTriggerGeometry::get();
  const CSCChamber* cscChamberME1b(geo_manager->chamber(theEndcap, theStation, theSector, theSubsector, theTrigChamber));

  const CSCDetId me1bId(cscChamberME1b->id());
  const CSCDetId me1aId(me1bId.endcap(), 1, 4, me1bId.chamber());
  const CSCDetId me1abId(ME==ME1A ? me1aId : me1bId);
  const int chamber(me1abId.chamber());
  const bool is_odd(chamber%2==1);
  auto cscChamber = csc_g->chamber(me1abId);

  if (debug_gem_dphi) std::cout<<"++++++++  matchGEMPads "<< me1abId <<" +++++++++ "<<std::endl;

  // "key" layer id is used to calculate global position of stub
  CSCDetId key_id(me1abId.endcap(), me1abId.station(), me1abId.ring(), me1abId.chamber(), CSCConstants::KEY_CLCT_LAYER);

  // check if there are any pads 
  if (pads_.empty()) {
    if (debug_gem_dphi) std::cout<<"igotnopads"<<std::endl;
    return;
  }

  // walk over BXs
  for (int bx = 0; bx < MAX_LCT_BINS; ++bx)
  {
    auto in_pads = pads_.find(bx);

    // walk over potential LCTs in this BX
    for (unsigned int mbx = 0; mbx < match_trig_window_size; ++mbx)
      for (int i=0; i<2; ++i)
      {
        CSCCorrelatedLCTDigi& lct = allLCTs[bx][mbx][i];
        if (!lct.isValid() or fabs(lct.getGEMDPhi()) < 0.000001) continue;
        if (debug_gem_dphi) std::cout<<"LCTbefore "<<bx<<" "<<mbx<<" "<<i<<" "<<lct;

        // use -99 as default value whe we don't know if there could have been a gem match
        lct.setGEMDPhi(-99.);

        // "strip" here is actually a half-strip in geometry's terms
        // note that LCT::getStrip() starts from 0
        float fractional_strip = 0.5 * (lct.getStrip() + 1) - 0.25;
        auto layer_geo = cscChamber->layer(CSCConstants::KEY_CLCT_LAYER)->geometry();
        // LCT::getKeyWG() also starts from 0
        float wire = layer_geo->middleWireOfGroup(lct.getKeyWG() + 1);

        LocalPoint csc_intersect = layer_geo->intersectionOfStripAndWire(fractional_strip, wire);
        GlobalPoint csc_gp = csc_g->idToDet(key_id)->surface().toGlobal(csc_intersect);
        
	if (debug_gem_dphi)
 	  std::cout<<"CSC det id"<<key_id <<" strip:"<<fractional_strip<<" wire:"<<wire<<" layer_geo " <<*layer_geo 
	       <<" global position phi:"<<csc_gp.phi()<<std::endl;
       
        // is LCT located in the high efficiency GEM eta range?
        if (is_odd){
          gem_match_min_eta = 1.55;
          gem_match_max_eta = 2.15;
        }else{
          gem_match_min_eta = 1.59;
          gem_match_max_eta = 2.08;
        }
        
        bool gem_fid = ( std::abs(csc_gp.eta()) >= gem_match_min_eta and 
                         std::abs(csc_gp.eta()) <= gem_match_max_eta);

        if (debug_gem_dphi) std::cout<<" lct eta "<<csc_gp.eta()<<" phi "<<csc_gp.phi()<<std::endl;

        if (!gem_fid)
        {
          if (debug_gem_dphi) std::cout<<"    -- lct pass no gem req"<<std::endl;
          continue;
        }
        // use 100 ad default value when within gem fiducial region
        lct.setGEMDPhi(100.);

        if (in_pads == pads_.end()) // has no potential GEM hits with similar BX -> zap it
        {
          if (gem_clear_nomatch_lcts) lct.clear();
          if (debug_gem_dphi) std::cout<<"    -- no gem"<<std::endl;
          continue;
        }
        if (debug_gem_dphi) std::cout<<"    -- gem possible"<<std::endl;
	// use 99 ad default value when we expect there to be a gem match
        lct.setGEMDPhi(99.);

        // to consider a GEM pad as "matched" it has to be 
        // within specified delta_eta and delta_phi ranges
        // and if there are multiple ones, only the min|delta_phi| is considered as matched
        bool gem_matched = false;
        //int gem_bx = 99;
        float min_dphi = 99.;
        for (auto& id_pad: in_pads->second)
        {
          GEMDetId gem_id(id_pad.first);
          LocalPoint gem_lp = gem_g->etaPartition(gem_id)->centreOfPad(id_pad.second->pad());
          GlobalPoint gem_gp = gem_g->idToDet(gem_id)->surface().toGlobal(gem_lp);
          float dphi = deltaPhi(csc_gp.phi(), gem_gp.phi());
          float deta = csc_gp.eta() - gem_gp.eta();
          if (debug_gem_dphi) std::cout<<"pad"<< id_pad.second->pad()<<" phi:"<<gem_gp.phi()
	      <<" gem with dphi "<< std::abs(dphi) <<" deta "<< std::abs(deta) <<std::endl;

          if( (              std::abs(deta) <= gem_match_delta_eta        ) and // within delta_eta
              ( (  is_odd and std::abs(dphi) <= gem_match_delta_phi_odd ) or
                ( !is_odd and std::abs(dphi) <= gem_match_delta_phi_even ) ) and // within delta_phi
              ( std::abs(dphi) < std::abs(min_dphi) )                          // minimal delta phi
            )
          {
            gem_matched = true;
            min_dphi = dphi;
            //gem_bx = id_pad.second->bx();
          }
        }
        if (gem_matched)
        {
          if (debug_gem_dphi) std::cout<<" GOT MATCHED GEM!"<<std::endl;
          lct.setGEMDPhi(min_dphi);
	  // assing the bit value
	  int oddEven = int(not is_odd) + 1;
	  int numberOfBendAngles(sizeof lut_pt_vs_dphi_gemcsc / sizeof *lut_pt_vs_dphi_gemcsc);
	  int iFound = 0;
	  if (abs(min_dphi) < lut_pt_vs_dphi_gemcsc[numberOfBendAngles-1][oddEven]) iFound = numberOfBendAngles;
	  else {
	    for (int i=0; i< numberOfBendAngles-1; ++i) {
	      if (debug_gem_dphi) std::cout<<"is_odd "<<is_odd <<" min_dphi "<<abs(min_dphi)<<" bend angle lib "<<i<<" "<<lut_pt_vs_dphi_gemcsc[i][oddEven]<< std::endl;
	      if (abs(min_dphi) < lut_pt_vs_dphi_gemcsc[i][oddEven] and abs(min_dphi) > lut_pt_vs_dphi_gemcsc[i+1][oddEven]) 
		iFound = i+1;
	    }
	  }
	  lct.setGEMDPhiBits(iFound);
	  if (debug_gem_dphi) std::cout<<"found bend angle "<<abs(min_dphi)<<" "<<lct.getGEMDPhiBits()<<" "<<lut_pt_vs_dphi_gemcsc[iFound][oddEven]<<" "<<iFound << std::endl;
        }
        else
        {
          if (debug_gem_dphi) std::cout<<" no gem match";
          if (gem_clear_nomatch_lcts)
          {
            lct.clear();
            if (debug_gem_dphi) std::cout<<" - cleared lct";
          }
          if (debug_gem_dphi) std::cout<<std::endl;
        }
        if (debug_gem_dphi) std::cout<<"LCTafter "<<bx<<" "<<mbx<<" "<<i<<" "<<lct;
      }
  }

  // final count
  int nlct_after = 0;
  for (int bx = 0; bx < MAX_LCT_BINS; bx++)
    for (unsigned int mbx = 0; mbx < match_trig_window_size; mbx++)
      for (int i=0;i<2;i++)
      {
        if (allLCTs[bx][mbx][i].isValid()) nlct_after++;
      }
  if (debug_gem_dphi) std::cout<<"before "<<nlct<<"  after "<<nlct_after<<std::endl;
  */
}


void CSCMotherboardME11GEM::buildCoincidencePads(const GEMPadDigiCollection* out_pads, 
	                                         GEMCoPadDigiCollection& out_co_pads,
						 CSCDetId csc_id)
{
  gemCoPadV.clear();

  // Build coincidences
  for (auto det_range = out_pads->begin(); det_range != out_pads->end(); ++det_range) {
    const GEMDetId& id = (*det_range).first;
   // same chamber
    if (id.region() != csc_id.zendcap() or id.station() != csc_id.station() or 
	id.ring() != csc_id.ring() or id.chamber() != csc_id.chamber()) continue;

    // all coincidences detIDs will have layer=1
    if (id.layer() != 1) continue;
    
    // find the corresponding id with layer=2
    GEMDetId co_id(id.region(), id.ring(), id.station(), 2, id.chamber(), id.roll());
    
    auto co_pads_range = out_pads->get(co_id);
    // empty range = no possible coincidence pads
    if (co_pads_range.first == co_pads_range.second) continue;
      
    // now let's correlate the pads in two layers of this partition
    const auto& pads_range = (*det_range).second;
    for (auto p = pads_range.first; p != pads_range.second; ++p) {
      for (auto co_p = co_pads_range.first; co_p != co_pads_range.second; ++co_p) {
        // check the match in pad
        if (std::abs(p->pad() - co_p->pad()) > maxDeltaPadInCoPad_) continue;
        // check the match in BX
        if (std::abs(p->bx() - co_p->bx()) > maxDeltaBXInCoPad_ ) continue;

        // make a new coincidence pad digi	
        gemCoPadV.push_back(GEMCoPadDigi(id.roll(),*p,*co_p));
        out_co_pads.insertDigi(id, GEMCoPadDigi(id.roll(),*p,*co_p));
      }
    }
  }
}


void CSCMotherboardME11GEM::createGEMRollEtaLUT(bool isEven)
{
  int ch(isEven ? 2 : 1);
  auto chamber(gem_g->chamber(GEMDetId(1,1,1,1,ch,0)));
  if (chamber==nullptr) return;

  int n = 1;
  if (isEven) n = 2; // this only works for the 9-10 partition geometry!!! FIXME
  for(auto roll : chamber->etaPartitions()) {
    const float half_striplength(roll->specs()->specificTopology().stripLength()/2.);
    const LocalPoint lp_top(0., half_striplength, 0.);
    const LocalPoint lp_bottom(0., -half_striplength, 0.);
    const GlobalPoint gp_top(roll->toGlobal(lp_top));
    const GlobalPoint gp_bottom(roll->toGlobal(lp_bottom));
    gemRollToEtaLimits_[n] = std::make_pair(gp_top.eta(), gp_bottom.eta());
    ++n;
  }
}


int CSCMotherboardME11GEM::assignGEMRoll(double eta)
{
  int result = -99;
  for(auto p : gemRollToEtaLimits_) {
    const float minEta((p.second).first);
    const float maxEta((p.second).second);
    if (minEta <= eta and eta <= maxEta) {
      result = p.first;
      break;
    }
  }
  return result;
}


CSCCorrelatedLCTDigi CSCMotherboardME11GEM::constructLCTsGEM(const CSCALCTDigi& alct,
                                                          const GEMPadDigi& gem, 
							  int ME, bool oldDataFormat) 
{
  auto mymap(ME==ME1A ? gemPadToCscHsME1a_ : gemPadToCscHsME1b_);
  auto wgvshs(ME==ME1A ? lut_wg_vs_hs_me1a : lut_wg_vs_hs_me1b);
  if (oldDataFormat){
    // CLCT pattern number - set it to a highest value
    // hack to get LCTs in the CSCTF
    unsigned int pattern = promoteALCTGEMpattern_ ? 10 : 0;
    
    // LCT quality number - set it to a very high value 
    // hack to get LCTs in the CSCTF
    unsigned int quality = promoteALCTGEMquality_ ? 14 : 11;
    
    // Bunch crossing
    int bx = alct.getBX();
    
    // get keyStrip from LUT
    int keyStrip = mymap[gem.pad()];

    // get wiregroup from ALCT
    int wg = alct.getKeyWG();

    if (keyStrip>wgvshs[wg][0] && keyStrip<wgvshs[wg][1])
    { // construct correlated LCT; temporarily assign track number of 0.
      return CSCCorrelatedLCTDigi(0, 1, quality, wg, keyStrip, pattern, 0, bx, 0, 0, 0, theTrigChamber);
    }
    else return CSCCorrelatedLCTDigi(0,0,0,0,0,0,0,0,0,0,0,0);
   } 
  else {
    
    // CLCT pattern number - no pattern
    unsigned int pattern = 0;

    // LCT quality number
    unsigned int quality = 1;
    
    // Bunch crossing
    int bx = gem.bx() + lct_central_bx;
    
    // get keyStrip from LUT
    int keyStrip = mymap[gem.pad()];

    // get wiregroup from ALCT
    int wg = alct.getKeyWG();
    
    if (keyStrip>wgvshs[wg][0] && keyStrip<wgvshs[wg][1])
    { // construct correlated LCT; temporarily assign track number of 0.
      return CSCCorrelatedLCTDigi(0, 1, quality, wg, keyStrip, pattern, 0, bx, 0, 0, 0, theTrigChamber);
    }
    else return CSCCorrelatedLCTDigi(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
   }
}

CSCCorrelatedLCTDigi CSCMotherboardME11GEM::constructLCTsGEM(const CSCCLCTDigi& clct,
                                                             const GEMPadDigi& gem, int roll,
                                                             int ME, bool oldDataFormat) 
{
//  auto mymap(ME==ME1A ? gemPadToCscHsME1a_ : gemPadToCscHsME1b_);
  if (oldDataFormat){
    // CLCT pattern number - no pattern
    unsigned int pattern = encodePatternGEM(clct.getPattern(), clct.getStripType());
    
    // LCT quality number -  dummy quality
    const bool promoteCLCTGEMquality(ME == ME1A ? promoteCLCTGEMquality_ME1a_:promoteCLCTGEMquality_ME1b_);
    unsigned int quality = promoteCLCTGEMquality ? 14 : 11;
    
    // Bunch crossing: get it from cathode LCT if anode LCT is not there.
    int bx = gem.bx() + lct_central_bx;;
    
   // pick a random WG in the roll range    
    int wg(5);
    
    // construct correlated LCT; temporarily assign track number of 0.
    return CSCCorrelatedLCTDigi(0, 1, quality, wg, clct.getKeyStrip(), pattern, clct.getBend(), bx, 0, 0, 0, theTrigChamber);
  }
  else {
    // CLCT pattern number - no pattern
    unsigned int pattern = encodePatternGEM(clct.getPattern(), clct.getStripType());
    
    // LCT quality number -  dummy quality
    unsigned int quality = 5;//findQualityGEM(alct, gem);
    
    // Bunch crossing: get it from cathode LCT if anode LCT is not there.
    int bx = gem.bx() + lct_central_bx;;
    
    // ALCT WG
    int wg();
    
    // construct correlated LCT; temporarily assign track number of 0.
    return CSCCorrelatedLCTDigi(0, 1, quality, 0, 0, pattern, 0, bx, 0, 0, 0, theTrigChamber);
  }
}


CSCCorrelatedLCTDigi CSCMotherboardME11GEM::constructLCTsGEM(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT, 
							  bool hasPad, bool hasCoPad)
{
  // CLCT pattern number
  unsigned int pattern = encodePattern(cLCT.getPattern(), cLCT.getStripType());

  // LCT quality number
  unsigned int quality = findQualityGEM(aLCT, cLCT, hasPad, hasCoPad);

  // Bunch crossing: get it from cathode LCT if anode LCT is not there.
  int bx = aLCT.isValid() ? aLCT.getBX() : cLCT.getBX();

  // construct correlated LCT; temporarily assign track number of 0.
  int trknmb = 0;
  CSCCorrelatedLCTDigi thisLCT(trknmb, 1, quality, aLCT.getKeyWG(),
                               cLCT.getKeyStrip(), pattern, cLCT.getBend(),
                               bx, 0, 0, 0, theTrigChamber);
  return thisLCT;
}


unsigned int CSCMotherboardME11GEM::encodePatternGEM(const int ptn, const int highPt)
{
  return 0;
}


unsigned int CSCMotherboardME11GEM::findQualityGEM(const CSCALCTDigi& aLCT, const CSCCLCTDigi& cLCT,
						bool hasPad, bool hasCoPad)
{

  /*
    Same LCT quality definition as standard LCTs
    c4 takes GEMs into account!!!
  */

  unsigned int quality = 0;

  if (!isTMB07) {
    bool isDistrip = (cLCT.getStripType() == 0);

    if (aLCT.isValid() && !(cLCT.isValid())) {    // no CLCT
      if (aLCT.getAccelerator()) {quality =  1;}
      else                       {quality =  3;}
    }
    else if (!(aLCT.isValid()) && cLCT.isValid()) { // no ALCT
      if (isDistrip)             {quality =  4;}
      else                       {quality =  5;}
    }
    else if (aLCT.isValid() && cLCT.isValid()) { // both ALCT and CLCT
      if (aLCT.getAccelerator()) {quality =  2;} // accelerator muon
      else {                                     // collision muon
        // CLCT quality is, in fact, the number of layers hit, so subtract 3
        // to get quality analogous to ALCT one.
        int sumQual = aLCT.getQuality() + (cLCT.getQuality()-3);
        if (sumQual < 1 || sumQual > 6) {
          if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongValues")
            << "+++ findQuality: sumQual = " << sumQual << "+++ \n";
        }
        if (isDistrip) { // distrip pattern
          if (sumQual == 2)      {quality =  6;}
          else if (sumQual == 3) {quality =  7;}
          else if (sumQual == 4) {quality =  8;}
          else if (sumQual == 5) {quality =  9;}
          else if (sumQual == 6) {quality = 10;}
        }
        else {            // halfstrip pattern
          if (sumQual == 2)      {quality = 11;}
          else if (sumQual == 3) {quality = 12;}
          else if (sumQual == 4) {quality = 13;}
          else if (sumQual == 5) {quality = 14;}
          else if (sumQual == 6) {quality = 15;}
        }
      }
    }
  }
#ifdef OLD
  else {
    // Temporary definition, used until July 2008.
    // First if statement is fictitious, just to help the CSC TF emulator
    // handle such cases (one needs to make sure they will be accounted for
    // in the new quality definition.
    if (!(aLCT.isValid()) || !(cLCT.isValid())) {
      if (aLCT.isValid() && !(cLCT.isValid()))      quality = 1; // no CLCT
      else if (!(aLCT.isValid()) && cLCT.isValid()) quality = 2; // no ALCT
      else quality = 0; // both absent; should never happen.
    }
    else {
      // Sum of ALCT and CLCT quality bits.  CLCT quality is, in fact, the
      // number of layers hit, so subtract 3 to put it to the same footing as
      // the ALCT quality.
      int sumQual = aLCT.getQuality() + (cLCT.getQuality()-3);
      if (sumQual < 1 || sumQual > 6) {
        if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongValues")
          << "+++ findQuality: Unexpected sumQual = " << sumQual << "+++\n";
      }

      // LCT quality is basically the sum of ALCT and CLCT qualities, but split
      // in two groups depending on the CLCT pattern id (higher quality for
      // straighter patterns).
      int offset = 0;
      if (cLCT.getPattern() <= 7) offset = 4;
      else                        offset = 9;
      quality = offset + sumQual;
    }
  }
#endif
  else {
    // 2008 definition.
    if (!(aLCT.isValid()) || !(cLCT.isValid())) {
      if (aLCT.isValid() && !(cLCT.isValid()))      quality = 1; // no CLCT
      else if (!(aLCT.isValid()) && cLCT.isValid()) quality = 2; // no ALCT
      else quality = 0; // both absent; should never happen.
    }
    else {
      int pattern = cLCT.getPattern();
      if (pattern == 1) quality = 3; // layer-trigger in CLCT
      else {
        // CLCT quality is the number of layers hit minus 3.
        // CLCT quality is the number of layers hit.
	//	const int n_gem((pad!=NULL and 1) or (copad!=NULL and 2));
	int n_gem = 0;  
	if (hasPad) n_gem = 1;
	if (hasCoPad) n_gem = 2;
        const bool a4(aLCT.getQuality() >= 1);
	const bool c4((cLCT.getQuality() >= 4) or (cLCT.getQuality() >= 3 and n_gem>=1));
        //              quality = 4; "reserved for low-quality muons in future"
        if      (!a4 && !c4) quality = 5; // marginal anode and cathode
        else if ( a4 && !c4) quality = 6; // HQ anode, but marginal cathode
        else if (!a4 &&  c4) quality = 7; // HQ cathode, but marginal anode
        else if ( a4 &&  c4) {
          if (aLCT.getAccelerator()) quality = 8; // HQ muon, but accel ALCT
          else {
            // quality =  9; "reserved for HQ muons with future patterns
            // quality = 10; "reserved for HQ muons with future patterns
            if (pattern == 2 || pattern == 3)      quality = 11;
            else if (pattern == 4 || pattern == 5) quality = 12;
            else if (pattern == 6 || pattern == 7) quality = 13;
            else if (pattern == 8 || pattern == 9) quality = 14;
            else if (pattern == 10)                quality = 15;
            else {
              if (infoV >= 0) edm::LogWarning("L1CSCTPEmulatorWrongValues")
                << "+++ findQuality: Unexpected CLCT pattern id = "
                << pattern << "+++\n";
            }
          }
        }
      }
    }
  }
  return quality;
}


unsigned int CSCMotherboardME11GEM::findQualityGEM(const CSCCLCTDigi& cLCT, const GEMPadDigi& gem)
{
  return 0;
}

 
void CSCMotherboardME11GEM::printGEMTriggerPads(int bx_start, int bx_stop, bool iscopad)
{
  // pads or copads?
  auto thePads(!iscopad ? pads_ : coPads_); 
  const bool hasPads(thePads.size()!=0);
  
  std::cout << "------------------------------------------------------------------------" << std::endl;
  bool first = true;
  for (int bx = bx_start; bx <= bx_stop; bx++) {
    // print only the pads for the central BX
    if (bx!=lct_central_bx and iscopad) continue;
    std::vector<std::pair<unsigned int, const GEMPadDigi*> > in_pads = thePads[bx];
    if (first) {
      if (!iscopad) std::cout << "* GEM trigger pads: " << std::endl;
      else          std::cout << "* GEM trigger coincidence pads: " << std::endl;
    }
    first = false;
    if (!iscopad) std::cout << "N(pads) BX " << bx << " : " << in_pads.size() << std::endl;
    else          std::cout << "N(copads) BX " << bx << " : " << in_pads.size() << std::endl;
    if (hasPads){
      for (auto pad : in_pads){
        auto roll_id(GEMDetId(pad.first));
        std::cout << "\tdetId " << pad.first << " " << roll_id << ", pad = " << pad.second->pad() << ", BX = " << pad.second->bx() + 6;
        if (isPadInOverlap(roll_id.roll())) std::cout << " (in overlap)" << std::endl;
        else std::cout << std::endl;
      }
    }
    else
      break;
  }
}

void CSCMotherboardME11GEM::retrieveGEMPads(const GEMPadDigiCollection* gemPads, unsigned id)
{
  auto superChamber(gem_g->superChamber(id));
  for (auto ch : superChamber->chambers()) {
    for (auto roll : ch->etaPartitions()) {
      GEMDetId roll_id(roll->id());
      auto pads_in_det = gemPads->get(roll_id);
      for (auto pad = pads_in_det.first; pad != pads_in_det.second; ++pad) {
        auto id_pad = std::make_pair(roll_id, &(*pad));
        const int bx_shifted(lct_central_bx + pad->bx());
        for (int bx = bx_shifted - maxDeltaBXPad_;bx <= bx_shifted + maxDeltaBXPad_; ++bx) {
	  pads_[bx].push_back(id_pad);  
        }
      }
    }
  }
}

void CSCMotherboardME11GEM::retrieveGEMCoPads(const GEMCoPadDigiCollection* gemPads, unsigned id)
{
  auto superChamber(gem_g->superChamber(id));
  for (auto ch : superChamber->chambers()) {
    for (auto roll : ch->etaPartitions()) {
      GEMDetId roll_id(roll->id());
      auto pads_in_det = gemPads->get(roll_id);
      for (auto pad = pads_in_det.first; pad != pads_in_det.second; ++pad) {
	GEMPadDigi gfirst((*pad).first());
	auto id_pad = std::make_pair(roll_id, &gfirst);
	const int bx_shifted(lct_central_bx + (pad->first()).bx());
	for (int bx = bx_shifted - maxDeltaBXPad_;bx <= bx_shifted + maxDeltaBXPad_; ++bx) {
	  if(bx != lct_central_bx) continue;
	  coPads_[bx].push_back(id_pad);  
	}
      }
    }
  }
}

bool CSCMotherboardME11GEM::isPadInOverlap(int roll)
{
  for (auto& p : cscWgToGemRoll_) {
    // overlap region are WGs 10-15
    if ((p.first < 10) or (p.first > 15)) continue;
    if (((p.second).first <= roll) and (roll <= (p.second).second)) return true;
  }
  return false;
}


int CSCMotherboardME11GEM::deltaRoll(int wg, int pad)
{
  const auto p(cscWgToGemRoll_[wg]);
  return std::min(std::abs(p.first - pad), std::abs(p.second - pad));
}


int CSCMotherboardME11GEM::deltaPad(int hs, int pad)
{
  const auto p(cscHsToGemPadME1b_[hs]);
  return std::min(std::abs(p.first - pad), std::abs(p.second - pad));
}


CSCMotherboardME11GEM::GEMPadsBX  
CSCMotherboardME11GEM::matchingGEMPads(const CSCCLCTDigi& clct, const GEMPadsBX& pads, enum ME11Part part, bool isCoPad, bool first)
{
  CSCMotherboardME11GEM::GEMPadsBX result;

  // fetch the low and high pad edges
  auto mymap(part==ME1A ? cscHsToGemPadME1a_ : cscHsToGemPadME1b_);
  int deltaPad(isCoPad ? maxDeltaPadCoPad_ : maxDeltaPadPad_);
  const int lowPad(mymap[clct.getKeyStrip()].first);
  const int highPad(mymap[clct.getKeyStrip()].second);
  const bool debug(false);
  if (debug) std::cout << "lowpad " << lowPad << " highpad " << highPad << " delta pad " << deltaPad <<std::endl;
  for (auto p: pads){
    auto padRoll((p.second)->pad());
    if (debug) std::cout << "padRoll " << padRoll << std::endl;
    if (std::abs(lowPad - padRoll) <= deltaPad or std::abs(padRoll - highPad) <= deltaPad){
    if (debug) std::cout << "++Matches! " << std::endl;
      result.push_back(p);
      if (first) return result;
    }
  }
  return result;
}


CSCMotherboardME11GEM::GEMPadsBX 
CSCMotherboardME11GEM::matchingGEMPads(const CSCALCTDigi& alct, const GEMPadsBX& pads, enum ME11Part part, bool isCoPad, bool first)
{
  CSCMotherboardME11GEM::GEMPadsBX result;
  
  auto alctRoll(cscWgToGemRoll_[alct.getKeyWG()]);
  const bool debug(false);
  if (debug) std::cout << "ALCT keyWG " << alct.getKeyWG() << ", rolls " << alctRoll.first << " " << alctRoll.second << std::endl;
  for (auto p: pads){
    auto padRoll(GEMDetId(p.first).roll());
    if (debug) std::cout << "Candidate ALCT: " << padRoll << std::endl;
    // only pads in overlap are good for ME1A
    if (part==ME1A and !isPadInOverlap(padRoll)) continue;
    if (alctRoll.first == -99 and alctRoll.second == -99) continue;  //invalid region
    else if (alctRoll.first == -99 and !(padRoll <= alctRoll.second)) continue; // top of the chamber
    else if (alctRoll.second == -99 and !(padRoll >= alctRoll.first)) continue; // bottom of the chamber
    else if ((alctRoll.first != -99 and alctRoll.second != -99) and // center
             (alctRoll.first > padRoll or padRoll > alctRoll.second)) continue;
    if (debug) std::cout << "++Matches! " << std::endl;
    result.push_back(p);
    if (first) return result;
  }
  return result;
}


CSCMotherboardME11GEM::GEMPadsBX 
CSCMotherboardME11GEM::matchingGEMPads(const CSCCLCTDigi& clct, const CSCALCTDigi& alct, const GEMPadsBX& pads, 
                                    enum ME11Part part, bool isCoPad, bool first)
{
  CSCMotherboardME11GEM::GEMPadsBX result;

  // Fetch all (!) pads matching to ALCTs and CLCTs
  auto padsClct(matchingGEMPads(clct, pads, part, isCoPad, false));
  auto padsAlct(matchingGEMPads(alct, pads, part, isCoPad, false));

  const bool debug(false);
  if (debug) std::cout << "-----------------------------------------------------------------------"<<std::endl;
  // Check if the pads overlap
  for (auto p : padsAlct){
    if (debug) std::cout<< "Candidate ALCT: " << p.first << " " << *(p.second) << std::endl;
    for (auto q: padsClct){
      if (debug) std::cout<< "++Candidate CLCT: " << q.first << " " << *(q.second) << std::endl;
      // look for exactly the same pads
      if ((p.first != q.first) or GEMPadDigi(*(p.second)) != *(q.second)) continue;
      if (debug) std::cout << "++Matches! " << std::endl;
      result.push_back(p);
      if (first) return result;
    }
  }
  if (debug) std::cout << "-----------------------------------------------------------------------"<<std::endl;
  return result;
}


std::vector<GEMCoPadDigi> CSCMotherboardME11GEM::readoutCoPads()
{
  return gemCoPadV;
}

