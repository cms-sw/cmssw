#include <L1Trigger/TrackFindingTMTT/interface/Settings.h>
#include "FWCore/Utilities/interface/Exception.h"
#include <set>
#include <cmath>

namespace TMTT {

// Set config params for HYBRID TRACKING via hard-wired consts to allow use outside CMSSW.

Settings::Settings() {
  //
  // TMTT related configuration parameters, including Kalman Filter.
  // Meaning of these parameters explained in TrackFindingTMTT/python/TMTrackProducer_Defaults_cfi.py
  //
  reduceLayerID_=true;
  useLayerID_=true;
  minFracMatchStubsOnReco_=-99;
  minFracMatchStubsOnTP_=-99;
  minNumMatchLayers_=4;
  minNumMatchPSLayers_=0;
  stubMatchStrict_=false;
  minStubLayers_=4;
  minPtToReduceLayers_=99999.;
  deadReduceLayers_=false;
  kalmanMinNumStubs_=4;
  kalmanMaxNumStubs_=6;
  numPhiNonants_=9;
  numPhiSectors_=9;
  etaRegions_ = {-2.4,-2.08,-1.68,-1.26,-0.90,-0.62,-0.41,-0.20,0.0,0.20,0.41,0.62,0.90,1.26,1.68,2.08,2.4}; // Used by KF
  kalmanRemove2PScut_=true;
  killScenario_=0;
  kalmanMaxSkipLayersHard_=1; // On "hard" input tracks
  kalmanMaxSkipLayersEasy_=2; // On "easy" input tracks 
  kalmanMaxStubsEasy_=10;  // Max. #stubs an input track can have to be defined "easy"
  kalmanMaxStubsPerLayer_=4; // To save resources, consider at most this many stubs per layer per track.
  kalmanDebugLevel_=0;
  //kalmanDebugLevel_=2; // Good for debugging
  enableDigitize_=false;
  houghMinPt_=2.0;
  chosenRofPhi_=67.240;
  chosenRofZ_=50.0;
  houghNbinsPt_=48; // Mini HT bins in 2 GeV HT array
  handleStripsPhiSec_=1;
  useApproxB_=true;
  kalmanHOtilted_=true; 
  kalmanHOhelixExp_=true;
  kalmanHOalpha_=1;
  kalmanHOdodgy_=false;
  kalmanHOprojZcorr_=1;
  bApprox_gradient_=0.886454;
  bApprox_intercept_=0.504148;
  handleStripsEtaSec_=false;
  kalmanFillInternalHists_=false;
  kalmanMultiScattTerm_=0.00075;
  kalmanChi2RphiScale_ = 8;
  //
  // Cfg params & constants required only for HYBRID tracking (as taken from DB for TMTT).
  //
  hybrid_=true;
  psStripPitch_=0.01;
  psNStrips_=960;
  psPixelLength_=0.1467;
  ssStripPitch_=0.009;
  ssNStrips_=1016;
  ssStripLength_=5.0250;
  zMaxNonTilted_[1] = 15.3; // max z at which non-tilted modules are found in inner 3 barrel layers.
  zMaxNonTilted_[2] = 24.6; 
  zMaxNonTilted_[3] = 33.9; 

  bField_=3.81120228767395;

  // Stub digitization params for hybrid (copied from TrackFindingTMTT/interface/HLS/KFconstants.h
  double rMult_hybrid   = 1. / 0.02929688;
  double phiSMult_hybrid = 1. / (7.828293e-6 * 8);
  double zMult_hybrid = rMult_hybrid / 2; // In KF VHDL, z/r mult = 1/2, whereas in HLS, they are identical.
  // Number of bits copied from TrackFindingTMTT/interface/HLS/KFstub.h (BR1, BPHI, BZ)
  rtBits_   = 12;
  phiSBits_ = 14;
  zBits_    = 14;
  rtRange_   = pow(2,rtBits_)/rMult_hybrid;
  phiSRange_ = pow(2,phiSBits_)/phiSMult_hybrid;
  zRange_    = pow(2,zBits_)/zMult_hybrid;

  if (hybrid_) {
    if (not useApproxB_) {
      std::cout<<"TMTT Settings Error: module tilt angle unknown, so must set useApproxB = true"<<std::endl;
      exit(1);
    }
  }
}

///=== Get configuration parameters from python cfg for TMTT tracking.

Settings::Settings(const edm::ParameterSet& iConfig) :

  // See either Analyze_Defaults_cfi.py or Settings.h for description of these parameters.

  //=== Parameter sets for differents types of configuration parameter.
  genCuts_                ( iConfig.getParameter< edm::ParameterSet >         ( "GenCuts"                ) ),
  stubCuts_               ( iConfig.getParameter< edm::ParameterSet >         ( "StubCuts"               ) ),
  stubDigitize_           ( iConfig.getParameter< edm::ParameterSet >         ( "StubDigitize"           ) ),
  geometricProc_          ( iConfig.getParameter< edm::ParameterSet >         ( "GeometricProc"          ) ),
  phiSectors_             ( iConfig.getParameter< edm::ParameterSet >         ( "PhiSectors"             ) ),
  etaSectors_             ( iConfig.getParameter< edm::ParameterSet >         ( "EtaSectors"             ) ),
  htArraySpecRphi_        ( iConfig.getParameter< edm::ParameterSet >         ( "HTArraySpecRphi"        ) ),
  htFillingRphi_          ( iConfig.getParameter< edm::ParameterSet >         ( "HTFillingRphi"          ) ),
  rzFilterOpts_           ( iConfig.getParameter< edm::ParameterSet >         ( "RZfilterOpts"           ) ),
  l1TrackDef_             ( iConfig.getParameter< edm::ParameterSet >         ( "L1TrackDef"             ) ),
  dupTrkRemoval_          ( iConfig.getParameter< edm::ParameterSet >         ( "DupTrkRemoval"          ) ),
  trackMatchDef_          ( iConfig.getParameter< edm::ParameterSet >         ( "TrackMatchDef"          ) ),
  trackFitSettings_       ( iConfig.getParameter< edm::ParameterSet >         ( "TrackFitSettings"       ) ),
  deadModuleOpts_         ( iConfig.getParameter< edm::ParameterSet >         ( "DeadModuleOpts"         ) ),
  trackDigi_              ( iConfig.getParameter< edm::ParameterSet >         ( "TrackDigi"              ) ),

  //=== General settings

  enableMCtruth_          ( iConfig.getParameter<bool>                        ( "EnableMCtruth"          ) ),
  enableHistos_           ( iConfig.getParameter<bool>                        ( "EnableHistos"           ) ),

  //=== Cuts on MC truth tracks used for tracking efficiency measurements.

  genMinPt_               ( genCuts_.getParameter<double>                     ( "GenMinPt"               ) ),
  genMaxAbsEta_           ( genCuts_.getParameter<double>                     ( "GenMaxAbsEta"           ) ),
  genMaxVertR_            ( genCuts_.getParameter<double>                     ( "GenMaxVertR"            ) ),
  genMaxVertZ_            ( genCuts_.getParameter<double>                     ( "GenMaxVertZ"            ) ),
  genMaxD0_               ( genCuts_.getParameter<double>                     ( "GenMaxD0"               ) ),
  genMaxZ0_               ( genCuts_.getParameter<double>                     ( "GenMaxZ0"               ) ),
  genMinStubLayers_       ( genCuts_.getParameter<unsigned int>               ( "GenMinStubLayers"       ) ),

  //=== Cuts applied to stubs before arriving in L1 track finding board.

  degradeBendRes_         ( stubCuts_.getParameter<unsigned int>              ( "DegradeBendRes"         ) ),
  maxStubEta_             ( stubCuts_.getParameter<double>                    ( "MaxStubEta"             ) ),
  killLowPtStubs_         ( stubCuts_.getParameter<bool>                      ( "KillLowPtStubs"         ) ),
  printStubWindows_       ( stubCuts_.getParameter<bool>                      ( "PrintStubWindows"       ) ),
  bendResolution_         ( stubCuts_.getParameter<double>                    ( "BendResolution"         ) ),
  bendResolutionExtra_    ( stubCuts_.getParameter<double>                    ( "BendResolutionExtra"    ) ),
  orderStubsByBend_       ( stubCuts_.getParameter<bool>                      ( "OrderStubsByBend"       ) ),

  //=== Optional stub digitization.

  enableDigitize_         ( stubDigitize_.getParameter<bool>                  ( "EnableDigitize"         ) ),

  //--- Parameters available in MP board.
  phiSectorBits_          ( stubDigitize_.getParameter<unsigned int>          ( "PhiSectorBits"          ) ),
  phiSBits_               ( stubDigitize_.getParameter<unsigned int>          ( "PhiSBits"               ) ),
  phiSRange_              ( stubDigitize_.getParameter<double>                ( "PhiSRange"              ) ),
  rtBits_                 ( stubDigitize_.getParameter<unsigned int>          ( "RtBits"                 ) ),
  rtRange_                ( stubDigitize_.getParameter<double>                ( "RtRange"                ) ),
  zBits_                  ( stubDigitize_.getParameter<unsigned int>          ( "ZBits"                  ) ),
  zRange_                 ( stubDigitize_.getParameter<double>                ( "ZRange"                 ) ),
  //--- Parameters available in GP board (excluding any in common with MP specified above).
  phiOBits_               ( stubDigitize_.getParameter<unsigned int>          ( "PhiOBits"               ) ),
  phiORange_              ( stubDigitize_.getParameter<double>                ( "PhiORange"              ) ),
  bendBits_               ( stubDigitize_.getParameter<unsigned int>          ( "BendBits"               ) ),

  //=== Configuration of Geometric Processor.
  useApproxB_             ( geometricProc_.getParameter<bool>                 ( "UseApproxB"             ) ),
  bApprox_gradient_       ( geometricProc_.getParameter<double>               ( "BApprox_gradient"       ) ),
  bApprox_intercept_      ( geometricProc_.getParameter<double>               ( "BApprox_intercept"      ) ),

  //=== Division of Tracker into phi sectors.
  numPhiNonants_          ( phiSectors_.getParameter<unsigned int>            ( "NumPhiNonants"          ) ),
  numPhiSectors_          ( phiSectors_.getParameter<unsigned int>            ( "NumPhiSectors"          ) ),
  chosenRofPhi_           ( phiSectors_.getParameter<double>                  ( "ChosenRofPhi"           ) ),
  useStubPhi_             ( phiSectors_.getParameter<bool>                    ( "UseStubPhi"             ) ), 
  useStubPhiTrk_          ( phiSectors_.getParameter<bool>                    ( "UseStubPhiTrk"          ) ), 
  assumedPhiTrkRes_       ( phiSectors_.getParameter<double>                  ( "AssumedPhiTrkRes"       ) ),
  calcPhiTrkRes_          ( phiSectors_.getParameter<bool>                    ( "CalcPhiTrkRes"          ) ), 
  handleStripsPhiSec_     ( phiSectors_.getParameter<bool>                    ( "HandleStripsPhiSec"     ) ),

  //=== Division of Tracker into eta sectors.
  etaRegions_             ( etaSectors_.getParameter<vector<double> >         ( "EtaRegions"             ) ),
  chosenRofZ_             ( etaSectors_.getParameter<double>                  ( "ChosenRofZ"             ) ),
  beamWindowZ_            ( etaSectors_.getParameter<double>                  ( "BeamWindowZ"            ) ),   
  handleStripsEtaSec_     ( etaSectors_.getParameter<bool>                    ( "HandleStripsEtaSec"     ) ),
  allowOver2EtaSecs_      ( etaSectors_.getParameter<bool>                    ( "AllowOver2EtaSecs"      ) ),
                               
  //=== r-phi Hough transform array specifications.
  houghMinPt_             ( htArraySpecRphi_.getParameter<double>             ( "HoughMinPt"             ) ),
  houghNbinsPt_           ( htArraySpecRphi_.getParameter<unsigned int>       ( "HoughNbinsPt"           ) ),
  houghNbinsPhi_          ( htArraySpecRphi_.getParameter<unsigned int>       ( "HoughNbinsPhi"          ) ),
  houghNcellsRphi_        ( htArraySpecRphi_.getParameter<int>                ( "HoughNcellsRphi"        ) ),
  enableMerge2x2_         ( htArraySpecRphi_.getParameter<bool>               ( "EnableMerge2x2"         ) ),
  maxPtToMerge2x2_        ( htArraySpecRphi_.getParameter<double>             ( "MaxPtToMerge2x2"        ) ),
  numSubSecsEta_          ( htArraySpecRphi_.getParameter<unsigned int>       ( "NumSubSecsEta"          ) ),
  shape_                  ( htArraySpecRphi_.getParameter<unsigned int>       ( "Shape"                  ) ),
  miniHTstage_            ( htArraySpecRphi_.getParameter<bool>               ( "MiniHTstage"            ) ),
  miniHoughNbinsPt_       ( htArraySpecRphi_.getParameter<unsigned int>       ( "MiniHoughNbinsPt"       ) ),
  miniHoughNbinsPhi_      ( htArraySpecRphi_.getParameter<unsigned int>       ( "MiniHoughNbinsPhi"      ) ),
  miniHoughMinPt_         ( htArraySpecRphi_.getParameter<double>             ( "MiniHoughMinPt"         ) ),
  miniHoughDontKill_      ( htArraySpecRphi_.getParameter<bool>               ( "MiniHoughDontKill"      ) ),
  miniHoughDontKillMinPt_ ( htArraySpecRphi_.getParameter<double>             ( "MiniHoughDontKillMinPt" ) ),
  miniHoughLoadBalance_   ( htArraySpecRphi_.getParameter<unsigned int>       ( "MiniHoughLoadBalance"   ) ),
                                
  //=== Rules governing how stubs are filled into the r-phi Hough Transform array.
  handleStripsRphiHT_     ( htFillingRphi_.getParameter<bool>                 ( "HandleStripsRphiHT"     ) ),
  killSomeHTCellsRphi_    ( htFillingRphi_.getParameter<unsigned int>         ( "KillSomeHTCellsRphi"    ) ),
  useBendFilter_          ( htFillingRphi_.getParameter<bool>                 ( "UseBendFilter"          ) ), 
  maxStubsInCell_         ( htFillingRphi_.getParameter<unsigned int>         ( "MaxStubsInCell"         ) ),
  maxStubsInCellMiniHough_( htFillingRphi_.getParameter<unsigned int>         ( "MaxStubsInCellMiniHough") ),
  busySectorKill_         ( htFillingRphi_.getParameter<bool>                 ( "BusySectorKill"         ) ),
  busySectorNumStubs_     ( htFillingRphi_.getParameter<unsigned int>         ( "BusySectorNumStubs"     ) ),
  busySectorMbinRanges_   ( htFillingRphi_.getParameter<vector<unsigned int>> ( "BusySectorMbinRanges"   ) ),
  busySectorMbinOrder_    ( htFillingRphi_.getParameter<vector<unsigned int>> ( "BusySectorMbinOrder"    ) ),
  busyInputSectorKill_    ( htFillingRphi_.getParameter<bool>                 ( "BusyInputSectorKill"    ) ),
  busyInputSectorNumStubs_( htFillingRphi_.getParameter<unsigned int>         ( "BusyInputSectorNumStubs") ),
  muxOutputsHT_           ( htFillingRphi_.getParameter<unsigned int>         ( "MuxOutputsHT"           ) ),
  etaRegWhitelist_        ( htFillingRphi_.getParameter< vector<unsigned int> > ( "EtaRegWhitelist"      ) ),

  //=== Options controlling r-z track filters (or any other track filters run after the Hough transform, as opposed to inside it).  

  rzFilterName_           ( rzFilterOpts_.getParameter<string>                ( "RZFilterName"           ) ), 
  seedResolution_         ( rzFilterOpts_.getParameter<double>                ( "SeedResolution"         ) ),
  keepAllSeed_            ( rzFilterOpts_.getParameter<bool>                  ( "KeepAllSeed"            ) ), 
  maxSeedCombinations_    ( rzFilterOpts_.getParameter<unsigned int>          ( "MaxSeedCombinations"    ) ),
  maxGoodSeedCombinations_( rzFilterOpts_.getParameter<unsigned int>          ( "MaxGoodSeedCombinations") ),
  maxSeedsPerStub_        ( rzFilterOpts_.getParameter<unsigned int>          ( "MaxSeedsPerStub"        ) ),
  zTrkSectorCheck_        ( rzFilterOpts_.getParameter<bool>                  ( "zTrkSectorCheck"        ) ),
  minFilterLayers_        ( rzFilterOpts_.getParameter<unsigned int>          ( "MinFilterLayers"        ) ),

  //=== Rules for deciding when the track finding has found an L1 track candidate

  minStubLayers_          ( l1TrackDef_.getParameter<unsigned int>            ( "MinStubLayers"          ) ),
  minPtToReduceLayers_    ( l1TrackDef_.getParameter<double>                  ( "MinPtToReduceLayers"    ) ),
  etaSecsReduceLayers_    ( l1TrackDef_.getParameter<vector<unsigned int>>    ( "EtaSecsReduceLayers"    ) ),
  useLayerID_             ( l1TrackDef_.getParameter<bool>                    ( "UseLayerID"             ) ),
  reduceLayerID_          ( l1TrackDef_.getParameter<bool>                    ( "ReducedLayerID"         ) ),

  //=== Specification of algorithm to eliminate duplicate tracks.

  dupTrkAlgRphi_          ( dupTrkRemoval_.getParameter<unsigned int>         ( "DupTrkAlgRphi"          ) ),
  dupTrkAlg3D_            ( dupTrkRemoval_.getParameter<unsigned int>         ( "DupTrkAlg3D"            ) ),
  dupTrkAlgFit_           ( dupTrkRemoval_.getParameter<unsigned int>         ( "DupTrkAlgFit"           ) ),
  dupTrkMinCommonHitsLayers_ ( dupTrkRemoval_.getParameter<unsigned int>      ( "DupTrkMinCommonHitsLayers" ) ),

  //=== Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle).

  minFracMatchStubsOnReco_( trackMatchDef_.getParameter<double>               ( "MinFracMatchStubsOnReco") ),
  minFracMatchStubsOnTP_  ( trackMatchDef_.getParameter<double>               ( "MinFracMatchStubsOnTP"  ) ),
  minNumMatchLayers_      ( trackMatchDef_.getParameter<unsigned int>         ( "MinNumMatchLayers"      ) ),
  minNumMatchPSLayers_    ( trackMatchDef_.getParameter<unsigned int>         ( "MinNumMatchPSLayers"    ) ),
  stubMatchStrict_        ( trackMatchDef_.getParameter<bool>                 ( "StubMatchStrict"        ) ),

  //=== Track Fitting Settings

  trackFitters_           ( trackFitSettings_.getParameter<vector<std::string>> ( "TrackFitters"         ) ),
  useRZfilter_            ( trackFitSettings_.getParameter<vector<std::string>> ( "UseRZfilter"          ) ),
  detailedFitOutput_      ( trackFitSettings_.getParameter < bool >           ( "DetailedFitOutput"      ) ),
  trackFitCheat_          ( trackFitSettings_.getParameter < bool >           ( "TrackFitCheat"          ) ),
  //
  numTrackFitIterations_  ( trackFitSettings_.getParameter<unsigned int>      ( "NumTrackFitIterations"  ) ),
  killTrackFitWorstHit_   ( trackFitSettings_.getParameter <bool>             ( "KillTrackFitWorstHit"   ) ),
  generalResidualCut_     ( trackFitSettings_.getParameter<double>            ( "GeneralResidualCut"     ) ),
  killingResidualCut_     ( trackFitSettings_.getParameter<double>            ( "KillingResidualCut"     ) ),
  maxIterationsLR_        ( trackFitSettings_.getParameter<unsigned int>      ( "MaxIterationsLR"        ) ),
  LRFillInternalHists_    ( trackFitSettings_.getParameter<bool>              ( "LRFillInternalHists"    ) ),
  combineResiduals_       ( trackFitSettings_.getParameter< bool >            ( "CombineResiduals"       ) ),
  lineariseStubPosition_  ( trackFitSettings_.getParameter< bool >            ( "LineariseStubPosition"  ) ),
  checkSectorConsistency_ ( trackFitSettings_.getParameter< bool >            ( "CheckSectorConsistency" ) ),
  checkHTCellConsistency_ ( trackFitSettings_.getParameter< bool >            ( "CheckHTCellConsistency" ) ),
  minPSLayers_            ( trackFitSettings_.getParameter< unsigned int >    ( "MinPSLayers"            ) ),
  digitizeLR_             ( trackFitSettings_.getParameter< bool >            ( "DigitizeLR"             ) ),
  PhiPrecision_           ( trackFitSettings_.getParameter< double >          ( "PhiPrecision"           ) ),
  RPrecision_             ( trackFitSettings_.getParameter< double >          ( "RPrecision"             ) ),
  ZPrecision_             ( trackFitSettings_.getParameter< double >          ( "ZPrecision"             ) ),
  ZSlopeWidth_            ( trackFitSettings_.getParameter< unsigned int >    ( "ZSlopeWidth"            ) ),
  ZInterceptWidth_        ( trackFitSettings_.getParameter< unsigned int >    ( "ZInterceptWidth"        ) ),
  //
  digitizeSLR_            ( trackFitSettings_.getParameter<bool>              ( "DigitizeSLR"            ) ),
  dividerBitsHelix_       ( trackFitSettings_.getParameter<unsigned int>      ( "DividerBitsHelix"       ) ),
  dividerBitsHelixZ_      ( trackFitSettings_.getParameter<unsigned int>      ( "DividerBitsHelixZ"      ) ),
  ShiftingBitsDenRPhi_           ( trackFitSettings_.getParameter<unsigned int>      ( "ShiftingBitsDenRPhi"           ) ), 
  ShiftingBitsDenRZ_      ( trackFitSettings_.getParameter<unsigned int>      ( "ShiftingBitsDenRZ"      ) ), 
  ShiftingBitsPt_         ( trackFitSettings_.getParameter<unsigned int>      ( "ShiftingBitsPt"         ) ),    
  ShiftingBitsPhi_        ( trackFitSettings_.getParameter<unsigned int>      ( "ShiftingBitsPhi"        ) ),    

  ShiftingBitsLambda_     ( trackFitSettings_.getParameter<unsigned int>      ( "ShiftingBitsLambda"     ) ),
  ShiftingBitsZ0_         ( trackFitSettings_.getParameter<unsigned int>      ( "ShiftingBitsZ0"         ) ),
  slr_chi2cut_            ( trackFitSettings_.getParameter<double>            ( "SLR_chi2cut"            ) ),   
  residualCut_            ( trackFitSettings_.getParameter<double>            ( "ResidualCut"            ) ),       
  //
  kalmanDebugLevel_        ( trackFitSettings_.getParameter<unsigned int>     ( "KalmanDebugLevel"       ) ),
  kalmanFillInternalHists_ ( trackFitSettings_.getParameter<bool>             ( "KalmanFillInternalHists") ),
  kalmanMinNumStubs_       ( trackFitSettings_.getParameter<unsigned int>     ( "KalmanMinNumStubs"      ) ),
  kalmanMaxNumStubs_       ( trackFitSettings_.getParameter<unsigned int>     ( "KalmanMaxNumStubs"      ) ),
  kalmanAddBeamConstr_     ( trackFitSettings_.getParameter<bool>             ( "KalmanAddBeamConstr"    ) ),
  kalmanRemove2PScut_      ( trackFitSettings_.getParameter<bool>             ( "KalmanRemove2PScut"     ) ),
  kalmanMaxSkipLayersHard_ ( trackFitSettings_.getParameter<unsigned>         ( "KalmanMaxSkipLayersHard") ),
  kalmanMaxSkipLayersEasy_ ( trackFitSettings_.getParameter<unsigned>         ( "KalmanMaxSkipLayersEasy") ),
  kalmanMaxStubsEasy_      ( trackFitSettings_.getParameter<unsigned>         ( "KalmanMaxStubsEasy"     ) ),
  kalmanMaxStubsPerLayer_  ( trackFitSettings_.getParameter<unsigned>         ( "KalmanMaxStubsPerLayer" ) ),
  kalmanMultiScattTerm_    ( trackFitSettings_.getParameter<double>           ( "KalmanMultiScattTerm"   ) ),
  kalmanChi2RphiScale_     ( trackFitSettings_.getParameter<unsigned>         ( "KalmanChi2RphiScale"    ) ),
  kalmanHOtilted_          ( trackFitSettings_.getParameter<bool>             ( "KalmanHOtilted"         ) ),
  kalmanHOhelixExp_        ( trackFitSettings_.getParameter<bool>             ( "KalmanHOhelixExp"       ) ),
  kalmanHOalpha_           ( trackFitSettings_.getParameter<unsigned int>     ( "KalmanHOalpha"          ) ),
  kalmanHOprojZcorr_       ( trackFitSettings_.getParameter<unsigned int>     ( "KalmanHOprojZcorr"      ) ),
  kalmanHOdodgy_           ( trackFitSettings_.getParameter<bool>             ( "KalmanHOdodgy"          ) ),
 
  //=== Treatment of dead modules.

  deadReduceLayers_       (deadModuleOpts_.getParameter<bool>                 ( "DeadReduceLayers"       ) ),
  deadSimulateFrac_       (deadModuleOpts_.getParameter<double>               ( "DeadSimulateFrac"       ) ),
  killScenario_           (deadModuleOpts_.getParameter<unsigned int>         ( "KillScenario"           ) ),
  killRecover_            (deadModuleOpts_.getParameter<bool>                 ( "KillRecover"            ) ),

  //=== Track digitisation configuration for various track fitters

  slr_skipTrackDigi_      (trackDigi_.getParameter<bool>                      ( "SLR_skipTrackDigi"      ) ),
  slr_oneOver2rBits_      (trackDigi_.getParameter<unsigned int>              ( "SLR_oneOver2rBits"      ) ),
  slr_oneOver2rRange_     (trackDigi_.getParameter<double>                    ( "SLR_oneOver2rRange"     ) ),
  slr_d0Bits_             (trackDigi_.getParameter<unsigned int>              ( "SLR_d0Bits"             ) ),
  slr_d0Range_            (trackDigi_.getParameter<double>                    ( "SLR_d0Range"            ) ),
  slr_phi0Bits_           (trackDigi_.getParameter<unsigned int>              ( "SLR_phi0Bits"           ) ),
  slr_phi0Range_          (trackDigi_.getParameter<double>                    ( "SLR_phi0Range"          ) ),
  slr_z0Bits_             (trackDigi_.getParameter<unsigned int>              ( "SLR_z0Bits"             ) ),
  slr_z0Range_            (trackDigi_.getParameter<double>                    ( "SLR_z0Range"            ) ),
  slr_tanlambdaBits_      (trackDigi_.getParameter<unsigned int>              ( "SLR_tanlambdaBits"      ) ),
  slr_tanlambdaRange_     (trackDigi_.getParameter<double>                    ( "SLR_tanlambdaRange"     ) ),
  slr_chisquaredBits_     (trackDigi_.getParameter<unsigned int>              ( "SLR_chisquaredBits"     ) ),
  slr_chisquaredRange_    (trackDigi_.getParameter<double>                    ( "SLR_chisquaredRange"    ) ),
  //
  kf_skipTrackDigi_       (trackDigi_.getParameter<bool>                      ( "KF_skipTrackDigi"       ) ),
  kf_oneOver2rBits_       (trackDigi_.getParameter<unsigned int>              ( "KF_oneOver2rBits"       ) ),
  kf_oneOver2rRange_      (trackDigi_.getParameter<double>                    ( "KF_oneOver2rRange"      ) ),
  kf_d0Bits_              (trackDigi_.getParameter<unsigned int>              ( "KF_d0Bits"              ) ),
  kf_d0Range_             (trackDigi_.getParameter<double>                    ( "KF_d0Range"             ) ),
  kf_phi0Bits_            (trackDigi_.getParameter<unsigned int>              ( "KF_phi0Bits"            ) ),
  kf_phi0Range_           (trackDigi_.getParameter<double>                    ( "KF_phi0Range"           ) ),
  kf_z0Bits_              (trackDigi_.getParameter<unsigned int>              ( "KF_z0Bits"              ) ),
  kf_z0Range_             (trackDigi_.getParameter<double>                    ( "KF_z0Range"             ) ),
  kf_tanlambdaBits_       (trackDigi_.getParameter<unsigned int>              ( "KF_tanlambdaBits"       ) ),
  kf_tanlambdaRange_      (trackDigi_.getParameter<double>                    ( "KF_tanlambdaRange"      ) ),
  kf_chisquaredBits_      (trackDigi_.getParameter<unsigned int>              ( "KF_chisquaredBits"      ) ),
  kf_chisquaredRange_     (trackDigi_.getParameter<double>                    ( "KF_chisquaredRange"     ) ),
  kf_chisquaredBinEdges_  (trackDigi_.getParameter<vector<double> >           ( "KF_chisquaredBinEdges"  ) ),
  //
  other_skipTrackDigi_    (trackDigi_.getParameter<bool>                      ( "Other_skipTrackDigi"    ) ),

  // Debug printout
  debug_                  ( iConfig.getParameter<unsigned int>                ( "Debug"                  ) ),
  resPlotOpt_             ( iConfig.getParameter<bool>                        ( "ResPlotOpt"             ) ),
  iPhiPlot_               ( iConfig.getParameter<unsigned int>                ( "iPhiPlot"               ) ),
  iEtaPlot_               ( iConfig.getParameter<unsigned int>                ( "iEtaPlot"               ) ),

  // Name of output EDM file if any.
  // N.B. This parameter does not appear inside TMTrackProducer_Defaults_cfi.py . It is created inside
  // tmtt_tf_analysis_cfg.py .
  writeOutEdmFile_        ( iConfig.getUntrackedParameter<bool>               ( "WriteOutEdmFile", true) ),

  // Bfield in Tesla. (Unknown at job initiation. Set to true value for each event
  bField_                 (0.),

  // Hybrid tracking
  hybrid_                 ( iConfig.getParameter<bool>                        ( "Hybrid"                 ) ),
  psStripPitch_           (0.),
  psNStrips_              (0.),
  psPixelLength_          (0.),
  ssStripPitch_           (0.),
  ssNStrips_              (0.),
  ssStripLength_          (0.)
{
  // If user didn't specify any PDG codes, use e,mu,pi,K,p, to avoid picking up unstable particles like Xi-.
  vector<unsigned int> genPdgIdsUnsigned( genCuts_.getParameter<vector<unsigned int> >   ( "GenPdgIds" ) ); 
  if (genPdgIdsUnsigned.empty()) {
    genPdgIdsUnsigned = {11, 13, 211, 321, 2212};  
  }
   
  // For simplicity, user need not distinguish particles from antiparticles in configuration file.
  // But here we must store both explicitely in Settings, since TrackingParticleSelector expects them.
  for (unsigned int i = 0; i < genPdgIdsUnsigned.size(); i++) {
    genPdgIds_.push_back(  genPdgIdsUnsigned[i] );
    genPdgIds_.push_back( -genPdgIdsUnsigned[i] );
  }

  // Clean up list of fitters that require the r-z track filter to be run before them, 
  // by removing those fitters that are not to be run.
  vector<string> useRZfilterTmp;
  for (const string& name : useRZfilter_) {
    if (std::count(trackFitters_.begin(), trackFitters_.end(), name) > 0) useRZfilterTmp.push_back(name);
  }
  useRZfilter_ = useRZfilterTmp;

  //--- Sanity checks

  if ( ! (useStubPhi_ || useStubPhiTrk_) ) throw cms::Exception("Settings.cc: Invalid cfg parameters - You cant set both UseStubPhi & useStubPhiTrk to false.");

  if (minNumMatchLayers_ > minStubLayers_)    throw cms::Exception("Settings.cc: Invalid cfg parameters - You are setting the minimum number of layers incorrectly : type A.");
  if (genMinStubLayers_  > minStubLayers_)    throw cms::Exception("Settings.cc: Invalid cfg parameters - You are setting the minimum number of layers incorrectly : type B.");
  if (minNumMatchLayers_ > genMinStubLayers_) throw cms::Exception("Settings.cc: Invalid cfg parameters - You are setting the minimum number of layers incorrectly : type C.");

  // If reducing number of required layers for high Pt tracks, then above checks must be redone.
  bool doReduceLayers = (minPtToReduceLayers_ < 10000. || etaSecsReduceLayers_.size() > 0) ;
  if (doReduceLayers && minStubLayers_ > 4) {
    if (minNumMatchLayers_ > minStubLayers_ - 1) throw cms::Exception("Settings.cc: Invalid cfg parameters - You are setting the minimum number of layers incorrectly : type D.");
    if (genMinStubLayers_  > minStubLayers_ - 1) throw cms::Exception("Settings.cc: Invalid cfg parameters - You are setting the minimum number of layers incorrectly : type E.");
  }

  for (const unsigned int& iEtaReg : etaSecsReduceLayers_) {
    if (iEtaReg >= etaRegions_.size()) throw cms::Exception("Settings.cc: You specified an eta sector number in EtaSecsReduceLayers which exceeds the total number of eta sectors!")<<iEtaReg<<" "<<etaRegions_.size()<<endl;
  }

  // Duplicate track removal algorithm 50 must not be run in parallel with any other.
  if (dupTrkAlgFit_ == 50) {
    if (dupTrkAlgRphi_ != 0 || dupTrkAlg3D_ != 0) throw cms::Exception("Settings.c: Invalid cfg parameters -- If using DupTrkAlgFit = 50, you must disable all other duplicate track removal algorithms.");
  }

  // Chains of m bin ranges for output of HT.
  if ( ! busySectorMbinOrder_.empty() ) {
    // User has specified an order in which the m bins should be chained together. Check if it makes sense.
    if (busySectorMbinOrder_.size() != houghNbinsPt_) throw cms::Exception("Settings.cc: Invalid cfg parameters - BusySectorMbinOrder used by HT MUX contains wrong number of elements. Unless you are optimising the MUX, suggest you configure it to an empty vector.");
    set<unsigned int> mOrderCheck;
    for (const unsigned int& m : busySectorMbinOrder_) {
      mOrderCheck.insert(m);
    }
    if (mOrderCheck.size() != houghNbinsPt_) throw cms::Exception("Settings.cc: Invalid cfg parameters - BusySectorMbinOrder used by HT MUX contains duplicate elements.");
    unsigned int sum_nr = 0;
    for (unsigned int nr : busySectorMbinRanges_) {
      sum_nr += nr;
    }
    if (sum_nr != houghNbinsPt_) throw cms::Exception("Settings.cc: Invalid cfg parameters - Sum of entries in BusySectorMbinRanges is incorrect.");
  }

  if (miniHTstage_) {
    if (enableMerge2x2_) throw cms::Exception("Settings.cc: it is not allowed to enable both MiniHTstage & EnableMerge2x2 options.");
    // Options for 2nd stage mini HT
    if (shape_ != 0) throw cms::Exception("Settings.cc: Invalid cfg parameters - 2nd stage mini HT only allowed for square-shaped cells.");
    if (miniHoughNbinsPt_ != 2 || miniHoughNbinsPhi_ != 2) throw cms::Exception("Settings.cc: 2nd mini HT has so dar only been implemented in C++ for 2x2.");
  }

  if (enableMerge2x2_) {
    if (miniHTstage_) throw cms::Exception("Settings.cc: it is not allowed to enable both MiniHTstage & EnableMerge2x2 options.");
    // Merging of HT cells has not yet been implemented for diamond or hexagonal HT cell shape.
    if (enableMerge2x2_ && shape_ != 0) throw cms::Exception("Settings.cc: Invalid cfg parameters - merging only allowed for square-shaped cells.");
  }

  // Do not use our private dead module emulation together with the communal Tracklet/TMTT dead module emulation 
  // developed for the Stress Test.
  if (deadSimulateFrac_ > 0. && killScenario_ > 0) throw cms::Exception("Settings.cc: Invalid cfg parameters - don't enable both DeadSimulateFrac and KillScenario");

  // Check Kalman fit params.
  if (kalmanMaxNumStubs_ < kalmanMinNumStubs_) throw cms::Exception("Settings.cc: Invalid cfg parameters - KalmanMaxNumStubs is less than KalmanMaxNumStubs.");
}


bool Settings::isHTRPhiEtaRegWhitelisted(unsigned const iEtaReg) const
{
  bool whitelisted = true;

  bool const whitelist_enabled = ( ! etaRegWhitelist_.empty() );
  if (whitelist_enabled) {
    whitelisted = (std::count(etaRegWhitelist_.begin(), etaRegWhitelist_.end(), iEtaReg) > 0);
  }

  return whitelisted;
}

}
