#include "L1Trigger/TrackFindingTMTT/interface/Settings.h"
#include "FWCore/Utilities/interface/Exception.h"
#include <set>
#include <cmath>

using namespace std;

namespace tmtt {

  ///=== Hybrid Tracking
  ///=== Set config params for HYBRID TRACKING via hard-wired consts to allow use outside CMSSW.

  Settings::Settings()
      :  //--------------------------------------------------------------------------------------------------
        // TMTT related configuration parameters, including Kalman Filter.
        // Meaning of these parameters explained in TrackFindingTMTT/python/TMTrackProducer_Defaults_cfi.py
        //--------------------------------------------------------------------------------------------------

        // General cfg params
        enableDigitize_(false),
        useApproxB_(true),
        bApprox_gradient_(0.886454),
        bApprox_intercept_(0.504148),
        numPhiNonants_(9),
        numPhiSectors_(9),
        chosenRofPhi_(55.),  // Hourglass radius in r-phi (tracklet)
        etaRegions_(
            {-2.4, -2.08, -1.68, -1.26, -0.90, -0.62, -0.41, -0.20, 0.0, 0.20, 0.41, 0.62, 0.90, 1.26, 1.68, 2.08, 2.4}),
        chosenRofZ_(50.0),  // Hourglass radius in r-z (this must be tmtt)
        houghMinPt_(2.0),   // L1 track pt cut
        minStubLayers_(4),
        minPtToReduceLayers_(99999.),
        reduceLayerID_(true),
        minFracMatchStubsOnReco_(-99),
        minFracMatchStubsOnTP_(-99),
        minNumMatchLayers_(4),
        minNumMatchPSLayers_(0),
        stubMatchStrict_(false),

        // Kalman filter track fit cfg
        kalmanDebugLevel_(0),
        //kalmanDebugLevel_(2), // Good for debugging
        kalmanMinNumStubs_(4),
        kalmanMaxNumStubs_(6),
        kalmanAddBeamConstr_(false),  // Apply post-fit beam-spot constraint to 5-param fit
        kalmanRemove2PScut_(true),
        kalmanMaxSkipLayersHard_(1),  // On "hard" input tracks
        kalmanMaxSkipLayersEasy_(2),  // On "easy" input tracks
        kalmanMaxStubsEasy_(10),      // Max. #stubs an input track can have to be defined "easy"
        kfUseMaybeLayers_(true),
        kfLayerVsPtToler_({999., 999., 0.1, 0.1, 0.05, 0.05, 0.05}),
        kfLayerVsD0Cut5_({999., 999., 999., 10., 10., 10., 10.}),
        kfLayerVsZ0Cut5_({999., 999., 25.5, 25.5, 25.5, 25.5, 25.5}),
        kfLayerVsZ0Cut4_({999., 999., 15., 15., 15., 15., 15.}),
        kfLayerVsChiSq5_({999., 999., 10., 30., 80., 120., 160.}),
        kfLayerVsChiSq4_({999., 999., 10., 30., 80., 120., 160.}),
        kalmanMaxStubsPerLayer_(4),  // To save resources, consider at most this many stubs per layer per track.
        kalmanMultiScattTerm_(0.00075),
        kalmanChi2RphiScale_(8),
        kalmanHOtilted_(true),
        kalmanHOhelixExp_(true),
        kalmanHOalpha_(1),
        kalmanHOprojZcorr_(1),
        kalmanHOfw_(false) {
    hybrid_ = true;
    magneticField_ = 0.;  // Value set later
    killScenario_ = 0;    // Emulation of dead modules

    if (hybrid_) {
      if (not useApproxB_) {
        throw cms::Exception("BadConfig")
            << "TMTT Settings Error: module tilt angle unknown, so must set useApproxB = true";
      }
    }
  }

  ///=== TMTT tracking.
  ///=== Get configuration parameters from python cfg for TMTT tracking.

  Settings::Settings(const edm::ParameterSet& iConfig)
      :

        // See either Analyze_Defaults_cfi.py or Settings.h for description of these parameters.

        //=== Tags for Input ES & ED data.
        magneticFieldInputTag_(iConfig.getParameter<edm::ESInputTag>("magneticFieldInputTag")),
        trackerGeometryInputTag_(iConfig.getParameter<edm::ESInputTag>("trackerGeometryInputTag")),
        trackerTopologyInputTag_(iConfig.getParameter<edm::ESInputTag>("trackerTopologyInputTag")),
        ttStubAlgoInputTag_(iConfig.getParameter<edm::ESInputTag>("ttStubAlgoInputTag")),

        stubInputTag_(iConfig.getParameter<edm::InputTag>("stubInputTag")),
        tpInputTag_(iConfig.getParameter<edm::InputTag>("tpInputTag")),
        stubTruthInputTag_(iConfig.getParameter<edm::InputTag>("stubTruthInputTag")),
        clusterTruthInputTag_(iConfig.getParameter<edm::InputTag>("clusterTruthInputTag")),
        genJetInputTag_(iConfig.getParameter<edm::InputTag>("genJetInputTag")),

        //=== Parameter sets for differents types of configuration parameter.
        genCuts_(iConfig.getParameter<edm::ParameterSet>("GenCuts")),
        stubCuts_(iConfig.getParameter<edm::ParameterSet>("StubCuts")),
        stubDigitize_(iConfig.getParameter<edm::ParameterSet>("StubDigitize")),
        trackerModuleType_(iConfig.getParameter<edm::ParameterSet>("TrackerModuleType")),
        geometricProc_(iConfig.getParameter<edm::ParameterSet>("GeometricProc")),
        phiSectors_(iConfig.getParameter<edm::ParameterSet>("PhiSectors")),
        etaSectors_(iConfig.getParameter<edm::ParameterSet>("EtaSectors")),
        htArraySpecRphi_(iConfig.getParameter<edm::ParameterSet>("HTArraySpecRphi")),
        htFillingRphi_(iConfig.getParameter<edm::ParameterSet>("HTFillingRphi")),
        rzFilterOpts_(iConfig.getParameter<edm::ParameterSet>("RZfilterOpts")),
        l1TrackDef_(iConfig.getParameter<edm::ParameterSet>("L1TrackDef")),
        dupTrkRemoval_(iConfig.getParameter<edm::ParameterSet>("DupTrkRemoval")),
        trackMatchDef_(iConfig.getParameter<edm::ParameterSet>("TrackMatchDef")),
        trackFitSettings_(iConfig.getParameter<edm::ParameterSet>("TrackFitSettings")),
        deadModuleOpts_(iConfig.getParameter<edm::ParameterSet>("DeadModuleOpts")),
        trackDigi_(iConfig.getParameter<edm::ParameterSet>("TrackDigi")),

        //=== General settings

        enableMCtruth_(iConfig.getParameter<bool>("EnableMCtruth")),
        enableHistos_(iConfig.getParameter<bool>("EnableHistos")),
        enableOutputIntermediateTTTracks_(iConfig.getParameter<bool>("EnableOutputIntermediateTTTracks")),

        //=== Cuts on MC truth tracks used for tracking efficiency measurements.

        genMinPt_(genCuts_.getParameter<double>("GenMinPt")),
        genMaxAbsEta_(genCuts_.getParameter<double>("GenMaxAbsEta")),
        genMaxVertR_(genCuts_.getParameter<double>("GenMaxVertR")),
        genMaxVertZ_(genCuts_.getParameter<double>("GenMaxVertZ")),
        genMaxD0_(genCuts_.getParameter<double>("GenMaxD0")),
        genMaxZ0_(genCuts_.getParameter<double>("GenMaxZ0")),
        genMinStubLayers_(genCuts_.getParameter<unsigned int>("GenMinStubLayers")),

        //=== Cuts applied to stubs before arriving in L1 track finding board.

        degradeBendRes_(stubCuts_.getParameter<unsigned int>("DegradeBendRes")),
        maxStubEta_(stubCuts_.getParameter<double>("MaxStubEta")),
        killLowPtStubs_(stubCuts_.getParameter<bool>("KillLowPtStubs")),
        printStubWindows_(stubCuts_.getParameter<bool>("PrintStubWindows")),
        bendCut_(stubCuts_.getParameter<double>("BendCut")),
        bendCutExtra_(stubCuts_.getParameter<double>("BendCutExtra")),
        orderStubsByBend_(stubCuts_.getParameter<bool>("OrderStubsByBend")),

        //=== Optional stub digitization.

        enableDigitize_(stubDigitize_.getParameter<bool>("EnableDigitize")),

        //--- Parameters available in MP board.
        phiSectorBits_(stubDigitize_.getParameter<unsigned int>("PhiSectorBits")),
        phiSBits_(stubDigitize_.getParameter<unsigned int>("PhiSBits")),
        phiSRange_(stubDigitize_.getParameter<double>("PhiSRange")),
        rtBits_(stubDigitize_.getParameter<unsigned int>("RtBits")),
        rtRange_(stubDigitize_.getParameter<double>("RtRange")),
        zBits_(stubDigitize_.getParameter<unsigned int>("ZBits")),
        zRange_(stubDigitize_.getParameter<double>("ZRange")),
        //--- Parameters available in GP board (excluding any in common with MP specified above).
        phiNBits_(stubDigitize_.getParameter<unsigned int>("PhiNBits")),
        phiNRange_(stubDigitize_.getParameter<double>("PhiNRange")),
        bendBits_(stubDigitize_.getParameter<unsigned int>("BendBits")),

        //=== Tracker Module Type for FW.
        pitchVsType_(trackerModuleType_.getParameter<vector<double>>("PitchVsType")),
        spaceVsType_(trackerModuleType_.getParameter<vector<double>>("SpaceVsType")),
        barrelVsTypeTmp_(trackerModuleType_.getParameter<vector<unsigned int>>("BarrelVsType")),
        psVsTypeTmp_(trackerModuleType_.getParameter<vector<unsigned int>>("PSVsType")),
        tiltedVsTypeTmp_(trackerModuleType_.getParameter<vector<unsigned int>>("TiltedVsType")),

        //=== Configuration of Geometric Processor.
        useApproxB_(geometricProc_.getParameter<bool>("UseApproxB")),
        bApprox_gradient_(geometricProc_.getParameter<double>("BApprox_gradient")),
        bApprox_intercept_(geometricProc_.getParameter<double>("BApprox_intercept")),

        //=== Division of Tracker into phi sectors.
        numPhiNonants_(phiSectors_.getParameter<unsigned int>("NumPhiNonants")),
        numPhiSectors_(phiSectors_.getParameter<unsigned int>("NumPhiSectors")),
        chosenRofPhi_(phiSectors_.getParameter<double>("ChosenRofPhi")),
        useStubPhi_(phiSectors_.getParameter<bool>("UseStubPhi")),
        useStubPhiTrk_(phiSectors_.getParameter<bool>("UseStubPhiTrk")),
        assumedPhiTrkRes_(phiSectors_.getParameter<double>("AssumedPhiTrkRes")),
        calcPhiTrkRes_(phiSectors_.getParameter<bool>("CalcPhiTrkRes")),

        //=== Division of Tracker into eta sectors.
        etaRegions_(etaSectors_.getParameter<vector<double>>("EtaRegions")),
        chosenRofZ_(etaSectors_.getParameter<double>("ChosenRofZ")),
        beamWindowZ_(etaSectors_.getParameter<double>("BeamWindowZ")),
        allowOver2EtaSecs_(etaSectors_.getParameter<bool>("AllowOver2EtaSecs")),

        //=== r-phi Hough transform array specifications.
        houghMinPt_(htArraySpecRphi_.getParameter<double>("HoughMinPt")),
        houghNbinsPt_(htArraySpecRphi_.getParameter<unsigned int>("HoughNbinsPt")),
        houghNbinsPhi_(htArraySpecRphi_.getParameter<unsigned int>("HoughNbinsPhi")),
        enableMerge2x2_(htArraySpecRphi_.getParameter<bool>("EnableMerge2x2")),
        maxPtToMerge2x2_(htArraySpecRphi_.getParameter<double>("MaxPtToMerge2x2")),
        numSubSecsEta_(htArraySpecRphi_.getParameter<unsigned int>("NumSubSecsEta")),
        shape_(htArraySpecRphi_.getParameter<unsigned int>("Shape")),
        miniHTstage_(htArraySpecRphi_.getParameter<bool>("MiniHTstage")),
        miniHoughNbinsPt_(htArraySpecRphi_.getParameter<unsigned int>("MiniHoughNbinsPt")),
        miniHoughNbinsPhi_(htArraySpecRphi_.getParameter<unsigned int>("MiniHoughNbinsPhi")),
        miniHoughMinPt_(htArraySpecRphi_.getParameter<double>("MiniHoughMinPt")),
        miniHoughDontKill_(htArraySpecRphi_.getParameter<bool>("MiniHoughDontKill")),
        miniHoughDontKillMinPt_(htArraySpecRphi_.getParameter<double>("MiniHoughDontKillMinPt")),
        miniHoughLoadBalance_(htArraySpecRphi_.getParameter<unsigned int>("MiniHoughLoadBalance")),

        //=== Rules governing how stubs are filled into the r-phi Hough Transform array.
        killSomeHTCellsRphi_(htFillingRphi_.getParameter<unsigned int>("KillSomeHTCellsRphi")),
        useBendFilter_(htFillingRphi_.getParameter<bool>("UseBendFilter")),
        maxStubsInCell_(htFillingRphi_.getParameter<unsigned int>("MaxStubsInCell")),
        maxStubsInCellMiniHough_(htFillingRphi_.getParameter<unsigned int>("MaxStubsInCellMiniHough")),
        busySectorKill_(htFillingRphi_.getParameter<bool>("BusySectorKill")),
        busySectorNumStubs_(htFillingRphi_.getParameter<unsigned int>("BusySectorNumStubs")),
        busySectorMbinRanges_(htFillingRphi_.getParameter<vector<unsigned int>>("BusySectorMbinRanges")),
        busySectorMbinOrder_(htFillingRphi_.getParameter<vector<unsigned int>>("BusySectorMbinOrder")),
        busyInputSectorKill_(htFillingRphi_.getParameter<bool>("BusyInputSectorKill")),
        busyInputSectorNumStubs_(htFillingRphi_.getParameter<unsigned int>("BusyInputSectorNumStubs")),
        muxOutputsHT_(htFillingRphi_.getParameter<unsigned int>("MuxOutputsHT")),
        etaRegWhitelist_(htFillingRphi_.getParameter<vector<unsigned int>>("EtaRegWhitelist")),

        //=== Options controlling r-z track filters (or any other track filters run after the Hough transform, as opposed to inside it).

        rzFilterName_(rzFilterOpts_.getParameter<string>("RZFilterName")),
        seedResCut_(rzFilterOpts_.getParameter<double>("SeedResCut")),
        keepAllSeed_(rzFilterOpts_.getParameter<bool>("KeepAllSeed")),
        maxSeedCombinations_(rzFilterOpts_.getParameter<unsigned int>("MaxSeedCombinations")),
        maxGoodSeedCombinations_(rzFilterOpts_.getParameter<unsigned int>("MaxGoodSeedCombinations")),
        maxSeedsPerStub_(rzFilterOpts_.getParameter<unsigned int>("MaxSeedsPerStub")),
        zTrkSectorCheck_(rzFilterOpts_.getParameter<bool>("zTrkSectorCheck")),
        minFilterLayers_(rzFilterOpts_.getParameter<unsigned int>("MinFilterLayers")),

        //=== Rules for deciding when the track finding has found an L1 track candidate

        minStubLayers_(l1TrackDef_.getParameter<unsigned int>("MinStubLayers")),
        minPtToReduceLayers_(l1TrackDef_.getParameter<double>("MinPtToReduceLayers")),
        etaSecsReduceLayers_(l1TrackDef_.getParameter<vector<unsigned int>>("EtaSecsReduceLayers")),
        reduceLayerID_(l1TrackDef_.getParameter<bool>("ReducedLayerID")),

        //=== Specification of algorithm to eliminate duplicate tracks.

        dupTrkAlgFit_(dupTrkRemoval_.getParameter<unsigned int>("DupTrkAlgFit")),

        //=== Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle).

        minFracMatchStubsOnReco_(trackMatchDef_.getParameter<double>("MinFracMatchStubsOnReco")),
        minFracMatchStubsOnTP_(trackMatchDef_.getParameter<double>("MinFracMatchStubsOnTP")),
        minNumMatchLayers_(trackMatchDef_.getParameter<unsigned int>("MinNumMatchLayers")),
        minNumMatchPSLayers_(trackMatchDef_.getParameter<unsigned int>("MinNumMatchPSLayers")),
        stubMatchStrict_(trackMatchDef_.getParameter<bool>("StubMatchStrict")),

        //=== Track Fitting Settings

        trackFitters_(trackFitSettings_.getParameter<vector<std::string>>("TrackFitters")),
        useRZfilter_(trackFitSettings_.getParameter<vector<std::string>>("UseRZfilter")),
        detailedFitOutput_(trackFitSettings_.getParameter<bool>("DetailedFitOutput")),
        trackFitCheat_(trackFitSettings_.getParameter<bool>("TrackFitCheat")),
        //
        numTrackFitIterations_(trackFitSettings_.getParameter<unsigned int>("NumTrackFitIterations")),
        killTrackFitWorstHit_(trackFitSettings_.getParameter<bool>("KillTrackFitWorstHit")),
        generalResidualCut_(trackFitSettings_.getParameter<double>("GeneralResidualCut")),
        killingResidualCut_(trackFitSettings_.getParameter<double>("KillingResidualCut")),
        //
        digitizeSLR_(trackFitSettings_.getParameter<bool>("DigitizeSLR")),
        dividerBitsHelix_(trackFitSettings_.getParameter<unsigned int>("DividerBitsHelix")),
        dividerBitsHelixZ_(trackFitSettings_.getParameter<unsigned int>("DividerBitsHelixZ")),
        ShiftingBitsDenRPhi_(trackFitSettings_.getParameter<unsigned int>("ShiftingBitsDenRPhi")),
        ShiftingBitsDenRZ_(trackFitSettings_.getParameter<unsigned int>("ShiftingBitsDenRZ")),
        ShiftingBitsPt_(trackFitSettings_.getParameter<unsigned int>("ShiftingBitsPt")),
        ShiftingBitsPhi_(trackFitSettings_.getParameter<unsigned int>("ShiftingBitsPhi")),

        ShiftingBitsLambda_(trackFitSettings_.getParameter<unsigned int>("ShiftingBitsLambda")),
        ShiftingBitsZ0_(trackFitSettings_.getParameter<unsigned int>("ShiftingBitsZ0")),
        slr_chi2cut_(trackFitSettings_.getParameter<double>("SLR_chi2cut")),
        residualCut_(trackFitSettings_.getParameter<double>("ResidualCut")),
        //
        kalmanDebugLevel_(trackFitSettings_.getParameter<unsigned int>("KalmanDebugLevel")),
        kalmanMinNumStubs_(trackFitSettings_.getParameter<unsigned int>("KalmanMinNumStubs")),
        kalmanMaxNumStubs_(trackFitSettings_.getParameter<unsigned int>("KalmanMaxNumStubs")),
        kalmanAddBeamConstr_(trackFitSettings_.getParameter<bool>("KalmanAddBeamConstr")),
        kalmanRemove2PScut_(trackFitSettings_.getParameter<bool>("KalmanRemove2PScut")),
        kalmanMaxSkipLayersHard_(trackFitSettings_.getParameter<unsigned>("KalmanMaxSkipLayersHard")),
        kalmanMaxSkipLayersEasy_(trackFitSettings_.getParameter<unsigned>("KalmanMaxSkipLayersEasy")),
        kalmanMaxStubsEasy_(trackFitSettings_.getParameter<unsigned>("KalmanMaxStubsEasy")),
        kfUseMaybeLayers_(trackFitSettings_.getParameter<bool>("KFUseMaybeLayers")),

        kfLayerVsPtToler_(trackFitSettings_.getParameter<vector<double>>("KFLayerVsPtToler")),
        kfLayerVsD0Cut5_(trackFitSettings_.getParameter<vector<double>>("KFLayerVsD0Cut5")),
        kfLayerVsZ0Cut5_(trackFitSettings_.getParameter<vector<double>>("KFLayerVsZ0Cut5")),
        kfLayerVsZ0Cut4_(trackFitSettings_.getParameter<vector<double>>("KFLayerVsZ0Cut4")),
        kfLayerVsChiSq5_(trackFitSettings_.getParameter<vector<double>>("KFLayerVsChiSq5")),
        kfLayerVsChiSq4_(trackFitSettings_.getParameter<vector<double>>("KFLayerVsChiSq4")),

        kalmanMaxStubsPerLayer_(trackFitSettings_.getParameter<unsigned>("KalmanMaxStubsPerLayer")),
        kalmanMultiScattTerm_(trackFitSettings_.getParameter<double>("KalmanMultiScattTerm")),
        kalmanChi2RphiScale_(trackFitSettings_.getParameter<unsigned>("KalmanChi2RphiScale")),
        kalmanHOtilted_(trackFitSettings_.getParameter<bool>("KalmanHOtilted")),
        kalmanHOhelixExp_(trackFitSettings_.getParameter<bool>("KalmanHOhelixExp")),
        kalmanHOalpha_(trackFitSettings_.getParameter<unsigned int>("KalmanHOalpha")),
        kalmanHOprojZcorr_(trackFitSettings_.getParameter<unsigned int>("KalmanHOprojZcorr")),
        kalmanHOfw_(trackFitSettings_.getParameter<bool>("KalmanHOfw")),

        //=== Treatment of dead modules.

        killScenario_(deadModuleOpts_.getParameter<unsigned int>("KillScenario")),
        killRecover_(deadModuleOpts_.getParameter<bool>("KillRecover")),

        //=== Track digitisation configuration for various track fitters

        slr_skipTrackDigi_(trackDigi_.getParameter<bool>("SLR_skipTrackDigi")),
        slr_oneOver2rBits_(trackDigi_.getParameter<unsigned int>("SLR_oneOver2rBits")),
        slr_oneOver2rRange_(trackDigi_.getParameter<double>("SLR_oneOver2rRange")),
        slr_d0Bits_(trackDigi_.getParameter<unsigned int>("SLR_d0Bits")),
        slr_d0Range_(trackDigi_.getParameter<double>("SLR_d0Range")),
        slr_phi0Bits_(trackDigi_.getParameter<unsigned int>("SLR_phi0Bits")),
        slr_phi0Range_(trackDigi_.getParameter<double>("SLR_phi0Range")),
        slr_z0Bits_(trackDigi_.getParameter<unsigned int>("SLR_z0Bits")),
        slr_z0Range_(trackDigi_.getParameter<double>("SLR_z0Range")),
        slr_tanlambdaBits_(trackDigi_.getParameter<unsigned int>("SLR_tanlambdaBits")),
        slr_tanlambdaRange_(trackDigi_.getParameter<double>("SLR_tanlambdaRange")),
        slr_chisquaredBits_(trackDigi_.getParameter<unsigned int>("SLR_chisquaredBits")),
        slr_chisquaredRange_(trackDigi_.getParameter<double>("SLR_chisquaredRange")),
        //
        kf_skipTrackDigi_(trackDigi_.getParameter<bool>("KF_skipTrackDigi")),
        kf_oneOver2rBits_(trackDigi_.getParameter<unsigned int>("KF_oneOver2rBits")),
        kf_oneOver2rRange_(trackDigi_.getParameter<double>("KF_oneOver2rRange")),
        kf_d0Bits_(trackDigi_.getParameter<unsigned int>("KF_d0Bits")),
        kf_d0Range_(trackDigi_.getParameter<double>("KF_d0Range")),
        kf_phi0Bits_(trackDigi_.getParameter<unsigned int>("KF_phi0Bits")),
        kf_phi0Range_(trackDigi_.getParameter<double>("KF_phi0Range")),
        kf_z0Bits_(trackDigi_.getParameter<unsigned int>("KF_z0Bits")),
        kf_z0Range_(trackDigi_.getParameter<double>("KF_z0Range")),
        kf_tanlambdaBits_(trackDigi_.getParameter<unsigned int>("KF_tanlambdaBits")),
        kf_tanlambdaRange_(trackDigi_.getParameter<double>("KF_tanlambdaRange")),
        kf_chisquaredBits_(trackDigi_.getParameter<unsigned int>("KF_chisquaredBits")),
        kf_chisquaredRange_(trackDigi_.getParameter<double>("KF_chisquaredRange")),
        kf_chisquaredBinEdges_(trackDigi_.getParameter<vector<double>>("KF_chisquaredBinEdges")),
        //
        other_skipTrackDigi_(trackDigi_.getParameter<bool>("Other_skipTrackDigi")),

        // Plot options
        resPlotOpt_(iConfig.getParameter<bool>("ResPlotOpt")),

        // Name of output EDM file if any.
        // N.B. This parameter does not appear inside TMTrackProducer_Defaults_cfi.py . It is created inside
        // tmtt_tf_analysis_cfg.py .
        writeOutEdmFile_(iConfig.getUntrackedParameter<bool>("WriteOutEdmFile", true)),

        // Bfield in Tesla. (Unknown at job initiation. Set to true value for each event
        magneticField_(0.),

        // Hybrid tracking
        hybrid_(iConfig.getParameter<bool>("Hybrid")) {
    // If user didn't specify any PDG codes, use e,mu,pi,K,p, to avoid picking up unstable particles like Xi-.
    vector<unsigned int> genPdgIdsUnsigned(genCuts_.getParameter<vector<unsigned int>>("GenPdgIds"));
    if (genPdgIdsUnsigned.empty()) {
      genPdgIdsUnsigned = {11, 13, 211, 321, 2212};
    }

    // For simplicity, user need not distinguish particles from antiparticles in configuration file.
    // But here we must store both explicitely in Settings, since TrackingParticleSelector expects them.
    for (unsigned int i = 0; i < genPdgIdsUnsigned.size(); i++) {
      genPdgIds_.push_back(genPdgIdsUnsigned[i]);
      genPdgIds_.push_back(-genPdgIdsUnsigned[i]);
    }

    // Clean up list of fitters that require the r-z track filter to be run before them,
    // by removing those fitters that are not to be run.
    vector<string> useRZfilterTmp;
    for (const string& name : useRZfilter_) {
      if (std::count(trackFitters_.begin(), trackFitters_.end(), name) > 0)
        useRZfilterTmp.push_back(name);
    }
    useRZfilter_ = useRZfilterTmp;

    // As python cfg doesn't know type "vbool", fix it here.
    for (unsigned int i = 0; i < barrelVsTypeTmp_.size(); i++) {
      barrelVsType_.push_back(bool(barrelVsTypeTmp_[i]));
      psVsType_.push_back(bool(psVsTypeTmp_[i]));
      tiltedVsType_.push_back(bool(tiltedVsTypeTmp_[i]));
    }

    //--- Sanity checks

    if (!(useStubPhi_ || useStubPhiTrk_))
      throw cms::Exception("BadConfig")
          << "Settings: Invalid cfg parameters - You cant set both UseStubPhi & useStubPhiTrk to false.";

    if (minNumMatchLayers_ > minStubLayers_)
      throw cms::Exception("BadConfig")
          << "Settings: Invalid cfg parameters - You are setting the minimum number of layers incorrectly : type A.";
    if (genMinStubLayers_ > minStubLayers_)
      throw cms::Exception("BadConfig")
          << "Settings: Invalid cfg parameters - You are setting the minimum number of layers incorrectly : type B.";
    if (minNumMatchLayers_ > genMinStubLayers_)
      throw cms::Exception("BadConfig")
          << "Settings: Invalid cfg parameters - You are setting the minimum number of layers incorrectly : type C.";

    // If reducing number of required layers for high Pt tracks, then above checks must be redone.
    bool doReduceLayers = (minPtToReduceLayers_ < 10000. || not etaSecsReduceLayers_.empty());
    if (doReduceLayers && minStubLayers_ > 4) {
      if (minNumMatchLayers_ > minStubLayers_ - 1)
        throw cms::Exception("BadConfig")
            << "Settings: Invalid cfg parameters - You are setting the minimum number of layers incorrectly : type D.";
      if (genMinStubLayers_ > minStubLayers_ - 1)
        throw cms::Exception("BadConfig")
            << "Settings: Invalid cfg parameters - You are setting the minimum number of layers incorrectly : type E.";
    }

    constexpr float verySmall = 0.1;
    if (houghMinPt_ < verySmall)
      throw cms::Exception("BadConfig") << "Settings: Invalid cfg parameters -- HoughMinPt must be positive.";
    miniHoughMinPt_ = std::max(miniHoughMinPt_, houghMinPt_);

    for (const unsigned int& iEtaReg : etaSecsReduceLayers_) {
      if (iEtaReg >= etaRegions_.size())
        throw cms::Exception("BadConfig") << "Settings: You specified an eta sector number in EtaSecsReduceLayers "
                                             "which exceeds the total number of eta sectors! "
                                          << iEtaReg << " " << etaRegions_.size();
    }

    // Chains of m bin ranges for output of HT.
    if (!busySectorMbinOrder_.empty()) {
      // User has specified an order in which the m bins should be chained together. Check if it makes sense.
      if (busySectorMbinOrder_.size() != houghNbinsPt_)
        throw cms::Exception("BadConfig")
            << "Settings: Invalid cfg parameters - BusySectorMbinOrder used by HT MUX contains wrong number of "
               "elements. Unless you are optimising the MUX, suggest you configure it to an empty vector.";
      set<unsigned int> mOrderCheck;
      for (const unsigned int& m : busySectorMbinOrder_) {
        mOrderCheck.insert(m);
      }
      if (mOrderCheck.size() != houghNbinsPt_)
        throw cms::Exception("BadConfig")
            << "Settings: Invalid cfg parameters - BusySectorMbinOrder used by HT MUX contains duplicate elements.";
      unsigned int sum_nr = 0;
      for (unsigned int nr : busySectorMbinRanges_) {
        sum_nr += nr;
      }
      if (sum_nr != houghNbinsPt_)
        throw cms::Exception("BadConfig")
            << "Settings: Invalid cfg parameters - Sum of entries in BusySectorMbinRanges is incorrect.";
    }

    if (miniHTstage_) {
      if (enableMerge2x2_)
        throw cms::Exception("BadConfig")
            << "Settings: it is not allowed to enable both MiniHTstage & EnableMerge2x2 options.";
      // Options for 2nd stage mini HT
      if (shape_ != 0)
        throw cms::Exception("BadConfig")
            << "Settings: Invalid cfg parameters - 2nd stage mini HT only allowed for square-shaped cells.";
      if (miniHoughNbinsPt_ != 2 || miniHoughNbinsPhi_ != 2)
        throw cms::Exception("BadConfig") << "Settings: 2nd mini HT has so dar only been implemented in C++ for 2x2.";
    }

    if (enableMerge2x2_) {
      if (miniHTstage_)
        throw cms::Exception("BadConfig")
            << "Settings: it is not allowed to enable both MiniHTstage & EnableMerge2x2 options.";
      // Merging of HT cells has not yet been implemented for diamond or hexagonal HT cell shape.
      if (enableMerge2x2_ && shape_ != 0)
        throw cms::Exception("BadConfig")
            << "Settings: Invalid cfg parameters - merging only allowed for square-shaped cells.";
    }

    // Check Kalman fit params.
    if (kalmanMaxNumStubs_ < kalmanMinNumStubs_)
      throw cms::Exception("BadConfig")
          << "Settings: Invalid cfg parameters - KalmanMaxNumStubs is less than KalmanMaxNumStubs.";
  }

  bool Settings::isHTRPhiEtaRegWhitelisted(unsigned const iEtaReg) const {
    bool whitelisted = true;

    bool const whitelist_enabled = (!etaRegWhitelist_.empty());
    if (whitelist_enabled) {
      whitelisted = (std::count(etaRegWhitelist_.begin(), etaRegWhitelist_.end(), iEtaReg) > 0);
    }

    return whitelisted;
  }

}  // namespace tmtt
