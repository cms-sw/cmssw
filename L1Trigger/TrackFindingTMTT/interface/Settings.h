#ifndef L1Trigger_TrackFindingTMTT_Settings_h
#define L1Trigger_TrackFindingTMTT_Settings_h

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "CLHEP/Units/PhysicalConstants.h"
#include <vector>
#include <iostream>
#include <atomic>

// Stores all configuration parameters + some hard-wired constants.

namespace tmtt {

  class Settings {
  public:
    // Constructor for HYBRID (sets config to hard-wired consts to allow use outside CMSSW).
    Settings();

    // Constructor for TMTT (reads config from python cfg)
    Settings(const edm::ParameterSet& iConfig);

    // Input tags for ES & ED data.
    edm::ESInputTag magneticFieldInputTag() const { return magneticFieldInputTag_; }
    edm::ESInputTag trackerGeometryInputTag() const { return trackerGeometryInputTag_; }
    edm::ESInputTag trackerTopologyInputTag() const { return trackerTopologyInputTag_; }
    edm::ESInputTag ttStubAlgoInputTag() const { return ttStubAlgoInputTag_; }

    edm::InputTag stubInputTag() const { return stubInputTag_; }
    edm::InputTag tpInputTag() const { return tpInputTag_; }
    edm::InputTag stubTruthInputTag() const { return stubTruthInputTag_; }
    edm::InputTag clusterTruthInputTag() const { return clusterTruthInputTag_; }
    edm::InputTag genJetInputTag() const { return genJetInputTag_; }

    //=== General settings.

    // Enable all use of MC truth info (disable to save CPU).
    bool enableMCtruth() const { return enableMCtruth_; }
    // Enable output histograms & job tracking performance summary (disable to save CPU).
    bool enableHistos() const { return enableHistos_; }
    // Enable output of TTTracks from part-way through tracking chain (after HT & RZ).
    bool enableOutputIntermediateTTTracks() const { return enableOutputIntermediateTTTracks_; }

    //=== Cuts on MC truth tracks for tracking efficiency measurements.

    double genMinPt() const { return genMinPt_; }
    double genMaxAbsEta() const { return genMaxAbsEta_; }
    double genMaxVertR() const { return genMaxVertR_; }
    double genMaxVertZ() const { return genMaxVertZ_; }
    double genMaxD0() const { return genMaxD0_; }
    double genMaxZ0() const { return genMaxZ0_; }
    const std::vector<int>& genPdgIds() const { return genPdgIds_; }
    // Additional cut on MC truth tracks for algorithmic tracking efficiency measurements.
    unsigned int genMinStubLayers() const { return genMinStubLayers_; }  // Min. number of layers TP made stub in.

    //=== Cuts applied to stubs before arriving in L1 track finding board.

    // Reduce number of bits used by front-end chips to store stub bend info?
    // = 0 (no); = 1 (yes using official recipe); = 2 (yes using TMTT method)
    unsigned int degradeBendRes() const { return degradeBendRes_; }
    // Don't use stubs with eta beyond this cut, since the tracker geometry makes it impossible to reconstruct tracks with them.
    double maxStubEta() const { return maxStubEta_; }
    // Don't use stubs whose measured Pt from bend info is significantly below HTArraySpec.HoughMinPt, where "significantly" means allowing for resolution in q/Pt derived from stub bend resolution HTFilling.BendResolution
    bool killLowPtStubs() const { return killLowPtStubs_; }
    // Print stub windows corresponding to KillLowPtStubs, in python cfg format used by CMSSW.
    bool printStubWindows() const { return printStubWindows_; }
    // Bend resolution assumed by bend filter in units of strip pitch. Also used when assigning stubs to sectors if calcPhiTrkRes() is true.
    double bendCut() const { return bendCut_; }
    // Additional contribution to bend resolution from its encoding into a reduced number of bits.
    // This number is the assumed resolution relative to the naive guess of its value.
    // It is ignored in DegradeBendRes = 0.
    double bendCutExtra() const { return bendCutExtra_; }
    // Order stubs by bend in DTC, such that highest Pt stubs are transmitted first.
    bool orderStubsByBend() const { return orderStubsByBend_; }

    //=== Optional stub digitization configuration

    bool enableDigitize() const { return enableDigitize_; }
    //--- Parameters available in MP board.
    unsigned int phiSectorBits() const { return phiSectorBits_; }
    unsigned int phiSBits() const { return phiSBits_; }
    double phiSRange() const { return phiSRange_; }
    unsigned int rtBits() const { return rtBits_; }
    double rtRange() const { return rtRange_; }
    unsigned int zBits() const { return zBits_; }
    double zRange() const { return zRange_; }
    //--- Parameters available in GP board (excluding any in common with MP specified above).
    unsigned int phiNBits() const { return phiNBits_; }
    double phiNRange() const { return phiNRange_; }
    unsigned int bendBits() const { return bendBits_; }

    //=== Tracker module type for FW.
    const std::vector<double>& pitchVsType() const { return pitchVsType_; }
    const std::vector<double>& spaceVsType() const { return spaceVsType_; }
    const std::vector<bool>& barrelVsType() const { return barrelVsType_; }
    const std::vector<bool>& psVsType() const { return psVsType_; }
    const std::vector<bool>& tiltedVsType() const { return tiltedVsType_; }

    //=== Configuration of Geometric Processor.
    // Use an FPGA-friendly approximation to determine track angle dphi from bend in GP?
    bool useApproxB() const { return useApproxB_; }
    double bApprox_gradient() const { return bApprox_gradient_; }
    double bApprox_intercept() const { return bApprox_intercept_; }

    //=== Definition of phi sectors.

    unsigned int numPhiNonants() const { return numPhiNonants_; }
    unsigned int numPhiSectors() const { return numPhiSectors_; }
    // Use phi of track at this radius as sector hourglass reference radius.
    double chosenRofPhi() const { return chosenRofPhi_; }
    // Require stub phi to be consistent with track of Pt > HTArraySpec.HoughMinPt that crosses HT phi axis?
    bool useStubPhi() const { return useStubPhi_; }
    // Require stub phi0 (or phi65 etc.) as estimated from stub bend, to lie within HT phi axis, allowing tolerance specified below?
    bool useStubPhiTrk() const { return useStubPhiTrk_; }
    // Tolerance in stub phi0 (or phi65) assumed to be this fraction of phi sector width. (N.B. If > 0.5, then stubs can be shared by more than 2 phi sectors).
    double assumedPhiTrkRes() const { return assumedPhiTrkRes_; }
    // If true, tolerance in stub phi0 (or phi65 etc.) will be reduced below AssumedPhiTrkRes if stub bend resolution specified in StubCuts.BendResolution suggests it is safe to do so.
    bool calcPhiTrkRes() const { return calcPhiTrkRes_; }

    //=== Definition of eta sectors.

    const std::vector<double>& etaRegions() const { return etaRegions_; }  // Boundaries of eta regions de
    unsigned int numEtaRegions() const { return (etaRegions_.size() - 1); }
    // Use z of track at this radius for assignment of stubs to phi sectors & also for one of the axes of the r-z HT.
    double chosenRofZ() const { return chosenRofZ_; }
    // Half-width of window supposed to contain beam-spot in z.
    double beamWindowZ() const { return beamWindowZ_; }
    // If True, the code will not throw an error if a stub is assigned to 3 or more eta sectors.
    bool allowOver2EtaSecs() const { return allowOver2EtaSecs_; }

    //=== r-phi Hough transform array specifications.

    double houghMinPt() const { return houghMinPt_; }
    // Dimension in any q/Pt related variable. (If MiniHTstage = True, this refers to mini cells in whole HT array).
    unsigned int houghNbinsPt() const { return houghNbinsPt_; }
    // Dimension in any track-phi related variable. (If MiniHTstage = True, this refers to mini cells in whole HT array).
    unsigned int houghNbinsPhi() const { return houghNbinsPhi_; }
    // Groups of neighbouring 2x2 cells in HT will be treated as if they are a single large cell. (Also enabled in MiniHTstage = True).
    bool enableMerge2x2() const { return enableMerge2x2_; }
    // but only cells with pt < maxPtToMerge2x2() will be merged in this way (irrelevant if enableMerge2x2() = false).
    double maxPtToMerge2x2() const { return maxPtToMerge2x2_; }
    // Subdivide each sector into this number of subsectors in eta within r-phi HT.
    unsigned int numSubSecsEta() const { return numSubSecsEta_; }
    // define cell shape (0 square, 1 diamond, 2 hexagon, 3 brick)
    unsigned int shape() const { return shape_; }
    // Run 2nd stage HT with mini cells inside each 1st stage normal HT cell. N.B. This automatically std::sets EnableMerge2x2 = True & MaxPtToMerge = 999999.
    bool miniHTstage() const { return miniHTstage_; }
    // Number of mini cells along q/Pt & phi axes inside each normal HT cell.
    unsigned int miniHoughNbinsPt() const { return miniHoughNbinsPt_; }
    unsigned int miniHoughNbinsPhi() const { return miniHoughNbinsPhi_; }
    // Below this Pt threshold, the mini HT will not be used, so only tracks found by 1st stage coarse HT will be output. (Used to improve low Pt tracking). (HT cell numbering remains as if mini HT were in use everywhere).
    float miniHoughMinPt() const { return miniHoughMinPt_; }
    // If true, allows tracks found by 1st stage coarse HT to be output if 2nd stage mini HT finds no tracks.
    bool miniHoughDontKill() const { return miniHoughDontKill_; }
    // If MiniHoughDontKill=True, this option restricts it to keep 1st stage HT tracks only if their Pt is exceeds this cut. (Used to improve electron tracking above this threshold).
    float miniHoughDontKillMinPt() const { return miniHoughDontKillMinPt_; }
    // load balancing disabled = 0; static load balancing of output links = 1; dynamic load balancing of output links = 2.
    unsigned int miniHoughLoadBalance() const { return miniHoughLoadBalance_; }

    //=== Rules governing how stubs are filled into the r-phi Hough Transform array.

    // Take all cells in HT array crossed by line corresponding to each stub (= 0) or take only some to reduce rate at cost
    // of efficiency ( > 0). If this option is > 0, it can be 1 or 2, corresponding to different algorithms for rejecting some of the cells.
    unsigned int killSomeHTCellsRphi() const { return killSomeHTCellsRphi_; }
    // Use filter in each HT cell using only stubs which have consistent bend, allowing for resolution specified in StubCuts.BendResolution.
    bool useBendFilter() const { return useBendFilter_; }
    // A filter is used each HT cell, which prevents more than the specified number of stubs being stored in the cell. (Reflecting memory limit of hardware). N.B. If mini-HT is in use, then this cut applies to coarse-HT.
    unsigned int maxStubsInCell() const { return maxStubsInCell_; }
    // Similar cut for Mini-HT.
    unsigned int maxStubsInCellMiniHough() const { return maxStubsInCellMiniHough_; }
    // If this returns true, and if more than busySectorNumStubs() stubs are assigned to tracks by an r-phi HT array, then
    // the excess tracks are killed, with lowest Pt ones killed first. This is because hardware has finite readout time.
    bool busySectorKill() const { return busySectorKill_; }
    unsigned int busySectorNumStubs() const { return busySectorNumStubs_; }
    // If this returns a non-empty std::vector, then the BusySectorNumStubs cut is instead applied to the subset of tracks appearing in the following m bin ranges (q/Pt) of the HT array. The sum of the entries in the std::vector should equal the number of m bins in the HT, although the entries will be rescaled if this is not the case. If the std::vector is empty, this option is disabled. (P.S. If the HT includes "merged" cells, then the m bin ranges specified here should correspond to the bins before merging).
    const std::vector<unsigned int>& busySectorMbinRanges() const { return busySectorMbinRanges_; }
    // If BusySecMbinOrder is empty, then the groupings specified in BusySectorMbinRanges are applied to the m bins in the order
    // 0,1,2,3,4,5 ... . If it is not empty, then they are grouped in the order specified here.
    const std::vector<unsigned int>& busySectorMbinOrder() const { return busySectorMbinOrder_; }
    // If this is True, and more than BusyInputSectorNumStubs() are input to the HT array from the GP, then
    // the excess stubs are killed. This is because HT hardware has finite readin time.
    bool busyInputSectorKill() const { return busyInputSectorKill_; }
    unsigned int busyInputSectorNumStubs() const { return busyInputSectorNumStubs_; }
    // Multiplex the outputs from several HTs onto a single pair of output optical links?
    // Options: 0 = disable Mux; 1 = Dec. 2016 Mux; 2 = Mar 2018 Mux (transverse HT readout by m-bin);
    // 3 = Sept 2019 Mux (transverse HT readout by m-bin)
    unsigned int muxOutputsHT() const { return muxOutputsHT_; }
    // Is specified eta sector enabled?
    bool isHTRPhiEtaRegWhitelisted(unsigned const iEtaReg) const;

    //=== Options controlling r-z track filters (or any other track filters run after the Hough transform, as opposed to inside it).

    // Specify preferred r-z filter (from those available inside TrkRZfilter.cc) - currently only "SeedFilter".
    const std::string& rzFilterName() const { return rzFilterName_; }
    // --- Options relevant for Seed filter, (so only relevant if rzFilterName()="SeedFilter").
    // Cut at this many standard deviations on seed resolution.
    double seedResCut() const { return seedResCut_; }
    // Store stubs compatible with all possible good seed (relevant for Seed filter)?
    bool keepAllSeed() const { return keepAllSeed_; }
    // Maximum number of seed combinations to check (relevant for Seed filter).
    unsigned int maxSeedCombinations() const { return maxSeedCombinations_; }
    // Maximum number of seed combinations consistent with (z0,eta) sector constraints to bother checking per track candidate.
    unsigned int maxGoodSeedCombinations() const { return maxGoodSeedCombinations_; }
    // Maximum number of seeds that a single stub can be included in.
    unsigned int maxSeedsPerStub() const { return maxSeedsPerStub_; }
    // Check that estimated zTrk from seeding stub is within the sector boundaries (relevant for Seed filter)?
    bool zTrkSectorCheck() const { return zTrkSectorCheck_; }
    // Min. number of layers in rz track that must have stubs for track to be declared found.
    unsigned int minFilterLayers() const { return minFilterLayers_; }

    //=== Rules for deciding when the (HT) track finding has found an L1 track candidate

    // Min. number of layers in HT cell that must have stubs for track to be declared found.
    unsigned int minStubLayers() const { return minStubLayers_; }
    // Change min. number of layers cut to (MinStubLayers - 1) for tracks with Pt exceeding this cut.
    // If this is std::set to > 10000, this option is disabled.
    double minPtToReduceLayers() const { return minPtToReduceLayers_; }
    // Change min. number of layers cut to (MinStubLayers - 1) for tracks in these rapidity sectors.
    const std::vector<unsigned int>& etaSecsReduceLayers() const { return etaSecsReduceLayers_; }
    //Reduce this layer ID, so that it takes no more than 8 different values in any eta region (simplifies firmware)?
    bool reduceLayerID() const { return reduceLayerID_; }

    //=== Specification of algorithm to eliminate duplicate tracks

    // Algorithm run on tracks after the track helix fit has been done.
    unsigned int dupTrkAlgFit() const { return dupTrkAlgFit_; }

    //=== Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle).

    //--- Three different ways to define if a tracking particle matches a reco track candidate. (Usually, std::set two of them to ultra loose).
    // Min. fraction of matched stubs relative to number of stubs on reco track.
    double minFracMatchStubsOnReco() const { return minFracMatchStubsOnReco_; }
    // Min. fraction of matched stubs relative to number of stubs on tracking particle.
    double minFracMatchStubsOnTP() const { return minFracMatchStubsOnTP_; }
    // Min. number of matched layers & min. number of matched PS layers..
    unsigned int minNumMatchLayers() const { return minNumMatchLayers_; }
    unsigned int minNumMatchPSLayers() const { return minNumMatchPSLayers_; }
    // Associate stub to TP only if the TP contributed to both its clusters? (If False, then associate even if only one cluster was made by TP).
    bool stubMatchStrict() const { return stubMatchStrict_; }

    //=== Track Fitting Settings

    //--- Options applicable to all track fitters ---

    // Track fitting algorithms to use. You can run several in parallel.
    const std::vector<std::string>& trackFitters() const { return trackFitters_; }
    // Indicate subset of fitters wanting r-z track filter to be run before them.
    // (Excludes fitters that are not run).
    const std::vector<std::string>& useRZfilter() const { return useRZfilter_; }
    // Print detailed summary of track fit performance at end of job (as opposed to a brief one)?
    bool detailedFitOutput() const { return detailedFitOutput_; }
    // Use MC truth to eliminate all fake tracks & all incorrect stubs assigned to tracks before doing fit.
    bool trackFitCheat() const { return trackFitCheat_; }

    //--- Options for chi2 track fitter ---

    // Number of iterations that the track fit should perform.
    unsigned int numTrackFitIterations() const { return numTrackFitIterations_; }
    // Optionally remove hit with worst residual in track fit? (Only used by chi2 track fit).
    bool killTrackFitWorstHit() const { return killTrackFitWorstHit_; }
    // Cuts in standard deviations used to kill hits with big residuals during fit. If the residual exceeds the "General"
    // cut, the hit is killed providing it leaves the track with enough hits to survive. If the residual exceeds the
    // "Killing" cut, the hit is killed even if that kills the track.
    double generalResidualCut() const { return generalResidualCut_; }
    double killingResidualCut() const { return killingResidualCut_; }

    //--- Additional options for Davide Cieri's Simple Linear Regression track fitter ---

    // Digitize Simple Linear Regression variables & calculation. (Disabled if EnableDigitize=False).
    bool digitizeSLR() const { return digitizeSLR_; }
    /// Number of bits to be used in hardware to compute the division needed to calculate the helix parameters
    unsigned int dividerBitsHelix() const { return dividerBitsHelix_; }
    unsigned int dividerBitsHelixZ() const { return dividerBitsHelixZ_; }
    /// Number of bits to reduce the RPhi helix parameter denominator calculation weight
    unsigned int ShiftingBitsDenRPhi() const { return ShiftingBitsDenRPhi_; }

    /// Number of bits to reduce the RZ helix parameter denominator calculation weight
    unsigned int ShiftingBitsDenRZ() const { return ShiftingBitsDenRZ_; }
    /// Number of bits to reduce the qOverPt parameter numerator calculation weight
    unsigned int ShiftingBitsPt() const { return ShiftingBitsPt_; }
    /// Number of bits to reduce the PhiT parameter numerator calculation weight
    unsigned int ShiftingBitsPhi() const { return ShiftingBitsPhi_; }
    /// Number of bits to reduce the tanLambda parameter calculation weight
    unsigned int ShiftingBitsLambda() const { return ShiftingBitsLambda_; }
    /// Number of bits to reduce the tanLambda parameter calculation weight
    unsigned int ShiftingBitsZ0() const { return ShiftingBitsZ0_; }
    /// ChiSquare Cut
    double slr_chi2cut() const { return slr_chi2cut_; }
    /// Cut on RPhi Residual (radians)
    double ResidualCut() const { return residualCut_; }

    //--- Options for Kalman filter track fitters ---

    // Larger number has more debugging printout.
    unsigned kalmanDebugLevel() const { return kalmanDebugLevel_; }
    // Fit will reject fitted tracks unless it can assign at least this number of stubs to them.
    unsigned int kalmanMinNumStubs() const { return kalmanMinNumStubs_; }
    // Fit will attempt to add up to this nummber of stubs to each fitted tracks, but won't bother adding more.
    unsigned int kalmanMaxNumStubs() const { return kalmanMaxNumStubs_; }
    // For 5-param helix fits, calculate also beam-constrained helix params after fit is complete, & use them for duplicate removal if DupTrkAlgFit=1.
    bool kalmanAddBeamConstr() const { return kalmanAddBeamConstr_; }
    // Remove requirement of at least 2 PS layers per track.
    bool kalmanRemove2PScut() const { return kalmanRemove2PScut_; }
    // Allow the KF to skip this many layers in total per track for "hard" or "easy" input tracks
    unsigned int kalmanMaxSkipLayersHard() const { return kalmanMaxSkipLayersHard_; }
    unsigned int kalmanMaxSkipLayersEasy() const { return kalmanMaxSkipLayersEasy_; }
    // Max #stubs an input track can have to be defined "easy".
    unsigned int kalmanMaxStubsEasy() const { return kalmanMaxStubsEasy_; }
    // Enable "maybe layer"
    bool kfUseMaybeLayers() const { return kfUseMaybeLayers_; }
    // Cuts applied to KF states as a function of the last KF tracker layer they had a stub in.
    // (If "4" or "5" in name, cut only applies to 4 or 5 param helix fit).
    const std::vector<double>& kfLayerVsPtToler() const { return kfLayerVsPtToler_; }
    const std::vector<double>& kfLayerVsD0Cut5() const { return kfLayerVsD0Cut5_; }
    const std::vector<double>& kfLayerVsZ0Cut5() const { return kfLayerVsZ0Cut5_; }
    const std::vector<double>& kfLayerVsZ0Cut4() const { return kfLayerVsZ0Cut4_; }
    const std::vector<double>& kfLayerVsChiSq5() const { return kfLayerVsChiSq5_; }
    const std::vector<double>& kfLayerVsChiSq4() const { return kfLayerVsChiSq4_; }
    // KF will consider only this no. of stubs per layer.
    unsigned int kalmanMaxStubsPerLayer() const { return kalmanMaxStubsPerLayer_; }
    // Multiple scattering term - inflate hit phi errors by this divided by Pt
    double kalmanMultiScattTerm() const { return kalmanMultiScattTerm_; }
    // Scale down chi2 in r-phi plane by this factor to improve electron performance.
    unsigned int kalmanChi2RphiScale() const { return kalmanChi2RphiScale_; }
    //--- Enable Higher order corrections
    // Treat z uncertainty in tilted barrel modules correctly.
    bool kalmanHOtilted() const { return kalmanHOtilted_; }
    // Higher order circle explansion terms for low Pt.
    bool kalmanHOhelixExp() const { return kalmanHOhelixExp_; }
    // Alpha correction for non-radial 2S endcap strips. (0=disable correction, 1=correct with offset, 2=correct with non-diagonal stub covariance matrix).
    unsigned int kalmanHOalpha() const { return kalmanHOalpha_; }
    // Projection from (r,phi) to (z,phi) for endcap 2S modules. (0=disable correction, 1=correct with offset, 2=correct with non-diagonal stub covariance matrix).
    unsigned int kalmanHOprojZcorr() const { return kalmanHOprojZcorr_; }
    // Use approx calc to account for non-radial endcap 2S modules corresponding to current FW, with  no special treatment for tilted modules.
    bool kalmanHOfw() const { return kalmanHOfw_; }

    //=== Treatment of dead modules.
    //
    // Emulate dead/inefficient modules using the StubKiller code, with stubs killed according to the scenarios of the Stress Test group.
    // (0=Don't kill any stubs; 1-5 = Scenarios described in StubKiller.cc).
    unsigned int killScenario() const { return killScenario_; }
    // Modify TMTT tracking to try to recover tracking efficiency in presence of dead modules. (Does nothing if KillScenario = 0).
    bool killRecover() const { return killRecover_; }

    //=== Track fit digitisation configuration for various track fitters

    // These are used only for SimpleLR4 track fitter.
    bool slr_skipTrackDigi() const { return slr_skipTrackDigi_; }
    unsigned int slr_oneOver2rBits() const { return slr_oneOver2rBits_; }
    double slr_oneOver2rRange() const { return slr_oneOver2rRange_; }
    unsigned int slr_d0Bits() const { return slr_d0Bits_; }
    double slr_d0Range() const { return slr_d0Range_; }
    unsigned int slr_phi0Bits() const { return slr_phi0Bits_; }
    double slr_phi0Range() const { return slr_phi0Range_; }
    unsigned int slr_z0Bits() const { return slr_z0Bits_; }
    double slr_z0Range() const { return slr_z0Range_; }
    unsigned int slr_tanlambdaBits() const { return slr_tanlambdaBits_; }
    double slr_tanlambdaRange() const { return slr_tanlambdaRange_; }
    unsigned int slr_chisquaredBits() const { return slr_chisquaredBits_; }
    double slr_chisquaredRange() const { return slr_chisquaredRange_; }
    // These are used for KF track fitter and for all other track fitters (though are probably not right for other track fitters ...)
    bool kf_skipTrackDigi() const { return kf_skipTrackDigi_; }
    unsigned int kf_oneOver2rBits() const { return kf_oneOver2rBits_; }
    double kf_oneOver2rRange() const { return kf_oneOver2rRange_; }
    unsigned int kf_d0Bits() const { return kf_d0Bits_; }
    double kf_d0Range() const { return kf_d0Range_; }
    unsigned int kf_phi0Bits() const { return kf_phi0Bits_; }
    double kf_phi0Range() const { return kf_phi0Range_; }
    unsigned int kf_z0Bits() const { return kf_z0Bits_; }
    double kf_z0Range() const { return kf_z0Range_; }
    unsigned int kf_tanlambdaBits() const { return kf_tanlambdaBits_; }
    double kf_tanlambdaRange() const { return kf_tanlambdaRange_; }
    unsigned int kf_chisquaredBits() const { return kf_chisquaredBits_; }
    double kf_chisquaredRange() const { return kf_chisquaredRange_; }
    const std::vector<double>& kf_chisquaredBinEdges() const { return kf_chisquaredBinEdges_; }
    // Skip track digitisation when fitted is not SimpleLR4 or KF?
    bool other_skipTrackDigi() const { return other_skipTrackDigi_; }

    //=== Debug printout & plots

    // When making helix parameter resolution plots, only use particles from the physics event (True)
    // or also use particles from pileup (False) ?
    bool resPlotOpt() const { return resPlotOpt_; }

    // Booleain indicating if an output EDM file will be written.
    // N.B. This parameter does not appear inside TMTrackProducer_Defaults_cfi.py . It is created inside tmtt_tf_analysis_cfg.py .
    bool writeOutEdmFile() const { return writeOutEdmFile_; }

    //=== Hard-wired constants

    double cSpeed() const { return 1.0e8 * CLHEP::c_light; }  // Speed of light, with (mm/ns) to (cm/s)
    // B*c/1E11 - converts q/Pt to 1/radius_of_curvature
    double invPtToInvR() const { return (this->magneticField()) * (this->cSpeed()) / 1.0E13; }
    // B*c/2E11 - converts q/Pt to track angle at some radius from beamline.
    double invPtToDphi() const { return (this->magneticField()) * (this->cSpeed()) / 2.0E13; }
    //=== Set and get B-field value (mutable) in Tesla.
    // N.B. This must bet std::set for each run, and can't be initialized at the beginning of the job.
    void setMagneticField(float magneticField) const { magneticField_ = magneticField; }
    float magneticField() const {
      if (magneticField_ == 0.)
        throw cms::Exception("LogicError") << "Settings: You attempted to access the B field before it was initialized";
      return magneticField_;
    }

    //=== Settings used for HYBRID TRACKING code only.
    // Is hybrid tracking in use?
    bool hybrid() const { return hybrid_; }

  private:
    // Input tags for ES & ED data.
    const edm::ESInputTag magneticFieldInputTag_;
    const edm::ESInputTag trackerGeometryInputTag_;
    const edm::ESInputTag trackerTopologyInputTag_;
    const edm::ESInputTag ttStubAlgoInputTag_;

    const edm::InputTag stubInputTag_;
    const edm::InputTag tpInputTag_;
    const edm::InputTag stubTruthInputTag_;
    const edm::InputTag clusterTruthInputTag_;
    const edm::InputTag genJetInputTag_;

    // Parameter std::sets for differents types of configuration parameter.
    edm::ParameterSet genCuts_;
    edm::ParameterSet stubCuts_;
    edm::ParameterSet stubDigitize_;
    edm::ParameterSet trackerModuleType_;
    edm::ParameterSet geometricProc_;
    edm::ParameterSet phiSectors_;
    edm::ParameterSet etaSectors_;
    edm::ParameterSet htArraySpecRphi_;
    edm::ParameterSet htFillingRphi_;
    edm::ParameterSet rzFilterOpts_;
    edm::ParameterSet l1TrackDef_;
    edm::ParameterSet dupTrkRemoval_;
    edm::ParameterSet trackMatchDef_;
    edm::ParameterSet trackFitSettings_;
    edm::ParameterSet deadModuleOpts_;
    edm::ParameterSet trackDigi_;

    // General settings
    bool enableMCtruth_;
    bool enableHistos_;
    bool enableOutputIntermediateTTTracks_;

    // Cuts on truth tracking particles.
    double genMinPt_;
    double genMaxAbsEta_;
    double genMaxVertR_;
    double genMaxVertZ_;
    double genMaxD0_;
    double genMaxZ0_;
    std::vector<int> genPdgIds_;
    unsigned int genMinStubLayers_;

    // Cuts applied to stubs before arriving in L1 track finding board.
    unsigned int degradeBendRes_;
    double maxStubEta_;
    bool killLowPtStubs_;
    bool printStubWindows_;
    double bendCut_;
    double bendCutExtra_;
    bool orderStubsByBend_;

    // Optional stub digitization.
    bool enableDigitize_;
    unsigned int phiSectorBits_;
    unsigned int phiSBits_;
    double phiSRange_;
    unsigned int rtBits_;
    double rtRange_;
    unsigned int zBits_;
    double zRange_;
    unsigned int phiNBits_;
    double phiNRange_;
    unsigned int bendBits_;

    // Tracker module type for FW.
    std::vector<double> pitchVsType_;
    std::vector<double> spaceVsType_;
    std::vector<bool> barrelVsType_;
    std::vector<bool> psVsType_;
    std::vector<bool> tiltedVsType_;
    std::vector<unsigned int> barrelVsTypeTmp_;
    std::vector<unsigned int> psVsTypeTmp_;
    std::vector<unsigned int> tiltedVsTypeTmp_;

    // Configuration of Geometric Processor.
    bool useApproxB_;
    double bApprox_gradient_;
    double bApprox_intercept_;

    // Definition of phi sectors.
    unsigned int numPhiNonants_;
    unsigned int numPhiSectors_;
    double chosenRofPhi_;
    bool useStubPhi_;
    bool useStubPhiTrk_;
    double assumedPhiTrkRes_;
    bool calcPhiTrkRes_;

    // Definition of eta sectors.
    std::vector<double> etaRegions_;
    double chosenRofZ_;
    double beamWindowZ_;
    bool allowOver2EtaSecs_;

    // r-phi Hough transform array specifications.
    double houghMinPt_;
    unsigned int houghNbinsPt_;
    unsigned int houghNbinsPhi_;
    bool enableMerge2x2_;
    double maxPtToMerge2x2_;
    unsigned int numSubSecsEta_;
    unsigned int shape_;
    bool miniHTstage_;
    unsigned int miniHoughNbinsPt_;
    unsigned int miniHoughNbinsPhi_;
    double miniHoughMinPt_;
    bool miniHoughDontKill_;
    double miniHoughDontKillMinPt_;
    unsigned int miniHoughLoadBalance_;

    // Rules governing how stubs are filled into the r-phi Hough Transform array.
    unsigned int killSomeHTCellsRphi_;
    bool useBendFilter_;
    unsigned int maxStubsInCell_;
    unsigned int maxStubsInCellMiniHough_;
    bool busySectorKill_;
    unsigned int busySectorNumStubs_;
    std::vector<unsigned int> busySectorMbinRanges_;
    std::vector<unsigned int> busySectorMbinOrder_;
    bool busyInputSectorKill_;
    unsigned int busyInputSectorNumStubs_;
    unsigned int muxOutputsHT_;
    std::vector<unsigned int> etaRegWhitelist_;

    // Options controlling r-z track filters (or any other track filters run after the Hough transform, as opposed to inside it).
    std::string rzFilterName_;
    double seedResCut_;
    bool keepAllSeed_;
    unsigned int maxSeedCombinations_;
    unsigned int maxGoodSeedCombinations_;
    unsigned int maxSeedsPerStub_;
    bool zTrkSectorCheck_;
    unsigned int minFilterLayers_;

    // Rules for deciding when the track-finding has found an L1 track candidate
    unsigned int minStubLayers_;
    double minPtToReduceLayers_;
    std::vector<unsigned int> etaSecsReduceLayers_;
    bool reduceLayerID_;

    // Specification of algorithm to eliminate duplicate tracks
    unsigned int dupTrkAlgFit_;

    // Rules for deciding when a reconstructed L1 track matches a MC truth particle (i.e. tracking particle).
    double minFracMatchStubsOnReco_;
    double minFracMatchStubsOnTP_;
    unsigned int minNumMatchLayers_;
    unsigned int minNumMatchPSLayers_;
    bool stubMatchStrict_;

    // Track Fitting Settings
    std::vector<std::string> trackFitters_;
    std::vector<std::string> useRZfilter_;
    double chi2OverNdfCut_;
    bool detailedFitOutput_;
    bool trackFitCheat_;
    //
    unsigned int numTrackFitIterations_;
    bool killTrackFitWorstHit_;
    double generalResidualCut_;
    double killingResidualCut_;
    //
    bool digitizeSLR_;
    unsigned int dividerBitsHelix_;
    unsigned int dividerBitsHelixZ_;
    unsigned int ShiftingBitsDenRPhi_;
    unsigned int ShiftingBitsDenRZ_;
    unsigned int ShiftingBitsPt_;
    unsigned int ShiftingBitsPhi_;

    unsigned int ShiftingBitsLambda_;
    unsigned int ShiftingBitsZ0_;
    double slr_chi2cut_;
    double residualCut_;
    //
    unsigned kalmanDebugLevel_;
    unsigned int kalmanMinNumStubs_;
    unsigned int kalmanMaxNumStubs_;
    bool kalmanAddBeamConstr_;
    bool kalmanRemove2PScut_;
    unsigned int kalmanMaxSkipLayersHard_;
    unsigned int kalmanMaxSkipLayersEasy_;
    unsigned int kalmanMaxStubsEasy_;
    bool kfUseMaybeLayers_;

    std::vector<double> kfLayerVsPtToler_;
    std::vector<double> kfLayerVsD0Cut5_;
    std::vector<double> kfLayerVsZ0Cut5_;
    std::vector<double> kfLayerVsZ0Cut4_;
    std::vector<double> kfLayerVsChiSq5_;
    std::vector<double> kfLayerVsChiSq4_;

    unsigned int kalmanMaxStubsPerLayer_;
    double kalmanMultiScattTerm_;
    unsigned int kalmanChi2RphiScale_;
    bool kalmanHOtilted_;
    bool kalmanHOhelixExp_;
    unsigned int kalmanHOalpha_;
    unsigned int kalmanHOprojZcorr_;
    bool kalmanHOfw_;

    // Treatment of dead modules.
    unsigned int killScenario_;
    bool killRecover_;

    // Track digitisation configuration for various track fitters
    bool slr_skipTrackDigi_;
    unsigned int slr_oneOver2rBits_;
    double slr_oneOver2rRange_;
    double slr_oneOver2rMult_;
    unsigned int slr_d0Bits_;
    double slr_d0Range_;
    unsigned int slr_phi0Bits_;
    double slr_phi0Range_;
    unsigned int slr_z0Bits_;
    double slr_z0Range_;
    unsigned int slr_tanlambdaBits_;
    double slr_tanlambdaRange_;
    unsigned int slr_chisquaredBits_;
    double slr_chisquaredRange_;
    //
    bool kf_skipTrackDigi_;
    unsigned int kf_oneOver2rBits_;
    double kf_oneOver2rRange_;
    double kf_oneOver2rMult_;
    unsigned int kf_d0Bits_;
    double kf_d0Range_;
    unsigned int kf_phi0Bits_;
    double kf_phi0Range_;
    unsigned int kf_z0Bits_;
    double kf_z0Range_;
    unsigned int kf_tanlambdaBits_;
    double kf_tanlambdaRange_;
    unsigned int kf_chisquaredBits_;
    double kf_chisquaredRange_;
    std::vector<double> kf_chisquaredBinEdges_;
    //
    bool other_skipTrackDigi_;

    // Debug printout
    bool resPlotOpt_;

    // Boolean indicating an an EDM output file will be written.
    bool writeOutEdmFile_;

    // B-field in Tesla
    mutable std::atomic<float> magneticField_;

    // Hybrid tracking
    bool hybrid_;
  };

}  // namespace tmtt

#endif
