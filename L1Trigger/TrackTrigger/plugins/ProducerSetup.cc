#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <memory>

namespace tt {

  /*! \class  tt::ProducerSetup
   *  \brief  Class to produce setup of Track Trigger emulators
   *  \author Thomas Schuh
   *  \date   2020, Apr
   */
  class ProducerSetup : public edm::ESProducer {
  public:
    ProducerSetup(const edm::ParameterSet& iConfig);
    ~ProducerSetup() override {}
    std::unique_ptr<Setup> produce(const SetupRcd& setupRcd);

  private:
    Setup::Config iConfig_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> getTokenTrackerGeometry_;
    edm::ESGetToken<TrackerTopology, TrackerTopologyRcd> getTokenTrackerTopology_;
    edm::ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> getTokenCablingMap_;
    edm::ESGetToken<StubAlgorithm, TTStubAlgorithmRecord> getTokenTTStubAlgorithm_;
  };

  ProducerSetup::ProducerSetup(const edm::ParameterSet& iConfig) {
    auto cc = setWhatProduced(this);
    getTokenTrackerGeometry_ = cc.consumes();
    getTokenTrackerTopology_ = cc.consumes();
    getTokenCablingMap_ = cc.consumes();
    getTokenTTStubAlgorithm_ = cc.consumes();
    const edm::ParameterSet& pSetTF = iConfig.getParameter<edm::ParameterSet>("TrackFinding");
    iConfig_.beamWindowZ_ = pSetTF.getParameter<double>("BeamWindowZ");
    iConfig_.minPt_ = pSetTF.getParameter<double>("MinPt");
    iConfig_.minPtCand_ = pSetTF.getParameter<double>("MinPtCand");
    iConfig_.maxEta_ = pSetTF.getParameter<double>("MaxEta");
    iConfig_.maxD0_ = pSetTF.getParameter<double>("MaxD0");
    iConfig_.chosenRofPhi_ = pSetTF.getParameter<double>("ChosenRofPhi");
    iConfig_.numLayers_ = pSetTF.getParameter<int>("NumLayers");
    iConfig_.minLayers_ = pSetTF.getParameter<int>("MinLayers");
    const edm::ParameterSet& pSetTMTT = iConfig.getParameter<edm::ParameterSet>("TMTT");
    iConfig_.tmttWidthR_ = pSetTMTT.getParameter<int>("WidthR");
    iConfig_.tmttWidthPhi_ = pSetTMTT.getParameter<int>("WidthPhi");
    iConfig_.tmttWidthZ_ = pSetTMTT.getParameter<int>("WidthZ");
    const edm::ParameterSet& pSetHybrid = iConfig.getParameter<edm::ParameterSet>("Hybrid");
    iConfig_.hybridNumLayers_ = pSetHybrid.getParameter<int>("NumLayers");
    iConfig_.hybridNumRingsPS_ = pSetHybrid.getParameter<std::vector<int>>("NumRingsPS");
    iConfig_.hybridWidthsR_ = pSetHybrid.getParameter<std::vector<int>>("WidthsR");
    iConfig_.hybridWidthsZ_ = pSetHybrid.getParameter<std::vector<int>>("WidthsZ");
    iConfig_.hybridWidthsPhi_ = pSetHybrid.getParameter<std::vector<int>>("WidthsPhi");
    iConfig_.hybridWidthsAlpha_ = pSetHybrid.getParameter<std::vector<int>>("WidthsAlpha");
    iConfig_.hybridWidthsBend_ = pSetHybrid.getParameter<std::vector<int>>("WidthsBend");
    iConfig_.hybridRangesR_ = pSetHybrid.getParameter<std::vector<double>>("RangesR");
    iConfig_.hybridRangesZ_ = pSetHybrid.getParameter<std::vector<double>>("RangesZ");
    iConfig_.hybridRangesAlpha_ = pSetHybrid.getParameter<std::vector<double>>("RangesAlpha");
    iConfig_.hybridLayerRs_ = pSetHybrid.getParameter<std::vector<double>>("LayerRs");
    iConfig_.hybridDiskZs_ = pSetHybrid.getParameter<std::vector<double>>("DiskZs");
    iConfig_.hybridDisk2SRsSet_ = pSetHybrid.getParameter<std::vector<edm::ParameterSet>>("Disk2SRsSet");
    iConfig_.tbBarrelHalfLength_ = pSetHybrid.getParameter<double>("BarrelHalfLength");
    iConfig_.tbInnerRadius_ = pSetHybrid.getParameter<double>("InnerRadius");
    iConfig_.tbWidthsR_ = pSetHybrid.getParameter<std::vector<int>>("WidthsRTB");
    const edm::ParameterSet& pSetFW = iConfig.getParameter<edm::ParameterSet>("Firmware");
    iConfig_.enableTruncation_ = pSetFW.getParameter<bool>("EnableTruncation");
    iConfig_.widthDSPa_ = pSetFW.getParameter<int>("WidthDSPa");
    iConfig_.widthDSPb_ = pSetFW.getParameter<int>("WidthDSPb");
    iConfig_.widthDSPc_ = pSetFW.getParameter<int>("WidthDSPc");
    iConfig_.widthAddrBRAM36_ = pSetFW.getParameter<int>("WidthAddrBRAM36");
    iConfig_.widthAddrBRAM18_ = pSetFW.getParameter<int>("WidthAddrBRAM18");
    iConfig_.numFramesInfra_ = pSetFW.getParameter<int>("NumFramesInfra");
    iConfig_.freqLHC_ = pSetFW.getParameter<double>("FreqLHC");
    iConfig_.freqBEHigh_ = pSetFW.getParameter<double>("FreqBEHigh");
    iConfig_.freqBELow_ = pSetFW.getParameter<double>("FreqBELow");
    iConfig_.tmpFE_ = pSetFW.getParameter<int>("TMP_FE");
    iConfig_.tmpTFP_ = pSetFW.getParameter<int>("TMP_TFP");
    iConfig_.speedOfLight_ = pSetFW.getParameter<double>("SpeedOfLight");
    const edm::ParameterSet& pSetOT = iConfig.getParameter<edm::ParameterSet>("Tracker");
    iConfig_.bField_ = pSetOT.getParameter<double>("BField");
    iConfig_.bFieldError_ = pSetOT.getParameter<double>("BFieldError");
    iConfig_.outerRadius_ = pSetOT.getParameter<double>("OuterRadius");
    iConfig_.innerRadius_ = pSetOT.getParameter<double>("InnerRadius");
    iConfig_.halfLength_ = pSetOT.getParameter<double>("HalfLength");
    iConfig_.tiltApproxSlope_ = pSetOT.getParameter<double>("TiltApproxSlope");
    iConfig_.tiltApproxIntercept_ = pSetOT.getParameter<double>("TiltApproxIntercept");
    iConfig_.tiltUncertaintyR_ = pSetOT.getParameter<double>("TiltUncertaintyR");
    iConfig_.scattering_ = pSetOT.getParameter<double>("Scattering");
    iConfig_.pitchRow2S_ = pSetOT.getParameter<double>("PitchRow2S");
    iConfig_.pitchRowPS_ = pSetOT.getParameter<double>("PitchRowPS");
    iConfig_.pitchCol2S_ = pSetOT.getParameter<double>("PitchCol2S");
    iConfig_.pitchColPS_ = pSetOT.getParameter<double>("PitchColPS");
    iConfig_.limitPSBarrel_ = pSetOT.getParameter<double>("LimitPSBarrel");
    iConfig_.limitsTiltedR_ = pSetOT.getParameter<std::vector<double>>("LimitsTiltedR");
    iConfig_.limitsTiltedZ_ = pSetOT.getParameter<std::vector<double>>("LimitsTiltedZ");
    iConfig_.limitsPSDiksZ_ = pSetOT.getParameter<std::vector<double>>("LimitsPSDiksZ");
    iConfig_.limitsPSDiksR_ = pSetOT.getParameter<std::vector<double>>("LimitsPSDiksR");
    iConfig_.tiltedLayerLimitsZ_ = pSetOT.getParameter<std::vector<double>>("TiltedLayerLimitsZ");
    iConfig_.psDiskLimitsR_ = pSetOT.getParameter<std::vector<double>>("PSDiskLimitsR");
    const edm::ParameterSet& pSetFE = iConfig.getParameter<edm::ParameterSet>("FrontEnd");
    iConfig_.widthBend_ = pSetFE.getParameter<int>("WidthBend");
    iConfig_.widthCol_ = pSetFE.getParameter<int>("WidthCol");
    iConfig_.widthRow_ = pSetFE.getParameter<int>("WidthRow");
    iConfig_.baseBend_ = pSetFE.getParameter<double>("BaseBend");
    iConfig_.baseCol_ = pSetFE.getParameter<double>("BaseCol");
    iConfig_.baseRow_ = pSetFE.getParameter<double>("BaseRow");
    iConfig_.baseWindowSize_ = pSetFE.getParameter<double>("BaseWindowSize");
    iConfig_.bendCut_ = pSetFE.getParameter<double>("BendCut");
    const edm::ParameterSet& pSetDTC = iConfig.getParameter<edm::ParameterSet>("DTC");
    iConfig_.numRegions_ = pSetDTC.getParameter<int>("NumRegions");
    iConfig_.numOverlappingRegions_ = pSetDTC.getParameter<int>("NumOverlappingRegions");
    iConfig_.numATCASlots_ = pSetDTC.getParameter<int>("NumATCASlots");
    iConfig_.numDTCsPerRegion_ = pSetDTC.getParameter<int>("NumDTCsPerRegion");
    iConfig_.numModulesPerDTC_ = pSetDTC.getParameter<int>("NumModulesPerDTC");
    iConfig_.dtcNumRoutingBlocks_ = pSetDTC.getParameter<int>("NumRoutingBlocks");
    iConfig_.dtcDepthMemory_ = pSetDTC.getParameter<int>("DepthMemory");
    iConfig_.dtcWidthRowLUT_ = pSetDTC.getParameter<int>("WidthRowLUT");
    iConfig_.dtcWidthInv2R_ = pSetDTC.getParameter<int>("WidthInv2R");
    iConfig_.offsetDetIdDSV_ = pSetDTC.getParameter<int>("OffsetDetIdDSV");
    iConfig_.offsetDetIdTP_ = pSetDTC.getParameter<int>("OffsetDetIdTP");
    iConfig_.offsetLayerDisks_ = pSetDTC.getParameter<int>("OffsetLayerDisks");
    iConfig_.offsetLayerId_ = pSetDTC.getParameter<int>("OffsetLayerId");
    iConfig_.numBarrelLayer_ = pSetDTC.getParameter<int>("NumBarrelLayer");
    iConfig_.slotLimitPS_ = pSetDTC.getParameter<int>("SlotLimitPS");
    iConfig_.slotLimit10gbps_ = pSetDTC.getParameter<int>("SlotLimit10gbps");
    const edm::ParameterSet& pSetTFP = iConfig.getParameter<edm::ParameterSet>("TFP");
    iConfig_.tfpWidthPhi0_ = pSetTFP.getParameter<int>("WidthPhi0");
    iConfig_.tfpWidthInvR_ = pSetTFP.getParameter<int>("WidthInvR");
    iConfig_.tfpWidthCot_ = pSetTFP.getParameter<int>("WidthCot");
    iConfig_.tfpWidthZ0_ = pSetTFP.getParameter<int>("WidthZ0");
    iConfig_.tfpNumChannel_ = pSetTFP.getParameter<int>("NumChannel");
    const edm::ParameterSet& pSetGP = iConfig.getParameter<edm::ParameterSet>("GeometricProcessor");
    iConfig_.gpNumBinsPhiT_ = pSetGP.getParameter<int>("NumBinsPhiT");
    iConfig_.gpNumBinsZT_ = pSetGP.getParameter<int>("NumBinsZT");
    iConfig_.chosenRofZ_ = pSetGP.getParameter<double>("ChosenRofZ");
    iConfig_.gpDepthMemory_ = pSetGP.getParameter<int>("DepthMemory");
    iConfig_.gpWidthModule_ = pSetGP.getParameter<int>("WidthModule");
    iConfig_.gpPosPS_ = pSetGP.getParameter<int>("PosPS");
    iConfig_.gpPosBarrel_ = pSetGP.getParameter<int>("PosBarrel");
    iConfig_.gpPosTilted_ = pSetGP.getParameter<int>("PosTilted");
    const edm::ParameterSet& pSetHT = iConfig.getParameter<edm::ParameterSet>("HoughTransform");
    iConfig_.htNumBinsInv2R_ = pSetHT.getParameter<int>("NumBinsInv2R");
    iConfig_.htNumBinsPhiT_ = pSetHT.getParameter<int>("NumBinsPhiT");
    iConfig_.htMinLayers_ = pSetHT.getParameter<int>("MinLayers");
    iConfig_.htDepthMemory_ = pSetHT.getParameter<int>("DepthMemory");
    const edm::ParameterSet& pSetCTB = iConfig.getParameter<edm::ParameterSet>("CleanTrackBuilder");
    iConfig_.ctbNumBinsInv2R_ = pSetCTB.getParameter<int>("NumBinsInv2R");
    iConfig_.ctbNumBinsPhiT_ = pSetCTB.getParameter<int>("NumBinsPhiT");
    iConfig_.ctbNumBinsCot_ = pSetCTB.getParameter<int>("NumBinsCot");
    iConfig_.ctbNumBinsZT_ = pSetCTB.getParameter<int>("NumBinsZT");
    iConfig_.ctbMinLayers_ = pSetCTB.getParameter<int>("MinLayers");
    iConfig_.ctbMaxTracks_ = pSetCTB.getParameter<int>("MaxTracks");
    iConfig_.ctbMaxStubs_ = pSetCTB.getParameter<int>("MaxStubs");
    iConfig_.ctbDepthMemory_ = pSetCTB.getParameter<int>("DepthMemory");
    const edm::ParameterSet& pSetKF = iConfig.getParameter<edm::ParameterSet>("KalmanFilter");
    iConfig_.kfUse5ParameterFit_ = pSetKF.getParameter<bool>("Use5ParameterFit");
    iConfig_.kfUseSimmulation_ = pSetKF.getParameter<bool>("UseSimmulation");
    iConfig_.kfUseTTStubResiduals_ = pSetKF.getParameter<bool>("UseTTStubResiduals");
    iConfig_.kfUseTTStubParameters_ = pSetKF.getParameter<bool>("UseTTStubParameters");
    iConfig_.kfApplyNonLinearCorrection_ = pSetKF.getParameter<bool>("ApplyNonLinearCorrection");
    iConfig_.kfNumWorker_ = pSetKF.getParameter<int>("NumWorker");
    iConfig_.kfMaxTracks_ = pSetKF.getParameter<int>("MaxTracks");
    iConfig_.kfMinLayers_ = pSetKF.getParameter<int>("MinLayers");
    iConfig_.kfMinLayersPS_ = pSetKF.getParameter<int>("MinLayersPS");
    iConfig_.kfMaxLayers_ = pSetKF.getParameter<int>("MaxLayers");
    iConfig_.kfMaxGaps_ = pSetKF.getParameter<int>("MaxGaps");
    iConfig_.kfMaxSeedingLayer_ = pSetKF.getParameter<int>("MaxSeedingLayer");
    iConfig_.kfNumSeedStubs_ = pSetKF.getParameter<int>("NumSeedStubs");
    iConfig_.kfMinSeedDeltaR_ = pSetKF.getParameter<double>("MinSeedDeltaR");
    iConfig_.kfRangeFactor_ = pSetKF.getParameter<double>("RangeFactor");
    iConfig_.kfShiftInitialC00_ = pSetKF.getParameter<int>("ShiftInitialC00");
    iConfig_.kfShiftInitialC11_ = pSetKF.getParameter<int>("ShiftInitialC11");
    iConfig_.kfShiftInitialC22_ = pSetKF.getParameter<int>("ShiftInitialC22");
    iConfig_.kfShiftInitialC33_ = pSetKF.getParameter<int>("ShiftInitialC33");
    iConfig_.kfShiftChi20_ = pSetKF.getParameter<int>("ShiftChi20");
    iConfig_.kfShiftChi21_ = pSetKF.getParameter<int>("ShiftChi21");
    iConfig_.kfCutChi2_ = pSetKF.getParameter<double>("CutChi2");
    iConfig_.kfWidthChi2_ = pSetKF.getParameter<int>("WidthChi2");
    const edm::ParameterSet& pSetDR = iConfig.getParameter<edm::ParameterSet>("DuplicateRemoval");
    iConfig_.drDepthMemory_ = pSetDR.getParameter<int>("DepthMemory");
    const edm::ParameterSet& pSetTQ = iConfig.getParameter<edm::ParameterSet>("TrackQuality");
    iConfig_.tqNumChannel_ = pSetTQ.getParameter<int>("NumChannel");
  }

  std::unique_ptr<Setup> ProducerSetup::produce(const SetupRcd& setupRcd) {
    const TrackerGeometry& trackerGeometry = setupRcd.get(getTokenTrackerGeometry_);
    const TrackerTopology& trackerTopology = setupRcd.get(getTokenTrackerTopology_);
    const TrackerDetToDTCELinkCablingMap& cablingMap = setupRcd.get(getTokenCablingMap_);
    const edm::ESHandle<StubAlgorithm> handleStubAlgorithm = setupRcd.getHandle(getTokenTTStubAlgorithm_);
    const StubAlgorithmOfficial& stubAlgoritm =
        *dynamic_cast<const StubAlgorithmOfficial*>(&setupRcd.get(getTokenTTStubAlgorithm_));
    const edm::ParameterSet& pSetStubAlgorithm = getParameterSet(handleStubAlgorithm.description()->pid_);
    return std::make_unique<Setup>(
        iConfig_, trackerGeometry, trackerTopology, cablingMap, stubAlgoritm, pSetStubAlgorithm);
  }
}  // namespace tt

DEFINE_FWK_EVENTSETUP_MODULE(tt::ProducerSetup);
