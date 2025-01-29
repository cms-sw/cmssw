#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/ESGetToken.h"
#include "L1Trigger/TrackTrigger/interface/Setup.h"

#include <memory>

using namespace std;
using namespace edm;

namespace tt {

  /*! \class  tt::ProducerSetup
   *  \brief  Class to produce setup of Track Trigger emulators
   *  \author Thomas Schuh
   *  \date   2020, Apr
   */
  class ProducerSetup : public ESProducer {
  public:
    ProducerSetup(const ParameterSet& iConfig);
    ~ProducerSetup() override {}
    unique_ptr<Setup> produce(const SetupRcd& setupRcd);

  private:
    Setup::Config iConfig_;
    ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> getTokenTrackerGeometry_;
    ESGetToken<TrackerTopology, TrackerTopologyRcd> getTokenTrackerTopology_;
    ESGetToken<TrackerDetToDTCELinkCablingMap, TrackerDetToDTCELinkCablingMapRcd> getTokenCablingMap_;
    ESGetToken<StubAlgorithm, TTStubAlgorithmRecord> getTokenTTStubAlgorithm_;
  };

  ProducerSetup::ProducerSetup(const ParameterSet& iConfig) {
    auto cc = setWhatProduced(this);
    getTokenTrackerGeometry_ = cc.consumes();
    getTokenTrackerTopology_ = cc.consumes();
    getTokenCablingMap_ = cc.consumes();
    getTokenTTStubAlgorithm_ = cc.consumes();
    const ParameterSet& pSetTF = iConfig.getParameter<ParameterSet>("TrackFinding");
    iConfig_.beamWindowZ_ = pSetTF.getParameter<double>("BeamWindowZ");
    iConfig_.minPt_ = pSetTF.getParameter<double>("MinPt");
    iConfig_.minPtCand_ = pSetTF.getParameter<double>("MinPtCand");
    iConfig_.maxEta_ = pSetTF.getParameter<double>("MaxEta");
    iConfig_.maxD0_ = pSetTF.getParameter<double>("MaxD0");
    iConfig_.chosenRofPhi_ = pSetTF.getParameter<double>("ChosenRofPhi");
    iConfig_.numLayers_ = pSetTF.getParameter<int>("NumLayers");
    iConfig_.minLayers_ = pSetTF.getParameter<int>("MinLayers");
    const ParameterSet& pSetTMTT = iConfig.getParameter<ParameterSet>("TMTT");
    iConfig_.tmttWidthR_ = pSetTMTT.getParameter<int>("WidthR");
    iConfig_.tmttWidthPhi_ = pSetTMTT.getParameter<int>("WidthPhi");
    iConfig_.tmttWidthZ_ = pSetTMTT.getParameter<int>("WidthZ");
    const ParameterSet& pSetHybrid = iConfig.getParameter<ParameterSet>("Hybrid");
    iConfig_.hybridNumLayers_ = pSetHybrid.getParameter<int>("NumLayers");
    iConfig_.hybridNumRingsPS_ = pSetHybrid.getParameter<vector<int>>("NumRingsPS");
    iConfig_.hybridWidthsR_ = pSetHybrid.getParameter<vector<int>>("WidthsR");
    iConfig_.hybridWidthsZ_ = pSetHybrid.getParameter<vector<int>>("WidthsZ");
    iConfig_.hybridWidthsPhi_ = pSetHybrid.getParameter<vector<int>>("WidthsPhi");
    iConfig_.hybridWidthsAlpha_ = pSetHybrid.getParameter<vector<int>>("WidthsAlpha");
    iConfig_.hybridWidthsBend_ = pSetHybrid.getParameter<vector<int>>("WidthsBend");
    iConfig_.hybridRangesR_ = pSetHybrid.getParameter<vector<double>>("RangesR");
    iConfig_.hybridRangesZ_ = pSetHybrid.getParameter<vector<double>>("RangesZ");
    iConfig_.hybridRangesAlpha_ = pSetHybrid.getParameter<vector<double>>("RangesAlpha");
    iConfig_.hybridLayerRs_ = pSetHybrid.getParameter<vector<double>>("LayerRs");
    iConfig_.hybridDiskZs_ = pSetHybrid.getParameter<vector<double>>("DiskZs");
    iConfig_.hybridDisk2SRsSet_ = pSetHybrid.getParameter<vector<ParameterSet>>("Disk2SRsSet");
    iConfig_.tbBarrelHalfLength_ = pSetHybrid.getParameter<double>("BarrelHalfLength");
    iConfig_.tbInnerRadius_ = pSetHybrid.getParameter<double>("InnerRadius");
    iConfig_.tbWidthsR_ = pSetHybrid.getParameter<vector<int>>("WidthsRTB");
    const ParameterSet& pSetFW = iConfig.getParameter<ParameterSet>("Firmware");
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
    const ParameterSet& pSetOT = iConfig.getParameter<ParameterSet>("Tracker");
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
    iConfig_.limitsTiltedR_ = pSetOT.getParameter<vector<double>>("LimitsTiltedR");
    iConfig_.limitsTiltedZ_ = pSetOT.getParameter<vector<double>>("LimitsTiltedZ");
    iConfig_.limitsPSDiksZ_ = pSetOT.getParameter<vector<double>>("LimitsPSDiksZ");
    iConfig_.limitsPSDiksR_ = pSetOT.getParameter<vector<double>>("LimitsPSDiksR");
    iConfig_.tiltedLayerLimitsZ_ = pSetOT.getParameter<vector<double>>("TiltedLayerLimitsZ");
    iConfig_.psDiskLimitsR_ = pSetOT.getParameter<vector<double>>("PSDiskLimitsR");
    const ParameterSet& pSetFE = iConfig.getParameter<ParameterSet>("FrontEnd");
    iConfig_.widthBend_ = pSetFE.getParameter<int>("WidthBend");
    iConfig_.widthCol_ = pSetFE.getParameter<int>("WidthCol");
    iConfig_.widthRow_ = pSetFE.getParameter<int>("WidthRow");
    iConfig_.baseBend_ = pSetFE.getParameter<double>("BaseBend");
    iConfig_.baseCol_ = pSetFE.getParameter<double>("BaseCol");
    iConfig_.baseRow_ = pSetFE.getParameter<double>("BaseRow");
    iConfig_.baseWindowSize_ = pSetFE.getParameter<double>("BaseWindowSize");
    iConfig_.bendCut_ = pSetFE.getParameter<double>("BendCut");
    const ParameterSet& pSetDTC = iConfig.getParameter<ParameterSet>("DTC");
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
    const ParameterSet& pSetTFP = iConfig.getParameter<ParameterSet>("TFP");
    iConfig_.tfpWidthPhi0_ = pSetTFP.getParameter<int>("WidthPhi0");
    iConfig_.tfpWidthInvR_ = pSetTFP.getParameter<int>("WidthInvR");
    iConfig_.tfpWidthCot_ = pSetTFP.getParameter<int>("WidthCot");
    iConfig_.tfpWidthZ0_ = pSetTFP.getParameter<int>("WidthZ0");
    iConfig_.tfpNumChannel_ = pSetTFP.getParameter<int>("NumChannel");
    const ParameterSet& pSetGP = iConfig.getParameter<ParameterSet>("GeometricProcessor");
    iConfig_.gpNumBinsPhiT_ = pSetGP.getParameter<int>("NumBinsPhiT");
    iConfig_.gpNumBinsZT_ = pSetGP.getParameter<int>("NumBinsZT");
    iConfig_.chosenRofZ_ = pSetGP.getParameter<double>("ChosenRofZ");
    iConfig_.gpDepthMemory_ = pSetGP.getParameter<int>("DepthMemory");
    iConfig_.gpWidthModule_ = pSetGP.getParameter<int>("WidthModule");
    iConfig_.gpPosPS_ = pSetGP.getParameter<int>("PosPS");
    iConfig_.gpPosBarrel_ = pSetGP.getParameter<int>("PosBarrel");
    iConfig_.gpPosTilted_ = pSetGP.getParameter<int>("PosTilted");
    const ParameterSet& pSetHT = iConfig.getParameter<ParameterSet>("HoughTransform");
    iConfig_.htNumBinsInv2R_ = pSetHT.getParameter<int>("NumBinsInv2R");
    iConfig_.htNumBinsPhiT_ = pSetHT.getParameter<int>("NumBinsPhiT");
    iConfig_.htMinLayers_ = pSetHT.getParameter<int>("MinLayers");
    iConfig_.htDepthMemory_ = pSetHT.getParameter<int>("DepthMemory");
    const ParameterSet& pSetCTB = iConfig.getParameter<ParameterSet>("CleanTrackBuilder");
    iConfig_.ctbNumBinsInv2R_ = pSetCTB.getParameter<int>("NumBinsInv2R");
    iConfig_.ctbNumBinsPhiT_ = pSetCTB.getParameter<int>("NumBinsPhiT");
    iConfig_.ctbNumBinsCot_ = pSetCTB.getParameter<int>("NumBinsCot");
    iConfig_.ctbNumBinsZT_ = pSetCTB.getParameter<int>("NumBinsZT");
    iConfig_.ctbMinLayers_ = pSetCTB.getParameter<int>("MinLayers");
    iConfig_.ctbMaxTracks_ = pSetCTB.getParameter<int>("MaxTracks");
    iConfig_.ctbMaxStubs_ = pSetCTB.getParameter<int>("MaxStubs");
    iConfig_.ctbDepthMemory_ = pSetCTB.getParameter<int>("DepthMemory");
    const ParameterSet& pSetKF = iConfig.getParameter<ParameterSet>("KalmanFilter");
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
    const ParameterSet& pSetDR = iConfig.getParameter<ParameterSet>("DuplicateRemoval");
    iConfig_.drDepthMemory_ = pSetDR.getParameter<int>("DepthMemory");
    const ParameterSet& pSetTQ = iConfig.getParameter<ParameterSet>("TrackQuality");
    iConfig_.tqNumChannel_ = pSetTQ.getParameter<int>("NumChannel");
  }

  unique_ptr<Setup> ProducerSetup::produce(const SetupRcd& setupRcd) {
    const TrackerGeometry& trackerGeometry = setupRcd.get(getTokenTrackerGeometry_);
    const TrackerTopology& trackerTopology = setupRcd.get(getTokenTrackerTopology_);
    const TrackerDetToDTCELinkCablingMap& cablingMap = setupRcd.get(getTokenCablingMap_);
    const ESHandle<StubAlgorithm> handleStubAlgorithm = setupRcd.getHandle(getTokenTTStubAlgorithm_);
    const StubAlgorithmOfficial& stubAlgoritm =
        *dynamic_cast<const StubAlgorithmOfficial*>(&setupRcd.get(getTokenTTStubAlgorithm_));
    const ParameterSet& pSetStubAlgorithm = getParameterSet(handleStubAlgorithm.description()->pid_);
    return make_unique<Setup>(iConfig_, trackerGeometry, trackerTopology, cablingMap, stubAlgoritm, pSetStubAlgorithm);
  }
}  // namespace tt

DEFINE_FWK_EVENTSETUP_MODULE(tt::ProducerSetup);
