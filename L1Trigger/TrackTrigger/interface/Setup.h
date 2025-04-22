#ifndef L1Trigger_TrackTrigger_Setup_h
#define L1Trigger_TrackTrigger_Setup_h

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_official.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "SimTracker/Common/interface/TrackingParticleSelector.h"

#include "SimDataFormats/Associations/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "L1Trigger/TrackTrigger/interface/SensorModule.h"
#include "L1Trigger/TrackTrigger/interface/SetupRcd.h"

#include <vector>
#include <set>
#include <unordered_map>

namespace tt {

  typedef TTStubAlgorithm<Ref_Phase2TrackerDigi_> StubAlgorithm;
  typedef TTStubAlgorithm_official<Ref_Phase2TrackerDigi_> StubAlgorithmOfficial;
  // handles 2 pi overflow
  inline double deltaPhi(double lhs, double rhs = 0.) { return reco::deltaPhi(lhs, rhs); }

  /*! \class  tt::Setup
   *  \brief  Class to process and provide run-time constants used by Track Trigger emulators
   *  \author Thomas Schuh
   *  \date   2020, Apr
   */
  class Setup {
  public:
    // Configuration
    struct Config {
      double beamWindowZ_;
      double minPt_;
      double minPtCand_;
      double maxEta_;
      double maxD0_;
      double chosenRofPhi_;
      int numLayers_;
      int minLayers_;
      int tmttWidthR_;
      int tmttWidthPhi_;
      int tmttWidthZ_;
      int hybridNumLayers_;
      std::vector<int> hybridNumRingsPS_;
      std::vector<int> hybridWidthsR_;
      std::vector<int> hybridWidthsZ_;
      std::vector<int> hybridWidthsPhi_;
      std::vector<int> hybridWidthsAlpha_;
      std::vector<int> hybridWidthsBend_;
      std::vector<double> hybridRangesR_;
      std::vector<double> hybridRangesZ_;
      std::vector<double> hybridRangesAlpha_;
      std::vector<double> hybridLayerRs_;
      std::vector<double> hybridDiskZs_;
      std::vector<edm::ParameterSet> hybridDisk2SRsSet_;
      double hybridRangePhi_;
      double tbBarrelHalfLength_;
      double tbInnerRadius_;
      std::vector<int> tbWidthsR_;
      int enableTruncation_;
      bool useHybrid_;
      int widthDSPa_;
      int widthDSPab_;
      int widthDSPau_;
      int widthDSPb_;
      int widthDSPbb_;
      int widthDSPbu_;
      int widthDSPc_;
      int widthDSPcb_;
      int widthDSPcu_;
      int widthAddrBRAM36_;
      int widthAddrBRAM18_;
      int numFramesInfra_;
      double freqLHC_;
      double freqBEHigh_;
      double freqBELow_;
      int tmpFE_;
      int tmpTFP_;
      double speedOfLight_;
      double bField_;
      double bFieldError_;
      double outerRadius_;
      double innerRadius_;
      double halfLength_;
      double maxPitchRow_;
      double maxPitchCol_;
      double tiltApproxSlope_;
      double tiltApproxIntercept_;
      double tiltUncertaintyR_;
      double scattering_;
      double pitchRow2S_;
      double pitchRowPS_;
      double pitchCol2S_;
      double pitchColPS_;
      double limitPSBarrel_;
      std::vector<double> limitsTiltedR_;
      std::vector<double> limitsTiltedZ_;
      std::vector<double> limitsPSDiksZ_;
      std::vector<double> limitsPSDiksR_;
      std::vector<double> tiltedLayerLimitsZ_;
      std::vector<double> psDiskLimitsR_;
      int widthBend_;
      int widthCol_;
      int widthRow_;
      double baseBend_;
      double baseCol_;
      double baseRow_;
      double baseWindowSize_;
      double bendCut_;
      int numRegions_;
      int numOverlappingRegions_;
      int numATCASlots_;
      int numDTCsPerRegion_;
      int numModulesPerDTC_;
      int dtcNumRoutingBlocks_;
      int dtcDepthMemory_;
      int dtcWidthRowLUT_;
      int dtcWidthInv2R_;
      int offsetDetIdDSV_;
      int offsetDetIdTP_;
      int offsetLayerDisks_;
      int offsetLayerId_;
      int numBarrelLayer_;
      int numBarrelLayerPS_;
      int dtcNumStreams_;
      int slotLimitPS_;
      int slotLimit10gbps_;
      int tfpWidthPhi0_;
      int tfpWidthInvR_;
      int tfpWidthCot_;
      int tfpWidthZ0_;
      int tfpNumChannel_;
      int gpNumBinsPhiT_;
      int gpNumBinsZT_;
      double chosenRofZ_;
      int gpDepthMemory_;
      int gpWidthModule_;
      int gpPosPS_;
      int gpPosBarrel_;
      int gpPosTilted_;
      int htNumBinsInv2R_;
      int htNumBinsPhiT_;
      int htMinLayers_;
      int htDepthMemory_;
      int ctbNumBinsInv2R_;
      int ctbNumBinsPhiT_;
      int ctbNumBinsCot_;
      int ctbNumBinsZT_;
      int ctbMinLayers_;
      int ctbMaxTracks_;
      int ctbMaxStubs_;
      int ctbDepthMemory_;
      bool kfUse5ParameterFit_;
      bool kfUseSimmulation_;
      bool kfUseTTStubResiduals_;
      bool kfUseTTStubParameters_;
      bool kfApplyNonLinearCorrection_;
      int kfNumWorker_;
      int kfMaxTracks_;
      int kfMinLayers_;
      int kfMinLayersPS_;
      int kfMaxLayers_;
      int kfMaxGaps_;
      int kfMaxSeedingLayer_;
      int kfNumSeedStubs_;
      double kfMinSeedDeltaR_;
      double kfRangeFactor_;
      int kfShiftInitialC00_;
      int kfShiftInitialC11_;
      int kfShiftInitialC22_;
      int kfShiftInitialC33_;
      int kfShiftChi20_;
      int kfShiftChi21_;
      double kfCutChi2_;
      int kfWidthChi2_;
      int drDepthMemory_;
      int tqNumChannel_;
    };
    Setup() {}
    Setup(const Config& iConfig,
          const TrackerGeometry& trackerGeometry,
          const TrackerTopology& trackerTopology,
          const TrackerDetToDTCELinkCablingMap& cablingMap,
          const StubAlgorithmOfficial& stubAlgorithm,
          const edm::ParameterSet& pSetStubAlgorithm);
    ~Setup() {}

    // converts tk layout id into dtc id
    int dtcId(int tklId) const;
    // converts dtci id into tk layout id
    int tkLayoutId(int dtcId) const;
    // converts TFP identifier (region[0-8], channel[0-47]) into dtcId [0-215]
    int dtcId(int tfpRegion, int tfpChannel) const;
    // checks if given dtcId is connected to PS or 2S sensormodules
    bool psModule(int dtcId) const;
    // checks if given dtcId is connected via 10 gbps link
    bool gbps10(int dtcId) const;
    // checks if given dtcId is connected to -z (false) or +z (true)
    bool side(int dtcId) const;
    // ATCA slot number [0-11] of given dtcId
    int slot(int dtcId) const;
    // sensor module for det id
    SensorModule* sensorModule(const DetId& detId) const;
    // sensor module for ttStubRef
    SensorModule* sensorModule(const TTStubRef& ttStubRef) const;
    // TrackerGeometry
    const TrackerGeometry* trackerGeometry() const { return trackerGeometry_; }
    // TrackerTopology
    const TrackerTopology* trackerTopology() const { return trackerTopology_; }
    // returns global TTStub position
    GlobalPoint stubPos(const TTStubRef& ttStubRef) const;
    // returns bit accurate hybrid stub radius for given TTStubRef and h/w bit word
    double stubR(const TTBV& hw, const TTStubRef& ttStubRef) const;
    // returns bit accurate position of a stub from a given tfp region [0-8]
    GlobalPoint stubPos(const tt::FrameStub& frame, int region) const;
    // empty trackerDTC EDProduct
    TTDTC ttDTC() const { return TTDTC(numRegions_, numOverlappingRegions_, numDTCsPerRegion_); }
    // stub layer id (barrel: 1 - 6, endcap: 11 - 15)
    int layerId(const TTStubRef& ttStubRef) const;
    // return tracklet layerId (barrel: [0-5], endcap: [6-10]) for given TTStubRef
    int trackletLayerId(const TTStubRef& ttStubRef) const;
    // return index layerId (barrel: [0-5], endcap: [0-6]) for given TTStubRef
    int indexLayerId(const TTStubRef& ttStubRef) const;
    // true if stub from barrel module
    bool barrel(const TTStubRef& ttStubRef) const;
    // true if stub from barrel module
    bool psModule(const TTStubRef& ttStubRef) const;
    // return sensor moduel type
    SensorModule::Type type(const TTStubRef& ttStubRef) const;
    // checks if stub collection is considered forming a reconstructable track
    bool reconstructable(const std::vector<TTStubRef>& ttStubRefs) const;
    //
    TTBV layerMap(const std::vector<int>& ints) const;
    //
    TTBV layerMap(const TTBV& hitPattern, const std::vector<int>& ints) const;
    //
    std::vector<int> layerMap(const TTBV& hitPattern, const TTBV& ttBV) const;
    //
    std::vector<int> layerMap(const TTBV& ttBV) const;
    // stub projected phi uncertainty
    double dPhi(const TTStubRef& ttStubRef, double inv2R) const;
    // stub projected z uncertainty
    double dZ(const TTStubRef& ttStubRef) const;
    // stub projected chi2phi wheight
    double v0(const TTStubRef& ttStubRef, double inv2R) const;
    // stub projected chi2z wheight
    double v1(const TTStubRef& ttStubRef, double cot) const;
    //
    const std::vector<SensorModule>& sensorModules() const { return sensorModules_; }
    //
    TTBV module(double r, double z) const;
    //
    bool ps(const TTBV& module) const { return module[gpPosPS_]; }
    //
    bool barrel(const TTBV& module) const { return module[gpPosBarrel_]; }
    //
    bool tilted(const TTBV& module) const { return module[gpPosTilted_]; }
    // stub projected phi uncertainty for given module type, stub radius and track curvature
    double dPhi(const TTBV& module, double r, double inv2R) const;

    // Firmware specific Parameter

    // enable emulation of truncation for TM, DR, KF, TQ and TFP
    int enableTruncation() const { return enableTruncation_; }
    // use Hybrid or TMTT as TT algorithm
    bool useHybrid() const { return useHybrid_; }
    // width of the 'A' port of an DSP slice
    int widthDSPa() const { return widthDSPa_; }
    // width of the 'A' port of an DSP slice using biased twos complement
    int widthDSPab() const { return widthDSPab_; }
    // width of the 'A' port of an DSP slice using biased binary
    int widthDSPau() const { return widthDSPau_; }
    // width of the 'B' port of an DSP slice
    int widthDSPb() const { return widthDSPb_; }
    // width of the 'B' port of an DSP slice using biased twos complement
    int widthDSPbb() const { return widthDSPbb_; }
    // width of the 'B' port of an DSP slice using biased binary
    int widthDSPbu() const { return widthDSPbu_; }
    // width of the 'C' port of an DSP slice
    int widthDSPc() const { return widthDSPc_; }
    // width of the 'C' port of an DSP slice using biased twos complement
    int widthDSPcb() const { return widthDSPcb_; }
    // width of the 'C' port of an DSP slice using biased binary
    int widthDSPcu() const { return widthDSPcu_; }
    // smallest address width of an BRAM36 configured as broadest simple dual port memory
    int widthAddrBRAM36() const { return widthAddrBRAM36_; }
    // smallest address width of an BRAM18 configured as broadest simple dual port memory
    int widthAddrBRAM18() const { return widthAddrBRAM18_; }
    // number of frames betwen 2 resets of 18 BX packets
    int numFramesHigh() const { return numFramesHigh_; }
    // number of frames betwen 2 resets of 18 BX packets
    int numFramesLow() const { return numFramesLow_; }
    // number of frames needed per reset
    int numFramesInfra() const { return numFramesInfra_; }
    // number of valid frames per 18 BX packet
    int numFramesIOHigh() const { return numFramesIOHigh_; }
    // number of valid frames per 18 BX packet
    int numFramesIOLow() const { return numFramesIOLow_; }
    // number of valid frames per 8 BX packet
    int numFramesFE() const { return numFramesFE_; }

    // Tracker specific Parameter

    // strip pitch of outer tracker sensors in cm
    double pitchRow2S() const { return pitchRow2S_; }
    // pixel pitch of outer tracker sensors in cm
    double pitchRowPS() const { return pitchRowPS_; }
    // strip length of outer tracker sensors in cm
    double pitchCol2S() const { return pitchCol2S_; }
    // pixel length of outer tracker sensors in cm
    double pitchColPS() const { return pitchColPS_; }
    // BField used in fw in T
    double bField() const { return bField_; }
    // outer radius of outer tracker in cm
    double outerRadius() const { return outerRadius_; }
    // inner radius of outer tracker in cm
    double innerRadius() const { return innerRadius_; }
    // half length of outer tracker in cm
    double halfLength() const { return halfLength_; }
    // max strip/pixel length of outer tracker sensors in cm
    double maxPitchCol() const { return maxPitchCol_; }
    // In tilted barrel, grad*|z|/r + int approximates |cosTilt| + |sinTilt * cotTheta|
    double tiltApproxSlope() const { return tiltApproxSlope_; }
    // In tilted barrel, grad*|z|/r + int approximates |cosTilt| + |sinTilt * cotTheta|
    double tiltApproxIntercept() const { return tiltApproxIntercept_; }
    // In tilted barrel, constant assumed stub radial uncertainty * sqrt(12) in cm
    double tiltUncertaintyR() const { return tiltUncertaintyR_; }
    // scattering term used to add stub phi uncertainty depending on assumed track inv2R
    double scattering() const { return scattering_; }
    // barrel layer limit z value to partition into tilted and untilted region
    double tiltedLayerLimitZ(int layer) const { return tiltedLayerLimitsZ_.at(layer); }
    // endcap disk limit r value to partition into PS and 2S region
    double psDiskLimitR(int layer) const { return psDiskLimitsR_.at(layer); }

    // Common track finding parameter

    // half lumi region size in cm
    double beamWindowZ() const { return beamWindowZ_; }
    // converts GeV in 1/cm
    double invPtToDphi() const { return invPtToDphi_; }
    // region size in rad
    double baseRegion() const { return baseRegion_; }
    // max cot(theta) of found tracks
    double maxCot() const { return maxCot_; }
    // cut on stub and TP pt, also defines region overlap shape in GeV
    double minPt() const { return minPt_; }
    // cut on candidate pt
    double minPtCand() const { return minPtCand_; }
    // cut on stub eta
    double maxEta() const { return maxEta_; }
    // constraints track reconstruction phase space
    double maxD0() const { return maxD0_; }
    // critical radius defining region overlap shape in cm
    double chosenRofPhi() const { return chosenRofPhi_; }
    // TMTT: number of detector layers a reconstructbale particle may cross; Hybrid: max number of layers connected to one DTC
    int numLayers() const { return numLayers_; }

    // TMTT specific parameter

    // number of bits used for stub r - ChosenRofPhi
    int tmttWidthR() const { return tmttWidthR_; }
    // number of bits used for stub phi w.r.t. phi sector centre
    int tmttWidthPhi() const { return tmttWidthPhi_; }
    // number of bits used for stub z
    int tmttWidthZ() const { return tmttWidthZ_; }
    // number of bits used for stub layer id
    int tmttWidthLayer() const { return tmttWidthLayer_; }
    // number of bits used for stub eta sector
    int tmttWidthSectorEta() const { return tmttWidthSectorEta_; }
    // number of bits used for stub inv2R
    int tmttWidthInv2R() const { return tmttWidthInv2R_; }
    // internal stub r precision in cm
    double tmttBaseR() const { return tmttBaseR_; }
    // internal stub z precision in cm
    double tmttBaseZ() const { return tmttBaseZ_; }
    // internal stub phi precision in rad
    double tmttBasePhi() const { return tmttBasePhi_; }
    // internal stub inv2R precision in 1/cm
    double tmttBaseInv2R() const { return tmttBaseInv2R_; }
    // internal stub phiT precision in rad
    double tmttBasePhiT() const { return tmttBasePhiT_; }
    // number of padded 0s in output data format
    int tmttNumUnusedBits() const { return tmttNumUnusedBits_; }

    // Hybrid specific parameter

    // max number of layer connected to one DTC
    double hybridNumLayers() const { return hybridNumLayers_; }
    // number of bits used for stub r w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S)
    int hybridWidthR(SensorModule::Type type) const { return hybridWidthsR_.at(type); }
    // number of bits used for stub z w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S)
    int hybridWidthZ(SensorModule::Type type) const { return hybridWidthsZ_.at(type); }
    // number of bits used for stub phi w.r.t. region centre for module types (barrelPS, barrel2S, diskPS, disk2S)
    int hybridWidthPhi(SensorModule::Type type) const { return hybridWidthsPhi_.at(type); }
    // number of bits used for stub row number for module types (barrelPS, barrel2S, diskPS, disk2S)
    int hybridWidthAlpha(SensorModule::Type type) const { return hybridWidthsAlpha_.at(type); }
    // number of bits used for stub bend number for module types (barrelPS, barrel2S, diskPS, disk2S)
    int hybridWidthBend(SensorModule::Type type) const { return hybridWidthsBend_.at(type); }
    // number of bits used for stub layer id
    int hybridWidthLayerId() const { return hybridWidthLayerId_; }
    // precision or r in cm for (barrelPS, barrel2S, diskPS, disk2S)
    double hybridBaseR(SensorModule::Type type) const { return hybridBasesR_.at(type); }
    double hybridBaseR() const { return hybridBaseR_; }
    // precision or phi in rad for (barrelPS, barrel2S, diskPS, disk2S)
    double hybridBasePhi(SensorModule::Type type) const { return hybridBasesPhi_.at(type); }
    double hybridBasePhi() const { return hybridBasePhi_; }
    // precision or z in cm for (barrelPS, barrel2S, diskPS, disk2S)
    double hybridBaseZ(SensorModule::Type type) const { return hybridBasesZ_.at(type); }
    double hybridBaseZ() const { return hybridBaseZ_; }
    // precision or alpha in pitch units for (barrelPS, barrel2S, diskPS, disk2S)
    double hybridBaseAlpha(SensorModule::Type type) const { return hybridBasesAlpha_.at(type); }
    // number of padded 0s in output data format for (barrelPS, barrel2S, diskPS, disk2S)
    int hybridNumUnusedBits(SensorModule::Type type) const { return hybridNumsUnusedBits_.at(type); }
    // stub cut on cot(theta) = tan(lambda) = sinh(eta)
    double hybridMaxCot() const { return hybridMaxCot_; }
    // number of outer PS rings for disk 1, 2, 3, 4, 5
    int hybridNumRingsPS(int layerId) const { return hybridNumRingsPS_.at(layerId); }
    // mean radius of outer tracker barrel layer
    double hybridLayerR(int layerId) const { return hybridLayerRs_.at(layerId); }
    // mean z of outer tracker endcap disks
    double hybridDiskZ(int layerId) const { return hybridDiskZs_.at(layerId); }
    // range of stub phi in rad
    double hybridRangePhi() const { return hybridRangePhi_; }
    // range of stub r in cm
    double hybridRangeR() const { return hybridRangesR_[SensorModule::DiskPS]; }
    // biggest barrel stub z position after TrackBuilder in cm
    double tbBarrelHalfLength() const { return tbBarrelHalfLength_; }
    // smallest stub radius after TrackBuilder in cm
    double tbInnerRadius() const { return tbInnerRadius_; }
    // center radius of outer tracker endcap 2S diks strips
    double disk2SR(int layerId, int r) const { return disk2SRs_.at(layerId).at(r); }
    // number of bits used for stub r w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S) after TrackBuilder
    int tbWidthR(SensorModule::Type type) const { return tbWidthsR_.at(type); }

    // Parameter specifying TTStub algorithm

    // number of tilted layer rings per barrel layer
    double numTiltedLayerRing(int layerId) const { return numTiltedLayerRings_.at(layerId); };
    // stub bend window sizes for flat barrel layer in full pitch units
    double windowSizeBarrelLayer(int layerId) const { return windowSizeBarrelLayers_.at(layerId); };
    // stub bend window sizes for tilted barrel layer rings in full pitch units
    double windowSizeTiltedLayerRing(int layerId, int ring) const {
      return windowSizeTiltedLayerRings_.at(layerId).at(ring);
    };
    // stub bend window sizes for endcap disks rings in full pitch units
    double windowSizeEndcapDisksRing(int layerId, int ring) const {
      return windowSizeEndcapDisksRings_.at(layerId).at(ring);
    };
    // precision of window sizes in pitch units
    double baseWindowSize() const { return baseWindowSize_; }
    // index = encoded bend, value = decoded bend for given window size and module type
    const std::vector<double>& encodingBend(int windowSize, bool psModule) const;
    //getBendCut
    const StubAlgorithmOfficial* stubAlgorithm() const { return stubAlgorithm_; }

    // Parameter specifying front-end

    // number of bits used for internal stub bend
    int widthBend() const { return widthBend_; }
    // number of bits used for internal stub column
    int widthCol() const { return widthCol_; }
    // number of bits used for internal stub row
    int widthRow() const { return widthRow_; }
    // precision of internal stub bend in pitch units
    double baseBend() const { return baseBend_; }
    // precision of internal stub column in pitch units
    double baseCol() const { return baseCol_; }
    // precision of internal stub row in pitch units
    double baseRow() const { return baseRow_; }
    // used stub bend uncertainty in pitch units
    double bendCut() const { return bendCut_; }

    // Parameter specifying DTC

    // number of phi slices the outer tracker readout is organized in
    int numRegions() const { return numRegions_; }
    // number of regions a reconstructable particles may cross
    int numOverlappingRegions() const { return numOverlappingRegions_; }
    // number of Tracker boards per ATCA crate.
    int numATCASlots() const { return numATCASlots_; }
    // number of DTC boards used to readout a detector region, likely constructed to be an integerer multiple of NumSlots_
    int numDTCsPerRegion() const { return numDTCsPerRegion_; }
    // max number of sensor modules connected to one DTC board
    int numModulesPerDTC() const { return numModulesPerDTC_; }
    // number of systiloic arrays in stub router firmware
    int dtcNumRoutingBlocks() const { return dtcNumRoutingBlocks_; }
    // fifo depth in stub router firmware
    int dtcDepthMemory() const { return dtcDepthMemory_; }
    // number of row bits used in look up table
    int dtcWidthRowLUT() const { return dtcWidthRowLUT_; }
    // number of bits used for stub inv2R. lut addr is col + bend = 11 => 1 BRAM -> 18 bits for min and max val -> 9
    int dtcWidthInv2R() const { return dtcWidthInv2R_; }
    // tk layout det id minus DetSetVec->detId
    int offsetDetIdDSV() const { return offsetDetIdDSV_; }
    // tk layout det id minus TrackerTopology lower det id
    int offsetDetIdTP() const { return offsetDetIdTP_; }
    // offset in layer ids between barrel layer and endcap disks
    int offsetLayerDisks() const { return offsetLayerDisks_; }
    // offset between 0 and smallest layer id (barrel layer 1)
    int offsetLayerId() const { return offsetLayerId_; }
    // number of barrel layer
    int numBarrelLayer() const { return numBarrelLayer_; }
    // number of barrel PS layer
    int numBarrelLayerPS() const { return numBarrelLayerPS_; }
    // total number of outer tracker DTCs
    int numDTCs() const { return numDTCs_; }
    // number of DTCs connected to one TFP (48)
    int numDTCsPerTFP() const { return numDTCsPerTFP_; }
    // total number of max possible outer tracker modules (72 per DTC)
    int numModules() const { return numModules_; }
    // max number of moudles connected to a systiloic array in stub router firmware
    int dtcNumModulesPerRoutingBlock() const { return dtcNumModulesPerRoutingBlock_; }
    // number of merged rows for look up
    int dtcNumMergedRows() const { return dtcNumMergedRows_; }
    // number of bits used for phi of row slope
    int dtcWidthM() const { return dtcWidthM_; }
    // internal stub inv2R precision in 1 /cm
    double dtcBaseInv2R() const { return dtcBaseInv2R_; }
    // phi of row slope precision in rad / pitch unit
    double dtcBaseM() const { return dtcBaseM_; }
    // sensor modules connected to given dtc id
    const std::vector<SensorModule*>& dtcModules(int dtcId) const { return dtcModules_.at(dtcId); }
    // total number of output channel
    int dtcNumStreams() const { return dtcNumStreams_; }

    // Parameter specifying TFP

    // number of bist used for phi0
    int tfpWidthPhi0() const { return tfpWidthPhi0_; }
    // umber of bist used for invR
    int tfpWidthInvR() const { return tfpWidthInvR_; }
    // number of bist used for cot(theta)
    int tfpWidthCot() const { return tfpWidthCot_; }
    // number of bist used for z0
    int tfpWidthZ0() const { return tfpWidthZ0_; }
    // number of output links
    int tfpNumChannel() const { return tfpNumChannel_; }

    // Parameter specifying GeometricProcessor

    // number of phi sectors in a processing nonant used in hough transform
    int gpNumBinsPhiT() const { return gpNumBinsPhiT_; }
    // number of eta sectors used in hough transform
    int gpNumBinsZT() const { return gpNumBinsZT_; }
    // # critical radius defining r-z sector shape in cm
    double chosenRofZ() const { return chosenRofZ_; }
    // fifo depth in stub router firmware
    int gpDepthMemory() const { return gpDepthMemory_; }
    //
    int gpWidthModule() const { return gpWidthModule_; }
    // phi sector size in rad
    double baseSector() const { return baseSector_; }
    // total number of sectors
    int numSectors() const { return numSectors_; }
    //
    double maxRphi() const { return maxRphi_; }
    //
    double maxRz() const { return maxRz_; }

    // Parameter specifying HoughTransform

    // number of inv2R bins used in hough transform
    int htNumBinsInv2R() const { return htNumBinsInv2R_; }
    // number of phiT bins used in hough transform
    int htNumBinsPhiT() const { return htNumBinsPhiT_; }
    // required number of stub layers to form a candidate
    int htMinLayers() const { return htMinLayers_; }
    // internal fifo depth
    int htDepthMemory() const { return htDepthMemory_; }

    // Parameter specifying Track Builder

    // number of finer inv2R bins inside HT bin
    int ctbNumBinsInv2R() const { return ctbNumBinsInv2R_; }
    // number of finer phiT bins inside HT bin
    int ctbNumBinsPhiT() const { return ctbNumBinsPhiT_; }
    // number of used z0 bins inside GP ZT bin
    int ctbNumBinsCot() const { return ctbNumBinsCot_; }
    //number of used zT bins inside GP ZT bin
    int ctbNumBinsZT() const { return ctbNumBinsZT_; }
    // required number of stub layers to form a candidate
    int ctbMinLayers() const { return ctbMinLayers_; }
    // max number of output tracks per node
    int ctbMaxTracks() const { return ctbMaxTracks_; }
    // cut on number of stub per layer for input candidates
    int ctbMaxStubs() const { return ctbMaxStubs_; }
    // internal memory depth
    int ctbDepthMemory() const { return ctbDepthMemory_; }

    // Parameter specifying KalmanFilter

    // double precision simulation of 5 parameter fit instead of bit accurate emulation of 4 parameter fit
    bool kfUse5ParameterFit() const { return kfUse5ParameterFit_; }
    // simulate KF instead of emulate
    bool kfUseSimmulation() const { return kfUseSimmulation_; }
    // stub residuals and radius are recalculated from seed parameter and TTStub position
    bool kfUseTTStubResiduals() const { return kfUseTTStubResiduals_; }
    // track parameter are recalculated from seed TTStub positions
    bool kfUseTTStubParameters() const { return kfUseTTStubParameters_; }
    //
    bool kfApplyNonLinearCorrection() const { return kfApplyNonLinearCorrection_; }
    // number of kf worker
    int kfNumWorker() const { return kfNumWorker_; }
    // max number of tracks a kf worker can process
    int kfMaxTracks() const { return kfMaxTracks_; }
    // required number of stub layers to form a track
    int kfMinLayers() const { return kfMinLayers_; }
    // required number of ps stub layers to form a track
    int kfMinLayersPS() const { return kfMinLayersPS_; }
    // maximum number of  layers added to a track
    int kfMaxLayers() const { return kfMaxLayers_; }
    //
    int kfMaxGaps() const { return kfMaxGaps_; }
    //
    int kfMaxSeedingLayer() const { return kfMaxSeedingLayer_; }
    //
    int kfNumSeedStubs() const { return kfNumSeedStubs_; }
    //
    double kfMinSeedDeltaR() const { return kfMinSeedDeltaR_; }
    // search window of each track parameter in initial uncertainties
    double kfRangeFactor() const { return kfRangeFactor_; }
    // initial C00 is given by inv2R uncertainty squared times this power of 2
    int kfShiftInitialC00() const { return kfShiftInitialC00_; }
    // initial C11 is given by phiT uncertainty squared times this power of 2
    int kfShiftInitialC11() const { return kfShiftInitialC11_; }
    // initial C22 is given by cot uncertainty squared times this power of 2
    int kfShiftInitialC22() const { return kfShiftInitialC22_; }
    // initial C33 is given by zT uncertainty squared times this power of 2
    int kfShiftInitialC33() const { return kfShiftInitialC33_; }
    //
    int kfShiftChi20() const { return kfShiftChi20_; }
    //
    int kfShiftChi21() const { return kfShiftChi21_; }
    //
    double kfCutChi2() const { return kfCutChi2_; }
    //
    int kfWidthChi2() const { return kfWidthChi2_; }

    // Parameter specifying DuplicateRemoval

    // internal memory depth
    int drDepthMemory() const { return drDepthMemory_; }

    // Parameter specifying TrackQuaility

    // number of output channel
    int tqNumChannel() const { return tqNumChannel_; }

  private:
    // checks consitency between history and current configuration for a specific module
    void checkHistory(const edm::ProcessHistory&,
                      const edm::pset::Registry*,
                      const std::string&,
                      const edm::ParameterSetID&) const;
    // dumps pSetHistory where incosistent lines with pSetProcess are highlighted
    std::string dumpDiff(const edm::ParameterSet& pSetHistory, const edm::ParameterSet& pSetProcess) const;
    // derive constants
    void calculateConstants();
    // convert configuration of TTStubAlgorithm
    void consumeStubAlgorithm();
    // create bend encodings
    void encodeBend(std::vector<std::vector<double>>&, bool) const;
    // create sensor modules
    void produceSensorModules();
    // range check of dtc id
    void checkDTCId(int dtcId) const;
    // range check of tklayout id
    void checkTKLayoutId(int tkLayoutId) const;
    // range check of tfp identifier
    void checkTFPIdentifier(int tfpRegion, int tfpChannel) const;
    // configure TPSelector
    void configureTPSelector();

    // TrackerGeometry
    const TrackerGeometry* trackerGeometry_;
    // TrackerTopology
    const TrackerTopology* trackerTopology_;
    // CablingMap
    const TrackerDetToDTCELinkCablingMap* cablingMap_;
    // TTStub algorithm used to create bend encodings
    const StubAlgorithmOfficial* stubAlgorithm_;
    // pSet of ttStub algorithm, used to identify bend window sizes of sensor modules
    const edm::ParameterSet* pSetSA_;

    // half lumi region size in cm
    double beamWindowZ_;
    // cut on stub and TP pt, also defines region overlap shape in GeV
    double minPt_;
    // cut on candidate pt
    double minPtCand_;
    // cut on stub eta
    double maxEta_;
    // in cm, constraints track reconstruction phase space
    double maxD0_;
    // critical radius defining region overlap shape in cm
    double chosenRofPhi_;
    // number of detector layers a reconstructbale particle may cross
    int numLayers_;
    // required number of stub layers to form a track
    int minLayers_;

    // number of bits used for stub r - ChosenRofPhi
    int tmttWidthR_;
    // number of bits used for stub phi w.r.t. phi sector centre
    int tmttWidthPhi_;
    // number of bits used for stub z
    int tmttWidthZ_;

    // max number of layers connected to one DTC
    int hybridNumLayers_;
    // number of outer PS rings for disk 1, 2, 3, 4, 5
    std::vector<int> hybridNumRingsPS_;
    // number of bits used for stub r w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<int> hybridWidthsR_;
    // number of bits used for stub z w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<int> hybridWidthsZ_;
    // number of bits used for stub phi w.r.t. region centre for module types (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<int> hybridWidthsPhi_;
    // number of bits used for stub row number for module types (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<int> hybridWidthsAlpha_;
    // number of bits used for stub bend number for module types (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<int> hybridWidthsBend_;
    // range in stub r which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<double> hybridRangesR_;
    // range in stub z which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<double> hybridRangesZ_;
    // range in stub row which needs to be covered for module types (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<double> hybridRangesAlpha_;
    // mean radius of outer tracker barrel layer
    std::vector<double> hybridLayerRs_;
    // mean z of outer tracker endcap disks
    std::vector<double> hybridDiskZs_;
    // center radius of outer tracker endcap 2S diks strips
    std::vector<edm::ParameterSet> hybridDisk2SRsSet_;
    // range of stub phi in rad
    double hybridRangePhi_;
    // biggest barrel stub z position after TrackBuilder in cm
    double tbBarrelHalfLength_;
    // smallest stub radius after TrackBuilder in cm
    double tbInnerRadius_;
    // number of bits used for stub r w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S) after TrackBuilder
    std::vector<int> tbWidthsR_;

    // enable emulation of truncation for TM, DR, KF, TQ and TFP
    int enableTruncation_;
    // use Hybrid or TMTT as TT algorithm
    bool useHybrid_;
    // width of the 'A' port of an DSP slice
    int widthDSPa_;
    // width of the 'A' port of an DSP slice using biased twos complement
    int widthDSPab_;
    // width of the 'A' port of an DSP slice using biased binary
    int widthDSPau_;
    // width of the 'B' port of an DSP slice
    int widthDSPb_;
    // width of the 'B' port of an DSP slice using biased twos complement
    int widthDSPbb_;
    // width of the 'B' port of an DSP slice using biased binary
    int widthDSPbu_;
    // width of the 'C' port of an DSP slice
    int widthDSPc_;
    // width of the 'C' port of an DSP slice using biased twos complement
    int widthDSPcb_;
    // width of the 'C' port of an DSP slice using biased binary
    int widthDSPcu_;
    // smallest address width of an BRAM36 configured as broadest simple dual port memory
    int widthAddrBRAM36_;
    // smallest address width of an BRAM18 configured as broadest simple dual port memory
    int widthAddrBRAM18_;
    // needed gap between events of emp-infrastructure firmware
    int numFramesInfra_;
    // LHC bunch crossing rate in MHz
    double freqLHC_;
    // processing Frequency of DTC & TFP in MHz, has to be integer multiple of FreqLHC
    double freqBEHigh_;
    // processing Frequency of DTC & TFP in MHz, has to be integer multiple of FreqLHC
    double freqBELow_;
    // number of events collected in front-end
    int tmpFE_;
    // time multiplexed period of track finding processor
    int tmpTFP_;
    // speed of light used in FW in e8 m/s
    double speedOfLight_;

    // BField used in fw in T
    double bField_;
    // accepted BField difference between FW to EventSetup in T
    double bFieldError_;
    // outer radius of outer tracker in cm
    double outerRadius_;
    // inner radius of outer tracker in cm
    double innerRadius_;
    // half length of outer tracker in cm
    double halfLength_;
    // max strip/pixel pitch of outer tracker sensors in cm
    double maxPitchRow_;
    // max strip/pixel length of outer tracker sensors in cm
    double maxPitchCol_;
    // approximated tilt correction parameter used to project r to z uncertainty
    double tiltApproxSlope_;
    // approximated tilt correction parameter used to project r to z uncertainty
    double tiltApproxIntercept_;
    // In tilted barrel, constant assumed stub radial uncertainty * sqrt(12) in cm
    double tiltUncertaintyR_;
    // scattering term used to add stub phi uncertainty depending on assumed track inv2R
    double scattering_;
    // strip pitch of outer tracker sensors in cm
    double pitchRow2S_;
    // pixel pitch of outer tracker sensors in cm
    double pitchRowPS_;
    // strip length of outer tracker sensors in cm
    double pitchCol2S_;
    // pixel length of outer tracker sensors in cm
    double pitchColPS_;
    // barrel layer limit r value to partition into PS and 2S region
    double limitPSBarrel_;
    // barrel layer limit r value to partition into tilted and untilted region
    std::vector<double> limitsTiltedR_;
    // barrel layer limit |z| value to partition into tilted and untilted region
    std::vector<double> limitsTiltedZ_;
    // endcap disk limit |z| value to partition into PS and 2S region
    std::vector<double> limitsPSDiksZ_;
    // endcap disk limit r value to partition into PS and 2S region
    std::vector<double> limitsPSDiksR_;
    // barrel layer limit |z| value to partition into tilted and untilted region
    std::vector<double> tiltedLayerLimitsZ_;
    // endcap disk limit r value to partition into PS and 2S region
    std::vector<double> psDiskLimitsR_;

    // number of bits used for internal stub bend
    int widthBend_;
    // number of bits used for internal stub column
    int widthCol_;
    // number of bits used for internal stub row
    int widthRow_;
    // precision of internal stub bend in pitch units
    double baseBend_;
    // precision of internal stub column in pitch units
    double baseCol_;
    // precision of internal stub row in pitch units
    double baseRow_;
    // precision of window sizes in pitch units
    double baseWindowSize_;
    // used stub bend uncertainty in pitch units
    double bendCut_;

    // number of phi slices the outer tracker readout is organized in
    int numRegions_;
    // number of regions a reconstructable particles may cross
    int numOverlappingRegions_;
    // number of Slots in used ATCA crates
    int numATCASlots_;
    // number of DTC boards used to readout a detector region, likely constructed to be an integerer multiple of NumSlots_
    int numDTCsPerRegion_;
    // max number of sensor modules connected to one DTC board
    int numModulesPerDTC_;
    // number of systiloic arrays in stub router firmware
    int dtcNumRoutingBlocks_;
    // fifo depth in stub router firmware
    int dtcDepthMemory_;
    // number of row bits used in look up table
    int dtcWidthRowLUT_;
    // number of bits used for stub inv2R. lut addr is col + bend = 11 => 1 BRAM -> 18 bits for min and max val -> 9
    int dtcWidthInv2R_;
    // tk layout det id minus DetSetVec->detId
    int offsetDetIdDSV_;
    // tk layout det id minus TrackerTopology lower det id
    int offsetDetIdTP_;
    // offset in layer ids between barrel layer and endcap disks
    int offsetLayerDisks_;
    // offset between 0 and smallest layer id (barrel layer 1)
    int offsetLayerId_;
    // number of barrel layer
    int numBarrelLayer_;
    // number of barrel ps layer
    int numBarrelLayerPS_;
    // total number of output channel
    int dtcNumStreams_;
    // slot number changing from PS to 2S (default: 6)
    int slotLimitPS_;
    // slot number changing from 10 gbps to 5gbps (default: 3)
    int slotLimit10gbps_;

    // number of bits used for phi0
    int tfpWidthPhi0_;
    // umber of bits used for qOverPt
    int tfpWidthInvR_;
    // number of bits used for cot(theta)
    int tfpWidthCot_;
    // number of bits used for z0
    int tfpWidthZ0_;
    // number of output links
    int tfpNumChannel_;

    // number of phi sectors used in hough transform
    int gpNumBinsPhiT_;
    // number of eta sectors used in hough transform
    int gpNumBinsZT_;
    // # critical radius defining r-z sector shape in cm
    double chosenRofZ_;
    // fifo depth in stub router firmware
    int gpDepthMemory_;
    //
    int gpWidthModule_;
    //
    int gpPosPS_;
    //
    int gpPosBarrel_;
    //
    int gpPosTilted_;

    // number of inv2R bins used in hough transform
    int htNumBinsInv2R_;
    // number of phiT bins used in hough transform
    int htNumBinsPhiT_;
    // required number of stub layers to form a candidate
    int htMinLayers_;
    // internal fifo depth
    int htDepthMemory_;

    // number of finer inv2R bins inside HT bin
    int ctbNumBinsInv2R_;
    // number of finer phiT bins inside HT bin
    int ctbNumBinsPhiT_;
    // number of used cot bins inside GP ZT bin
    int ctbNumBinsCot_;
    //number of used zT bins inside GP ZT bin
    int ctbNumBinsZT_;
    // required number of stub layers to form a candidate
    int ctbMinLayers_;
    // max number of output tracks per node
    int ctbMaxTracks_;
    // cut on number of stub per layer for input candidates
    int ctbMaxStubs_;
    // internal memory depth
    int ctbDepthMemory_;

    // double precision simulation of 5 parameter fit instead of bit accurate emulation of 4 parameter fit
    bool kfUse5ParameterFit_;
    // simulate KF instead of emulate
    bool kfUseSimmulation_;
    // stub residuals and radius are recalculated from seed parameter and TTStub position
    bool kfUseTTStubResiduals_;
    // track parameter are recalculated from seed TTStub positions
    bool kfUseTTStubParameters_;
    //
    bool kfApplyNonLinearCorrection_;
    // number of kf worker
    int kfNumWorker_;
    // max number of tracks a kf worker can process
    int kfMaxTracks_;
    // required number of stub layers to form a track
    int kfMinLayers_;
    // required number of ps stub layers to form a track
    int kfMinLayersPS_;
    // maximum number of  layers added to a track
    int kfMaxLayers_;
    //
    int kfMaxGaps_;
    //
    int kfMaxSeedingLayer_;
    //
    int kfNumSeedStubs_;
    //
    double kfMinSeedDeltaR_;
    // search window of each track parameter in initial uncertainties
    double kfRangeFactor_;
    // initial C00 is given by inv2R uncertainty squared times this power of 2
    int kfShiftInitialC00_;
    // initial C11 is given by phiT uncertainty squared times this power of 2
    int kfShiftInitialC11_;
    // initial C22 is given by cot uncertainty squared times this power of 2
    int kfShiftInitialC22_;
    // initial C33 is given by zT uncertainty squared times this power of 2
    int kfShiftInitialC33_;
    //
    int kfShiftChi20_;
    //
    int kfShiftChi21_;
    //
    double kfCutChi2_;
    //
    int kfWidthChi2_;

    // internal memory depth
    int drDepthMemory_;

    // number of output channel
    int tqNumChannel_;

    //
    // Derived constants
    //

    // TTStubAlgorithm

    // number of tilted layer rings per barrel layer
    std::vector<double> numTiltedLayerRings_;
    // stub bend window sizes for flat barrel layer in full pitch units
    std::vector<double> windowSizeBarrelLayers_;
    // stub bend window sizes for tilted barrel layer rings in full pitch units
    std::vector<std::vector<double>> windowSizeTiltedLayerRings_;
    // stub bend window sizes for endcap disks rings in full pitch units
    std::vector<std::vector<double>> windowSizeEndcapDisksRings_;
    // maximum stub bend window in half strip units
    int maxWindowSize_;

    // common Track finding

    // number of frames betwen 2 resets of 18 BX packets
    int numFramesHigh_;
    // number of frames betwen 2 resets of 18 BX packets
    int numFramesLow_;
    // number of valid frames per 18 BX packet
    int numFramesIOHigh_;
    // number of valid frames per 18 BX packet
    int numFramesIOLow_;
    // number of valid frames per 8 BX packet
    int numFramesFE_;
    // converts GeV in 1/cm
    double invPtToDphi_;
    // region size in rad
    double baseRegion_;
    // max cot(theta) of found tracks
    double maxCot_;

    // TMTT

    // number of bits used for stub layer id
    int widthLayerId_;
    // internal stub r precision in cm
    double tmttBaseR_;
    // internal stub z precision in cm
    double tmttBaseZ_;
    // internal stub phi precision in rad
    double tmttBasePhi_;
    // internal stub inv2R precision in 1/cm
    double tmttBaseInv2R_;
    // internal stub phiT precision in rad
    double tmttBasePhiT_;
    // number of padded 0s in output data format
    int dtcNumUnusedBits_;
    // number of bits used for stub layer id
    int tmttWidthLayer_;
    // number of bits used for stub eta sector
    int tmttWidthSectorEta_;
    // number of bits used for stub inv2R
    int tmttWidthInv2R_;
    // number of padded 0s in output data format
    int tmttNumUnusedBits_;

    // hybrid

    // number of bits used for stub layer id
    int hybridWidthLayerId_;
    // precision or r in cm for (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<double> hybridBasesR_;
    // precision or phi in rad for (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<double> hybridBasesPhi_;
    // precision or z in cm for (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<double> hybridBasesZ_;
    // precision or alpha in pitch units for (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<double> hybridBasesAlpha_;
    // stub r precision in cm
    double hybridBaseZ_;
    // stub z precision in cm
    double hybridBaseR_;
    // stub phi precision in rad
    double hybridBasePhi_;
    // stub cut on cot(theta) = tan(lambda) = sinh(eta)
    double hybridMaxCot_;
    // number of padded 0s in output data format for (barrelPS, barrel2S, diskPS, disk2S)
    std::vector<int> hybridNumsUnusedBits_;
    // center radius of outer tracker endcap 2S diks strips
    std::vector<std::vector<double>> disk2SRs_;

    // DTC

    // total number of outer tracker DTCs
    int numDTCs_;
    // number of DTCs connected to one TFP (48)
    int numDTCsPerTFP_;
    // total number of max possible outer tracker modules (72 per DTC)
    int numModules_;
    // max number of moudles connected to a systiloic array in stub router firmware
    int dtcNumModulesPerRoutingBlock_;
    // number of merged rows for look up
    int dtcNumMergedRows_;
    // number of bits used for phi of row slope
    int dtcWidthM_;
    // internal stub inv2R precision in 1 /cm
    double dtcBaseInv2R_;
    // phi of row slope precision in rad / pitch unit
    double dtcBaseM_;
    // outer index = module window size, inner index = encoded bend, inner value = decoded bend, for ps modules
    std::vector<std::vector<double>> encodingsBendPS_;
    // outer index = module window size, inner index = encoded bend, inner value = decoded bend, for 2s modules
    std::vector<std::vector<double>> encodingsBend2S_;
    // collection of outer tracker sensor modules
    std::vector<SensorModule> sensorModules_;
    // collection of outer tracker sensor modules organised in DTCS [0-215][0-71]
    std::vector<std::vector<SensorModule*>> dtcModules_;
    // hepler to convert Stubs quickly
    std::unordered_map<DetId, SensorModule*> detIdToSensorModule_;

    // GP

    // phi sector size in rad
    double baseSector_;
    //
    double maxRphi_;
    //
    double maxRz_;
    // total number of sectors
    int numSectors_;

    // CTB

    // number of bits used to count stubs per layer
    int ctbWidthLayerCount_;

    // KFout

    // Bins used to digitize dPhi for chi2 calculation
    std::vector<int> kfoutdPhiBins_;
    // Bins used to digitize dZ for chi2 calculation
    std::vector<int> kfoutdZBins_;
    // v0 weight Bins corresponding to dPhi Bins for chi2 calculation
    std::vector<int> kfoutv0Bins_;
    // v1 weight Bins corresponding to dZ Bins for chi2 calculation
    std::vector<int> kfoutv1Bins_;
  };

}  // namespace tt

EVENTSETUP_DATA_DEFAULT_RECORD(tt::Setup, tt::SetupRcd);

#endif
