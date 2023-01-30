#ifndef L1Trigger_TrackTrigger_Setup_h
#define L1Trigger_TrackTrigger_Setup_h

#include "FWCore/Framework/interface/data_default_record_trait.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/Provenance/interface/ProcessHistory.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/GeometryVector/interface/GlobalPoint.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_official.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"
#include "SimTracker/Common/interface/TrackingParticleSelector.h"

#include "SimTracker/TrackTriggerAssociation/interface/TTTypes.h"
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
    Setup() {}
    Setup(const edm::ParameterSet& iConfig,
          const MagneticField& magneticField,
          const TrackerGeometry& trackerGeometry,
          const TrackerTopology& trackerTopology,
          const TrackerDetToDTCELinkCablingMap& cablingMap,
          const StubAlgorithmOfficial& stubAlgorithm,
          const edm::ParameterSet& pSetStubAlgorithm,
          const edm::ParameterSet& pSetGeometryConfiguration,
          const edm::ParameterSetID& pSetIdTTStubAlgorithm,
          const edm::ParameterSetID& pSetIdGeometryConfiguration);
    ~Setup() {}

    // true if tracker geometry and magnetic field supported
    bool configurationSupported() const { return configurationSupported_; }
    // checks current configuration vs input sample configuration
    void checkHistory(const edm::ProcessHistory& processHistory) const;
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
    // TrackerGeometry
    const TrackerGeometry* trackerGeometry() const { return trackerGeometry_; }
    // TrackerTopology
    const TrackerTopology* trackerTopology() const { return trackerTopology_; }
    // returns global TTStub position
    GlobalPoint stubPos(const TTStubRef& ttStubRef) const;
    // returns bit accurate hybrid stub radius for given TTStubRef and h/w bit word
    double stubR(const TTBV& hw, const TTStubRef& ttStubRef) const;
    // returns bit accurate position of a stub from a given tfp region [0-8]
    GlobalPoint stubPos(bool hybrid, const tt::FrameStub& frame, int region) const;
    // empty trackerDTC EDProduct
    TTDTC ttDTC() const { return TTDTC(numRegions_, numOverlappingRegions_, numDTCsPerRegion_); }
    // checks if stub collection is considered forming a reconstructable track
    bool reconstructable(const std::vector<TTStubRef>& ttStubRefs) const;
    // checks if tracking particle is selected for efficiency measurements
    bool useForAlgEff(const TrackingParticle& tp) const;
    // checks if tracking particle is selected for fake and duplicate rate measurements
    bool useForReconstructable(const TrackingParticle& tp) const { return tpSelectorLoose_(tp); }
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
    double dZ(const TTStubRef& ttStubRef, double cot) const;
    // stub projected chi2phi wheight
    double v0(const TTStubRef& ttStubRef, double inv2R) const;
    // stub projected chi2z wheight
    double v1(const TTStubRef& ttStubRef, double cot) const;
    //
    const std::vector<SensorModule>& sensorModules() const { return sensorModules_; }

    // Firmware specific Parameter

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
    int numFrames() const { return numFrames_; }
    // number of frames needed per reset
    int numFramesInfra() const { return numFramesInfra_; }
    // number of valid frames per 18 BX packet
    int numFramesIO() const { return numFramesIO_; }
    // number of valid frames per 8 BX packet
    int numFramesFE() const { return numFramesFE_; }
    // maximum representable stub phi uncertainty
    double maxdPhi() const { return maxdPhi_; }
    // maximum representable stub z uncertainty
    double maxdZ() const { return maxdZ_; }
    // barrel layer limit z value to partition into tilted and untilted region
    double tiltedLayerLimitZ(int layer) const { return tiltedLayerLimitsZ_.at(layer); }
    // endcap disk limit r value to partition into PS and 2S region
    double psDiskLimitR(int layer) const { return psDiskLimitsR_.at(layer); }
    // strip pitch of outer tracker sensors in cm
    double pitch2S() const { return pitch2S_; }
    // pixel pitch of outer tracker sensors in cm
    double pitchPS() const { return pitchPS_; }
    // strip length of outer tracker sensors in cm
    double length2S() const { return length2S_; }
    // pixel length of outer tracker sensors in cm
    double lengthPS() const { return lengthPS_; }

    // Common track finding parameter

    // half lumi region size in cm
    double beamWindowZ() const { return beamWindowZ_; }
    // converts GeV in 1/cm
    double invPtToDphi() const { return invPtToDphi_; }
    // region size in rad
    double baseRegion() const { return baseRegion_; }
    // pt cut
    double tpMinPt() const { return tpMinPt_; }
    // TP eta cut
    double tpMaxEta() const { return tpMaxEta_; }
    // TP cut on vertex pos r in cm
    double tpMaxVertR() const { return tpMaxVertR_; }
    // TP cut on vertex pos z in cm
    double tpMaxVertZ() const { return tpMaxVertZ_; }
    // TP cut on impact parameter in cm
    double tpMaxD0() const { return tpMaxD0_; }
    // required number of associated layers to a TP to consider it reconstruct-able
    int tpMinLayers() const { return tpMinLayers_; }
    // required number of associated ps layers to a TP to consider it reconstruct-able
    int tpMinLayersPS() const { return tpMinLayersPS_; }
    // max number of unassociated 2S stubs allowed to still associate TTTrack with TP
    int tpMaxBadStubs2S() const { return tpMaxBadStubs2S_; }
    // max number of unassociated PS stubs allowed to still associate TTTrack with TP
    int tpMaxBadStubsPS() const { return tpMaxBadStubsPS_; }
    // BField used in fw in T
    double bField() const { return bField_; }

    // TMTT specific parameter

    // cut on stub and TP pt, also defines region overlap shape in GeV
    double minPt() const { return minPt_; }
    // cut on stub eta
    double maxEta() const { return maxEta_; }
    // critical radius defining region overlap shape in cm
    double chosenRofPhi() const { return chosenRofPhi_; }
    // number of detector layers a reconstructbale particle may cross
    int numLayers() const { return numLayers_; }
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
    // outer radius of outer tracker in cm
    double outerRadius() const { return outerRadius_; }
    // inner radius of outer tracker in cm
    double innerRadius() const { return innerRadius_; }
    // half length of outer tracker in cm
    double halfLength() const { return halfLength_; }
    // max strip/pixel length of outer tracker sensors in cm
    double maxLength() const { return maxLength_; }
    // In tilted barrel, grad*|z|/r + int approximates |cosTilt| + |sinTilt * cotTheta|
    double tiltApproxSlope() const { return tiltApproxSlope_; }
    // In tilted barrel, grad*|z|/r + int approximates |cosTilt| + |sinTilt * cotTheta|
    double tiltApproxIntercept() const { return tiltApproxIntercept_; }
    // In tilted barrel, constant assumed stub radial uncertainty * sqrt(12) in cm
    double tiltUncertaintyR() const { return tiltUncertaintyR_; }
    // scattering term used to add stub phi uncertainty depending on assumed track inv2R
    double scattering() const { return scattering_; }

    // Hybrid specific parameter

    // cut on stub pt in GeV, also defines region overlap shape
    double hybridMinPtStub() const { return hybridMinPtStub_; }
    // cut on andidate pt in GeV
    double hybridMinPtCand() const { return hybridMinPtCand_; }
    // cut on stub eta
    double hybridMaxEta() const { return hybridMaxEta_; }
    // critical radius defining region overlap shape in cm
    double hybridChosenRofPhi() const { return hybridChosenRofPhi_; }
    // max number of detector layer connected to one DTC
    int hybridNumLayers() const { return hybridNumLayers_; }
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
    // precision or phi in rad for (barrelPS, barrel2S, diskPS, disk2S)
    double hybridBasePhi(SensorModule::Type type) const { return hybridBasesPhi_.at(type); }
    // precision or z in cm for (barrelPS, barrel2S, diskPS, disk2S)
    double hybridBaseZ(SensorModule::Type type) const { return hybridBasesZ_.at(type); }
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
    //
    int numBarrelLayer() const { return numBarrelLayer_; }
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
    // umber of bist used for inv2R
    int tfpWidthInv2R() const { return tfpWidthInv2R_; }
    // number of bist used for cot(theta)
    int tfpWidthCot() const { return tfpWidthCot_; }
    // number of bist used for z0
    int tfpWidthZ0() const { return tfpWidthZ0_; }
    // number of output links
    int tfpNumChannel() const { return tfpNumChannel_; }

    // Parameter specifying GeometricProcessor

    // number of phi sectors in a processing nonant used in hough transform
    int numSectorsPhi() const { return numSectorsPhi_; }
    // number of eta sectors used in hough transform
    int numSectorsEta() const { return numSectorsEta_; }
    // # critical radius defining r-z sector shape in cm
    double chosenRofZ() const { return chosenRofZ_; }
    // fifo depth in stub router firmware
    int gpDepthMemory() const { return gpDepthMemory_; }
    // defining r-z sector shape
    double boundarieEta(int eta) const { return boundariesEta_.at(eta); }
    std::vector<double> boundarieEta() const { return boundariesEta_; }
    // phi sector size in rad
    double baseSector() const { return baseSector_; }
    // cut on zT
    double maxZT() const { return maxZT_; }
    // cut on stub cot theta
    double maxCot() const { return maxCot_; }
    // total number of sectors
    int numSectors() const { return numSectors_; }
    // cot(theta) of given eta sector
    double sectorCot(int eta) const { return sectorCots_.at(eta); }
    //
    double neededRangeChiZ() const { return neededRangeChiZ_; }

    // Parameter specifying HoughTransform

    // number of inv2R bins used in hough transform
    int htNumBinsInv2R() const { return htNumBinsInv2R_; }
    // number of phiT bins used in hough transform
    int htNumBinsPhiT() const { return htNumBinsPhiT_; }
    // required number of stub layers to form a candidate
    int htMinLayers() const { return htMinLayers_; }
    // internal fifo depth
    int htDepthMemory() const { return htDepthMemory_; }

    // Parameter specifying MiniHoughTransform

    // number of finer inv2R bins inside HT bin
    int mhtNumBinsInv2R() const { return mhtNumBinsInv2R_; }
    // number of finer phiT bins inside HT bin
    int mhtNumBinsPhiT() const { return mhtNumBinsPhiT_; }
    // number of dynamic load balancing steps
    int mhtNumDLBs() const { return mhtNumDLBs_; }
    // number of units per dynamic load balancing step
    int mhtNumDLBNodes() const { return mhtNumDLBNodes_; }
    // number of inputs per dynamic load balancing unit
    int mhtNumDLBChannel() const { return mhtNumDLBChannel_; }
    // required number of stub layers to form a candidate
    int mhtMinLayers() const { return mhtMinLayers_; }
    // number of mht cells
    int mhtNumCells() const { return mhtNumCells_; }

    // Parameter specifying ZHoughTransform

    //number of used zT bins
    int zhtNumBinsZT() const { return zhtNumBinsZT_; }
    // number of used cot bins
    int zhtNumBinsCot() const { return zhtNumBinsCot_; }
    //  number of stages
    int zhtNumStages() const { return zhtNumStages_; }
    // required number of stub layers to form a candidate
    int zhtMinLayers() const { return zhtMinLayers_; }
    // max number of output tracks per node
    int zhtMaxTracks() const { return zhtMaxTracks_; }
    // cut on number of stub per layer for input candidates
    int zhtMaxStubsPerLayer() const { return zhtMaxStubsPerLayer_; }
    // number of zht cells
    int zhtNumCells() const { return zhtNumCells_; }

    // Parameter specifying KalmanFilter Input Formatter

    // power of 2 multiplier of stub phi residual range
    int kfinShiftRangePhi() const { return kfinShiftRangePhi_; }
    // power of 2 multiplier of stub z residual range
    int kfinShiftRangeZ() const { return kfinShiftRangeZ_; }

    // Parameter specifying KalmanFilter

    // number of kf worker
    int kfNumWorker() const { return kfNumWorker_; }
    // required number of stub layers to form a track
    int kfMinLayers() const { return kfMinLayers_; }
    // maximum number of  layers added to a track
    int kfMaxLayers() const { return kfMaxLayers_; }
    // search window of each track parameter in initial uncertainties
    double kfRangeFactor() const { return kfRangeFactor_; }
    //
    int kfShiftInitialC00() const { return kfShiftInitialC00_; }
    //
    int kfShiftInitialC11() const { return kfShiftInitialC11_; }
    //
    int kfShiftInitialC22() const { return kfShiftInitialC22_; }
    //
    int kfShiftInitialC33() const { return kfShiftInitialC33_; }

    // Parameter specifying KalmanFilter Output Formatter
    // Conversion factor between dphi^2/weight and chi2rphi
    int kfoutchi2rphiConv() const { return kfoutchi2rphiConv_; }
    // Conversion factor between dz^2/weight and chi2rz
    int kfoutchi2rzConv() const { return kfoutchi2rzConv_; }
    // Fraction of total dphi and dz ranges to calculate v0 and v1 LUT for
    int weightBinFraction() const { return weightBinFraction_; }
    // Constant used in FW to prevent 32-bit int overflow
    int dzTruncation() const { return dzTruncation_; }
    // Constant used in FW to prevent 32-bit int overflow
    int dphiTruncation() const { return dphiTruncation_; }

    // Parameter specifying DuplicateRemoval

    // internal memory depth
    int drDepthMemory() const { return drDepthMemory_; }

    //getBendCut
    const StubAlgorithmOfficial* stubAlgorithm() const { return stubAlgorithm_; }

  private:
    // checks consitency between history and current configuration for a specific module
    void checkHistory(const edm::ProcessHistory&,
                      const edm::pset::Registry*,
                      const std::string&,
                      const edm::ParameterSetID&) const;
    // dumps pSetHistory where incosistent lines with pSetProcess are highlighted
    std::string dumpDiff(const edm::ParameterSet& pSetHistory, const edm::ParameterSet& pSetProcess) const;
    // check if bField is supported
    void checkMagneticField();
    // check if geometry is supported
    void checkGeometry();
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

    // MagneticField
    const MagneticField* magneticField_;
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
    // pSet of geometry configuration, used to identify if geometry is supported
    const edm::ParameterSet* pSetGC_;
    // pset id of current TTStubAlgorithm
    edm::ParameterSetID pSetIdTTStubAlgorithm_;
    // pset id of current geometry configuration
    edm::ParameterSetID pSetIdGeometryConfiguration_;

    // DD4hep
    bool fromDD4hep_;

    // Parameter to check if configured Tracker Geometry is supported
    edm::ParameterSet pSetSG_;
    // label of ESProducer/ESSource
    std::string sgXMLLabel_;
    // compared path
    std::string sgXMLPath_;
    // compared filen ame
    std::string sgXMLFile_;
    // list of supported versions
    std::vector<std::string> sgXMLVersions_;

    // Parameter to check if Process History is consistent with process configuration
    edm::ParameterSet pSetPH_;
    // label of compared GeometryConfiguration
    std::string phGeometryConfiguration_;
    // label of compared TTStubAlgorithm
    std::string phTTStubAlgorithm_;

    // Common track finding parameter
    edm::ParameterSet pSetTF_;
    // half lumi region size in cm
    double beamWindowZ_;
    // required number of layers a found track has to have in common with a TP to consider it matched to it
    int matchedLayers_;
    // required number of ps layers a found track has to have in common with a TP to consider it matched to it
    int matchedLayersPS_;
    // allowed number of stubs a found track may have not in common with its matched TP
    int unMatchedStubs_;
    // allowed number of PS stubs a found track may have not in common with its matched TP
    int unMatchedStubsPS_;
    // scattering term used to add stub phi uncertainty depending on assumed track inv2R
    double scattering_;

    // TMTT specific parameter
    edm::ParameterSet pSetTMTT_;
    // cut on stub and TP pt, also defines region overlap shape in GeV
    double minPt_;
    // cut on stub eta
    double maxEta_;
    // critical radius defining region overlap shape in cm
    double chosenRofPhi_;
    // number of detector layers a reconstructbale particle may cross
    int numLayers_;
    // number of bits used for stub r - ChosenRofPhi
    int tmttWidthR_;
    // number of bits used for stub phi w.r.t. phi sector centre
    int tmttWidthPhi_;
    // number of bits used for stub z
    int tmttWidthZ_;

    // Hybrid specific parameter
    edm::ParameterSet pSetHybrid_;
    // cut on stub pt in GeV, also defines region overlap shape
    double hybridMinPtStub_;
    // cut on andidate pt in GeV
    double hybridMinPtCand_;
    // cut on stub eta
    double hybridMaxEta_;
    // critical radius defining region overlap shape in cm
    double hybridChosenRofPhi_;
    // max number of detector layer connected to one DTC
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
    // smallest stub radius after TrackBuilder in cm
    double tbInnerRadius_;
    // number of bits used for stub r w.r.t layer/disk centre for module types (barrelPS, barrel2S, diskPS, disk2S) after TrackBuilder
    std::vector<int> tbWidthsR_;

    // Parameter specifying TrackingParticle used for Efficiency measurements
    edm::ParameterSet pSetTP_;
    // pt cut
    double tpMinPt_;
    // eta cut
    double tpMaxEta_;
    // cut on vertex pos r in cm
    double tpMaxVertR_;
    // cut on vertex pos z in cm
    double tpMaxVertZ_;
    // cut on impact parameter in cm
    double tpMaxD0_;
    // required number of associated layers to a TP to consider it reconstruct-able
    int tpMinLayers_;
    // required number of associated ps layers to a TP to consider it reconstruct-able
    int tpMinLayersPS_;
    // max number of unassociated 2S stubs allowed to still associate TTTrack with TP
    int tpMaxBadStubs2S_;
    // max number of unassociated PS stubs allowed to still associate TTTrack with TP
    int tpMaxBadStubsPS_;

    // Firmware specific Parameter
    edm::ParameterSet pSetFW_;
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
    double freqBE_;
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
    double maxPitch_;
    // max strip/pixel length of outer tracker sensors in cm
    double maxLength_;
    // approximated tilt correction parameter used to project r to z uncertainty
    double tiltApproxSlope_;
    // approximated tilt correction parameter used to project r to z uncertainty
    double tiltApproxIntercept_;
    // In tilted barrel, constant assumed stub radial uncertainty * sqrt(12) in cm
    double tiltUncertaintyR_;
    // minimum representable stub phi uncertainty
    double mindPhi_;
    // maximum representable stub phi uncertainty
    double maxdPhi_;
    // minimum representable stub z uncertainty
    double mindZ_;
    // maximum representable stub z uncertainty
    double maxdZ_;
    // strip pitch of outer tracker sensors in cm
    double pitch2S_;
    // pixel pitch of outer tracker sensors in cm
    double pitchPS_;
    // strip length of outer tracker sensors in cm
    double length2S_;
    // pixel length of outer tracker sensors in cm
    double lengthPS_;
    // barrel layer limit |z| value to partition into tilted and untilted region
    std::vector<double> tiltedLayerLimitsZ_;
    // endcap disk limit r value to partition into PS and 2S region
    std::vector<double> psDiskLimitsR_;

    // Parameter specifying front-end
    edm::ParameterSet pSetFE_;
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

    // Parameter specifying DTC
    edm::ParameterSet pSetDTC_;
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
    //
    int numBarrelLayer_;
    // total number of output channel
    int dtcNumStreams_;
    // slot number changing from PS to 2S (default: 6)
    int slotLimitPS_;
    // slot number changing from 10 gbps to 5gbps (default: 3)
    int slotLimit10gbps_;

    // Parameter specifying TFP
    edm::ParameterSet pSetTFP_;
    // number of bist used for phi0
    int tfpWidthPhi0_;
    // umber of bist used for qOverPt
    int tfpWidthInv2R_;
    // number of bist used for cot(theta)
    int tfpWidthCot_;
    // number of bist used for z0
    int tfpWidthZ0_;
    // number of output links
    int tfpNumChannel_;

    // Parameter specifying GeometricProcessor
    edm::ParameterSet pSetGP_;
    // number of phi sectors used in hough transform
    int numSectorsPhi_;
    // number of eta sectors used in hough transform
    int numSectorsEta_;
    // # critical radius defining r-z sector shape in cm
    double chosenRofZ_;
    // range of stub z residual w.r.t. sector center which needs to be covered
    double neededRangeChiZ_;
    // fifo depth in stub router firmware
    int gpDepthMemory_;
    // defining r-z sector shape
    std::vector<double> boundariesEta_;

    // Parameter specifying HoughTransform
    edm::ParameterSet pSetHT_;
    // number of inv2R bins used in hough transform
    int htNumBinsInv2R_;
    // number of phiT bins used in hough transform
    int htNumBinsPhiT_;
    // required number of stub layers to form a candidate
    int htMinLayers_;
    // internal fifo depth
    int htDepthMemory_;

    // Parameter specifying MiniHoughTransform
    edm::ParameterSet pSetMHT_;
    // number of finer inv2R bins inside HT bin
    int mhtNumBinsInv2R_;
    // number of finer phiT bins inside HT bin
    int mhtNumBinsPhiT_;
    // number of dynamic load balancing steps
    int mhtNumDLBs_;
    // number of units per dynamic load balancing step
    int mhtNumDLBNodes_;
    // number of inputs per dynamic load balancing unit
    int mhtNumDLBChannel_;
    // required number of stub layers to form a candidate
    int mhtMinLayers_;

    // Parameter specifying ZHoughTransform
    edm::ParameterSet pSetZHT_;
    //number of used zT bins
    int zhtNumBinsZT_;
    // number of used cot bins
    int zhtNumBinsCot_;
    // number of stages
    int zhtNumStages_;
    // required number of stub layers to form a candidate
    int zhtMinLayers_;
    // max number of output tracks per node
    int zhtMaxTracks_;
    // cut on number of stub per layer for input candidates
    int zhtMaxStubsPerLayer_;

    // Parameter specifying KalmanFilter Input Formatter
    edm::ParameterSet pSetKFin_;
    // power of 2 multiplier of stub phi residual range
    int kfinShiftRangePhi_;
    // power of 2 multiplier of stub z residual range
    int kfinShiftRangeZ_;

    // Parameter specifying KalmanFilter
    edm::ParameterSet pSetKF_;
    // number of kf worker
    int kfNumWorker_;
    // required number of stub layers to form a track
    int kfMinLayers_;
    // maximum number of  layers added to a track
    int kfMaxLayers_;
    // search window of each track parameter in initial uncertainties
    double kfRangeFactor_;
    //
    int kfShiftInitialC00_;
    //
    int kfShiftInitialC11_;
    //
    int kfShiftInitialC22_;
    //
    int kfShiftInitialC33_;

    // Parameter specifying KalmanFilter Output Formatter
    edm::ParameterSet pSetKFOut_;
    // Conversion factor between dphi^2/weight and chi2rphi
    int kfoutchi2rphiConv_;
    // Conversion factor between dz^2/weight and chi2rz
    int kfoutchi2rzConv_;
    // Fraction of total dphi and dz ranges to calculate v0 and v1 LUT for
    int weightBinFraction_;
    // Constant used in FW to prevent 32-bit int overflow
    int dzTruncation_;
    // Constant used in FW to prevent 32-bit int overflow
    int dphiTruncation_;

    // Parameter specifying DuplicateRemoval
    edm::ParameterSet pSetDR_;
    // internal memory depth
    int drDepthMemory_;

    //
    // Derived constants
    //

    // true if tracker geometry and magnetic field supported
    bool configurationSupported_;
    // selector to partly select TPs for efficiency measurements
    TrackingParticleSelector tpSelector_;
    // selector to partly select TPs for fake and duplicate rate measurements
    TrackingParticleSelector tpSelectorLoose_;

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
    int numFrames_;
    // number of valid frames per 18 BX packet
    int numFramesIO_;
    // number of valid frames per 8 BX packet
    int numFramesFE_;
    // converts GeV in 1/cm
    double invPtToDphi_;
    // region size in rad
    double baseRegion_;

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
    // cut on zT
    double maxZT_;
    // cut on stub cot theta
    double maxCot_;
    // total number of sectors
    int numSectors_;
    // number of unused bits in GP output format
    int gpNumUnusedBits_;
    // cot(theta) of eta sectors
    std::vector<double> sectorCots_;

    // MHT

    // number of mht cells
    int mhtNumCells_;

    // ZHT

    // number of zht cells
    int zhtNumCells_;

    // KF

    int kfWidthLayerCount_;
  };

}  // namespace tt

EVENTSETUP_DATA_DEFAULT_RECORD(tt::Setup, tt::SetupRcd);

#endif
