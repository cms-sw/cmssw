#ifndef L1Trigger_TrackerDTC_Setup_h
#define L1Trigger_TrackerDTC_Setup_h

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
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_official.h"
#include "MagneticField/Engine/interface/MagneticField.h"
#include "CondFormats/SiPhase2TrackerObjects/interface/TrackerDetToDTCELinkCablingMap.h"

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "L1Trigger/TrackerDTC/interface/SetupRcd.h"
#include "L1Trigger/TrackerDTC/interface/SensorModule.h"

#include <vector>
#include <unordered_map>

namespace trackerDTC {

  typedef TTStubAlgorithm<Ref_Phase2TrackerDigi_> StubAlgorithm;
  typedef TTStubAlgorithm_official<Ref_Phase2TrackerDigi_> StubAlgorithmOfficial;
  // handles 2 pi overflow
  inline double deltaPhi(double lhs, double rhs = 0.) { return reco::deltaPhi(lhs, rhs); }

  /*! \class  trackerDTC::Setup
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
    // returns bit accurate position of a stub from a given tfp identifier region [0-8] channel [0-47]
    GlobalPoint stubPos(bool hybrid, const TTDTC::Frame& frame, int tfpRegion, int tfpChannel) const;
    // returns global TTStub position
    GlobalPoint stubPos(const TTStubRef& ttStubRef) const;
    // empty trackerDTC EDProduct
    TTDTC ttDTC() const { return TTDTC(numRegions_, numOverlappingRegions_, numDTCsPerRegion_); }

    // Common track finding parameter

    // half lumi region size in cm
    double beamWindowZ() const { return beamWindowZ_; }
    // number of frames betwen 2 resets of 18 BX packets
    int numFrames() const { return numFrames_; }
    // number of valid frames per 18 BX packet
    int numFramesIO() const { return numFramesIO_; }
    // number of valid frames per 8 BX packet
    int numFramesFE() const { return numFramesFE_; }
    // converts GeV in 1/cm
    double invPtToDphi() const { return invPtToDphi_; }
    // region size in rad
    double baseRegion() const { return baseRegion_; }
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
    int widthR() const { return widthR_; }
    // number of bits used for stub phi w.r.t. phi sector centre
    int widthPhi() const { return widthPhi_; }
    // number of bits used for stub z
    int widthZ() const { return widthZ_; }
    // number of bits used for stub layer id
    int widthLayer() const { return widthLayer_; }
    // internal stub r precision in cm
    double baseR() const { return baseR_; }
    // internal stub z precision in cm
    double baseZ() const { return baseZ_; }
    // internal stub phi precision in rad
    double basePhi() const { return basePhi_; }
    // number of padded 0s in output data format
    int dtcNumUnusedBits() const { return dtcNumUnusedBits_; }

    // Hybrid specific parameter

    // cut on stub and TP pt, also defines region overlap shape in GeV
    double hybridMinPt() const { return hybridMinPt_; }
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
    int hybridWidthLayer() const { return hybridWidthLayer_; }
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
    // number of bits used for stub qOverPt. lut addr is col + bend = 11 => 1 BRAM -> 18 bits for min and max val -> 9
    int dtcWidthQoverPt() const { return dtcWidthQoverPt_; }
    // tk layout det id minus DetSetVec->detId
    int offsetDetIdDSV() const { return offsetDetIdDSV_; }
    // tk layout det id minus TrackerTopology lower det id
    int offsetDetIdTP() const { return offsetDetIdTP_; }
    // offset in layer ids between barrel layer and endcap disks
    int offsetLayerDisks() const { return offsetLayerDisks_; }
    // offset between 0 and smallest layer id (barrel layer 1)
    int offsetLayerId() const { return offsetLayerId_; }
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
    // internal stub q over pt precision in 1 /cm
    double dtcBaseQoverPt() const { return dtcBaseQoverPt_; }
    // phi of row slope precision in rad / pitch unit
    double dtcBaseM() const { return dtcBaseM_; }
    // number of bits for internal stub phi
    int widthPhiDTC() const { return widthPhiDTC_; }
    // sensor modules connected to given dtc id
    const std::vector<SensorModule*>& dtcModules(int dtcId) const { return dtcModules_.at(dtcId); }
    // index = encoded layerId, inner value = decoded layerId for given tfp channel [0-47]
    const std::vector<int>& encodingLayerId(int tfpChannel) const;

    // Parameter specifying GeometricProcessor

    // number of phi sectors used in hough transform
    int numSectorsPhi() const { return numSectorsPhi_; }
    // number of eta sectors used in hough transform
    int numSectorsEta() const { return numSectorsEta_; }
    // # critical radius defining r-z sector shape in cm
    double chosenRofZ() const { return chosenRofZ_; }
    // fifo depth in stub router firmware
    int gpDepthMemory() const { return gpDepthMemory_; }
    // defining r-z sector shape
    double boundarieEta(int eta) const { return boundariesEta_.at(eta); }
    // phi sector size in rad
    double baseSector() const { return baseSector_; }
    // cut on zT
    double maxZT() const { return maxZT_; }
    // cut on stub cot theta
    double maxCot() const { return maxCot_; }
    // number of bits used for internal stub sector eta
    int widthSectorEta() const { return widthSectorEta_; }
    // number of bits to represent z residual w.r.t. sector center
    int widthChiZ() const { return widthChiZ_; }

    // Parameter specifying HoughTransform

    // number of qOverPt bins used in hough transform
    int htNumBinsQoverPt() const { return htNumBinsQoverPt_; }
    // number of phiT bins used in hough transform
    int htNumBinsPhiT() const { return htNumBinsPhiT_; }
    // required number of stub layers to form a candidate
    int htMinLayers() const { return htMinLayers_; }
    // internal fifo depth
    int htDepthMemory() const { return htDepthMemory_; }
    // number of bits used for candidate q over pt
    int htWidthQoverPt() const { return htWidthQoverPt_; }
    // number of bits used for candidate phiT
    int htWidthPhiT() const { return htWidthPhiT_; }
    // number of bits to represent phi residual w.r.t. ht candiate
    int widthChiPhi() const { return widthChiPhi_; }
    // q over pt bin width precision in 1 /cm
    double htBaseQoverPt() const { return htBaseQoverPt_; }
    // phiT bin width in rad
    double htBasePhiT() const { return htBasePhiT_; }

    // Parameter specifying MiniHoughTransform

    // number of finer qOverPt bins inside HT bin
    int mhtNumBinsQoverPt() const { return mhtNumBinsQoverPt_; }
    // number of finer phiT bins inside HT bin
    int mhtNumBinsPhiT() const { return mhtNumBinsPhiT_; }
    // number of dynamic load balancing steps
    int mhtNumDLB() const { return mhtNumDLB_; }
    // required number of stub layers to form a candidate
    int mhtMinLayers() const { return mhtMinLayers_; }
    // number of mht cells
    int mhtNumCells() const { return mhtNumCells_; }
    // number of bits used for candidate q over pt
    int mhtWidthQoverPt() const { return mhtWidthQoverPt_; }
    // number of bits used for candidate phiT
    int mhtWidthPhiT() const { return mhtWidthPhiT_; }
    // q over pt bin width precision in 1 /cm
    double mhtBaseQoverPt() const { return mhtBaseQoverPt_; }
    // phiT bin width in rad
    double mhtBasePhiT() const { return mhtBasePhiT_; }

    // Parameter specifying SeedFilter

    // required number of stub layers to form a candidate
    int sfMinLayers() const { return sfMinLayers_; }
    // cot(theta) precision
    double sfBaseCot() const { return sfBaseCot_; }
    // zT precision in cm
    double sfBaseZT() const { return sfBaseZT_; }

    // Parameter specifying KalmanFilter

    // number of bits for internal reciprocal look up
    int kfWidthLutInvPhi() const { return kfWidthLutInvPhi_; }
    // number of bits for internal reciprocal look up
    int kfWidthLutInvZ() const { return kfWidthLutInvZ_; }
    // cut on number of input candidates
    int kfNumTracks() const { return kfNumTracks_; }
    // required number of stub layers to form a track
    int kfMinLayers() const { return kfMinLayers_; }
    // maximum number of  layers added to a track
    int kfMaxLayers() const { return kfMaxLayers_; }
    // cut on number of stub per layer for input candidates
    int kfMaxStubsPerLayer() const { return kfMaxStubsPerLayer_; }
    // maximum allowed skipped layers from inside to outside to form a track
    int kfMaxSkippedLayers() const { return kfMaxSkippedLayers_; }
    double kfBasem0() const { return kfBasem0_; }
    double kfBasem1() const { return kfBasem1_; }
    double kfBasev0() const { return kfBasev0_; }
    double kfBasev1() const { return kfBasev1_; }
    double kfBasex0() const { return kfBasex0_; }
    double kfBasex1() const { return kfBasex1_; }
    double kfBasex2() const { return kfBasex2_; }
    double kfBasex3() const { return kfBasex3_; }
    double kfBaseH00() const { return kfBaseH00_; }
    double kfBaseH12() const { return kfBaseH12_; }
    double kfBaser0() const { return kfBaser0_; }
    double kfBaser1() const { return kfBaser1_; }
    double kfBaser02() const { return kfBaser02_; }
    double kfBaser12() const { return kfBaser12_; }
    double kfBaseS00() const { return kfBaseS00_; }
    double kfBaseS01() const { return kfBaseS01_; }
    double kfBaseS12() const { return kfBaseS12_; }
    double kfBaseS13() const { return kfBaseS13_; }
    double kfBaseR00() const { return kfBaseR00_; }
    double kfBaseR11() const { return kfBaseR11_; }
    double kfBaseInvR00() const { return kfBaseInvR00_; }
    double kfBaseInvR11() const { return kfBaseInvR11_; }
    double kfBaseK00() const { return kfBaseK00_; }
    double kfBaseK10() const { return kfBaseK10_; }
    double kfBaseK21() const { return kfBaseK21_; }
    double kfBaseK31() const { return kfBaseK31_; }
    double kfBaseC00() const { return kfBaseC00_; }
    double kfBaseC01() const { return kfBaseC01_; }
    double kfBaseC11() const { return kfBaseC11_; }
    double kfBaseC22() const { return kfBaseC22_; }
    double kfBaseC23() const { return kfBaseC23_; }
    double kfBaseC33() const { return kfBaseC33_; }
    double kfBaseChi20() const { return kfBaseChi20_; }
    double kfBaseChi21() const { return kfBaseChi21_; }
    double kfBaseChi2() const { return kfBaseChi2_; }

    // Parameter specifying DuplicateRemoval

    // internal memory depth
    int drDepthMemory() const { return drDepthMemory_; }
    // number of bist used for phi0
    int drWidthPhi0() const { return drWidthPhi0_; }
    // umber of bist used for qOverPt
    int drWidthQoverPt() const { return drWidthQoverPt_; }
    // number of bist used for cot(theta)
    int drWidthCot() const { return drWidthCot_; }
    // number of bist used for z0
    int drWidthZ0() const { return drWidthZ0_; }
    double drBaseQoverPt() const { return drBaseQoverPt_; }
    double drBasePhi0() const { return drBasePhi0_; }
    double drBaseCot() const { return drBaseCot_; }
    double drBaseZ0() const { return drBaseZ0_; }

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
    // create encodingsLayerId
    void encodeLayerId();
    // create sensor modules
    void produceSensorModules();
    // range check of dtc id
    void checkDTCId(int dtcId) const;
    // range check of tklayout id
    void checkTKLayoutId(int tkLayoutId) const;
    // range check of tfp identifier
    void checkTFPIdentifier(int tfpRegion, int tfpChannel) const;

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
    int widthR_;
    // number of bits used for stub phi w.r.t. phi sector centre
    int widthPhi_;
    // number of bits used for stub z
    int widthZ_;

    // Hybrid specific parameter
    edm::ParameterSet pSetHybrid_;
    // cut on stub and TP pt, also defines region overlap shape in GeV
    double hybridMinPt_;
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

    // Parameter specifying TrackingParticle used for Efficiency measurements
    edm::ParameterSet pSetTP_;
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

    // Fimrware specific Parameter
    edm::ParameterSet pSetFW_;
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
    // number of bits used for stub qOverPt. lut addr is col + bend = 11 => 1 BRAM -> 18 bits for min and max val -> 9
    int dtcWidthQoverPt_;
    // tk layout det id minus DetSetVec->detId
    int offsetDetIdDSV_;
    // tk layout det id minus TrackerTopology lower det id
    int offsetDetIdTP_;
    // offset in layer ids between barrel layer and endcap disks
    int offsetLayerDisks_;
    // offset between 0 and smallest layer id (barrel layer 1)
    int offsetLayerId_;

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
    // number of qOverPt bins used in hough transform
    int htNumBinsQoverPt_;
    // number of phiT bins used in hough transform
    int htNumBinsPhiT_;
    // required number of stub layers to form a candidate
    int htMinLayers_;
    // internal fifo depth
    int htDepthMemory_;

    // Parameter specifying MiniHoughTransform
    edm::ParameterSet pSetMHT_;
    // number of finer qOverPt bins inside HT bin
    int mhtNumBinsQoverPt_;
    // number of finer phiT bins inside HT bin
    int mhtNumBinsPhiT_;
    // number of dynamic load balancing steps
    int mhtNumDLB_;
    // required number of stub layers to form a candidate
    int mhtMinLayers_;

    // Parameter specifying SeedFilter
    edm::ParameterSet pSetSF_;
    // used cot(Theta) bin width = 2 ** this
    int sfPowerBaseCot_;
    // used zT bin width = baseZ * 2 ** this
    int sfBaseDiffZ_;
    // required number of stub layers to form a candidate
    int sfMinLayers_;

    // Parameter specifying KalmanFilter
    edm::ParameterSet pSetKF_;
    // number of bits for internal reciprocal look up
    int kfWidthLutInvPhi_;
    // number of bits for internal reciprocal look up
    int kfWidthLutInvZ_;
    // cut on number of input candidates
    int kfNumTracks_;
    // required number of stub layers to form a track
    int kfMinLayers_;
    // maximum number of  layers added to a track
    int kfMaxLayers_;
    // cut on number of stub per layer for input candidates
    int kfMaxStubsPerLayer_;
    // maximum allowed skipped layers from inside to outside to form a track
    int kfMaxSkippedLayers_;
    int kfBaseShiftr0_;
    int kfBaseShiftr02_;
    int kfBaseShiftv0_;
    int kfBaseShiftS00_;
    int kfBaseShiftS01_;
    int kfBaseShiftK00_;
    int kfBaseShiftK10_;
    int kfBaseShiftR00_;
    int kfBaseShiftInvR00_;
    int kfBaseShiftChi20_;
    int kfBaseShiftC00_;
    int kfBaseShiftC01_;
    int kfBaseShiftC11_;
    int kfBaseShiftr1_;
    int kfBaseShiftr12_;
    int kfBaseShiftv1_;
    int kfBaseShiftS12_;
    int kfBaseShiftS13_;
    int kfBaseShiftK21_;
    int kfBaseShiftK31_;
    int kfBaseShiftR11_;
    int kfBaseShiftInvR11_;
    int kfBaseShiftChi21_;
    int kfBaseShiftC22_;
    int kfBaseShiftC23_;
    int kfBaseShiftC33_;
    int kfBaseShiftChi2_;

    // Parameter specifying DuplicateRemoval
    edm::ParameterSet pSetDR_;
    // internal memory depth
    int drDepthMemory_;
    // number of bist used for phi0
    int drWidthPhi0_;
    // umber of bist used for qOverPt
    int drWidthQoverPt_;
    // number of bist used for cot(theta)
    int drWidthCot_;
    // number of bist used for z0
    int drWidthZ0_;

    //
    // Derived constants
    //

    // true if tracker geometry and magnetic field supported
    bool configurationSupported_;

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
    int widthLayer_;
    // internal stub r precision in cm
    double baseR_;
    // internal stub z precision in cm
    double baseZ_;
    // internal stub phi precision in rad
    double basePhi_;
    // number of padded 0s in output data format
    int dtcNumUnusedBits_;

    // hybrid

    // number of bits used for stub layer id
    int hybridWidthLayer_;
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
    // internal stub q over pt precision in 1 /cm
    double dtcBaseQoverPt_;
    // phi of row slope precision in rad / pitch unit
    double dtcBaseM_;
    // number of bits for internal stub phi
    int widthPhiDTC_;
    // outer index = module window size, inner index = encoded bend, inner value = decoded bend, for ps modules
    std::vector<std::vector<double>> encodingsBendPS_;
    // outer index = module window size, inner index = encoded bend, inner value = decoded bend, for 2s modules
    std::vector<std::vector<double>> encodingsBend2S_;
    // outer index = dtc id in region, inner index = encoded layerId, inner value = decoded layerId
    std::vector<std::vector<int>> encodingsLayerId_;
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
    // number of bits used for internal stub sector eta
    int widthSectorEta_;
    // number of bits to represent z residual w.r.t. sector center
    int widthChiZ_;

    // HT

    // number of bits used for candidate q over pt
    int htWidthQoverPt_;
    // number of bits used for candidate phiT
    int htWidthPhiT_;
    // number of bits to represent phi residual w.r.t. ht candiate
    int widthChiPhi_;
    // q over pt bin width precision in 1 /cm
    double htBaseQoverPt_;
    // phiT bin width precision in rad
    double htBasePhiT_;

    // MHT

    // number of mht cells
    int mhtNumCells_;
    // number of bits used for candidate q over pt
    int mhtWidthQoverPt_;
    // number of bits used for candidate phiT
    int mhtWidthPhiT_;
    // q over pt bin width precision in 1 /cm
    double mhtBaseQoverPt_;
    // phiT bin width in rad
    double mhtBasePhiT_;

    // SF

    // cot(theta) precision
    double sfBaseCot_;
    // zT precision in cm
    double sfBaseZT_;

    // KF

    double kfBasem0_;
    double kfBasem1_;
    double kfBasev0_;
    double kfBasev1_;
    double kfBasex0_;
    double kfBasex1_;
    double kfBasex2_;
    double kfBasex3_;
    double kfBasex4_;
    double kfBaseH00_;
    double kfBaseH04_;
    double kfBaseH12_;
    double kfBaser0_;
    double kfBaser1_;
    double kfBaser02_;
    double kfBaser12_;
    double kfBaseS00_;
    double kfBaseS01_;
    double kfBaseS04_;
    double kfBaseS12_;
    double kfBaseS13_;
    double kfBaseR00_;
    double kfBaseR11_;
    double kfBaseInvR00_;
    double kfBaseInvR11_;
    double kfBaseK00_;
    double kfBaseK10_;
    double kfBaseK21_;
    double kfBaseK31_;
    double kfBaseK40_;
    double kfBaseC00_;
    double kfBaseC01_;
    double kfBaseC04_;
    double kfBaseC11_;
    double kfBaseC14_;
    double kfBaseC44_;
    double kfBaseC22_;
    double kfBaseC23_;
    double kfBaseC33_;
    double kfBaseChi20_;
    double kfBaseChi21_;
    double kfBaseChi2_;

    // DR

    double drBaseQoverPt_;
    double drBasePhi0_;
    double drBaseCot_;
    double drBaseZ0_;
  };

}  // namespace trackerDTC

EVENTSETUP_DATA_DEFAULT_RECORD(trackerDTC::Setup, trackerDTC::SetupRcd);

#endif