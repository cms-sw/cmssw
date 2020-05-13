#include "L1Trigger/TrackerDTC/interface/Setup.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/L1TrackTrigger/interface/TTBV.h"

#include <cmath>
#include <algorithm>
#include <vector>
#include <set>
#include <string>
#include <sstream>

using namespace std;
using namespace edm;

namespace trackerDTC {

  Setup::Setup(const ParameterSet& iConfig,
               const MagneticField& magneticField,
               const TrackerGeometry& trackerGeometry,
               const TrackerTopology& trackerTopology,
               const TrackerDetToDTCELinkCablingMap& cablingMap,
               const StubAlgorithmOfficial& stubAlgorithm,
               const ParameterSet& pSetStubAlgorithm,
               const ParameterSet& pSetGeometryConfiguration,
               const ParameterSetID& pSetIdTTStubAlgorithm,
               const ParameterSetID& pSetIdGeometryConfiguration)
      : magneticField_(&magneticField),
        trackerGeometry_(&trackerGeometry),
        trackerTopology_(&trackerTopology),
        cablingMap_(&cablingMap),
        stubAlgorithm_(&stubAlgorithm),
        pSetSA_(&pSetStubAlgorithm),
        pSetGC_(&pSetGeometryConfiguration),
        pSetIdTTStubAlgorithm_(pSetIdTTStubAlgorithm),
        pSetIdGeometryConfiguration_(pSetIdGeometryConfiguration),
        // Parameter to check if configured Tracker Geometry is supported
        pSetSG_(iConfig.getParameter<ParameterSet>("SupportedGeometry")),
        sgXMLLabel_(pSetSG_.getParameter<string>("XMLLabel")),
        sgXMLPath_(pSetSG_.getParameter<string>("XMLPath")),
        sgXMLFile_(pSetSG_.getParameter<string>("XMLFile")),
        sgXMLVersions_(pSetSG_.getParameter<vector<string>>("XMLVersions")),
        // Parameter to check if Process History is consistent with process configuration
        pSetPH_(iConfig.getParameter<ParameterSet>("ProcessHistory")),
        phGeometryConfiguration_(pSetPH_.getParameter<string>("GeometryConfiguration")),
        phTTStubAlgorithm_(pSetPH_.getParameter<string>("TTStubAlgorithm")),
        // Common track finding parameter
        pSetTF_(iConfig.getParameter<ParameterSet>("TrackFinding")),
        beamWindowZ_(pSetTF_.getParameter<double>("BeamWindowZ")),
        matchedLayers_(pSetTF_.getParameter<int>("MatchedLayers")),
        matchedLayersPS_(pSetTF_.getParameter<int>("MatchedLayersPS")),
        unMatchedStubs_(pSetTF_.getParameter<int>("UnMatchedStubs")),
        unMatchedStubsPS_(pSetTF_.getParameter<int>("UnMatchedStubsPS")),
        // TMTT specific parameter
        pSetTMTT_(iConfig.getParameter<ParameterSet>("TMTT")),
        minPt_(pSetTMTT_.getParameter<double>("MinPt")),
        maxEta_(pSetTMTT_.getParameter<double>("MaxEta")),
        chosenRofPhi_(pSetTMTT_.getParameter<double>("ChosenRofPhi")),
        numLayers_(pSetTMTT_.getParameter<int>("NumLayers")),
        widthR_(pSetTMTT_.getParameter<int>("WidthR")),
        widthPhi_(pSetTMTT_.getParameter<int>("WidthPhi")),
        widthZ_(pSetTMTT_.getParameter<int>("WidthZ")),
        // Hybrid specific parameter
        pSetHybrid_(iConfig.getParameter<ParameterSet>("Hybrid")),
        hybridMinPt_(pSetHybrid_.getParameter<double>("MinPt")),
        hybridMaxEta_(pSetHybrid_.getParameter<double>("MaxEta")),
        hybridChosenRofPhi_(pSetHybrid_.getParameter<double>("ChosenRofPhi")),
        hybridNumLayers_(pSetHybrid_.getParameter<int>("NumLayers")),
        hybridNumRingsPS_(pSetHybrid_.getParameter<vector<int>>("NumRingsPS")),
        hybridWidthsR_(pSetHybrid_.getParameter<vector<int>>("WidthsR")),
        hybridWidthsZ_(pSetHybrid_.getParameter<vector<int>>("WidthsZ")),
        hybridWidthsPhi_(pSetHybrid_.getParameter<vector<int>>("WidthsPhi")),
        hybridWidthsAlpha_(pSetHybrid_.getParameter<vector<int>>("WidthsAlpha")),
        hybridWidthsBend_(pSetHybrid_.getParameter<vector<int>>("WidthsBend")),
        hybridRangesR_(pSetHybrid_.getParameter<vector<double>>("RangesR")),
        hybridRangesZ_(pSetHybrid_.getParameter<vector<double>>("RangesZ")),
        hybridRangesAlpha_(pSetHybrid_.getParameter<vector<double>>("RangesAlpha")),
        hybridLayerRs_(pSetHybrid_.getParameter<vector<double>>("LayerRs")),
        hybridDiskZs_(pSetHybrid_.getParameter<vector<double>>("DiskZs")),
        hybridDisk2SRsSet_(pSetHybrid_.getParameter<vector<ParameterSet>>("Disk2SRsSet")),
        // Parameter specifying TrackingParticle used for Efficiency measurements
        pSetTP_(iConfig.getParameter<ParameterSet>("TrackingParticle")),
        tpMaxEta_(pSetTP_.getParameter<double>("MaxEta")),
        tpMaxVertR_(pSetTP_.getParameter<double>("MaxVertR")),
        tpMaxVertZ_(pSetTP_.getParameter<double>("MaxVertZ")),
        tpMaxD0_(pSetTP_.getParameter<double>("MaxD0")),
        tpMinLayers_(pSetTP_.getParameter<int>("MinLayers")),
        tpMinLayersPS_(pSetTP_.getParameter<int>("MinLayersPS")),
        // Fimrware specific Parameter
        pSetFW_(iConfig.getParameter<ParameterSet>("Firmware")),
        numFramesInfra_(pSetFW_.getParameter<int>("NumFramesInfra")),
        freqLHC_(pSetFW_.getParameter<double>("FreqLHC")),
        freqBE_(pSetFW_.getParameter<double>("FreqBE")),
        tmpFE_(pSetFW_.getParameter<int>("TMP_FE")),
        tmpTFP_(pSetFW_.getParameter<int>("TMP_TFP")),
        speedOfLight_(pSetFW_.getParameter<double>("SpeedOfLight")),
        bField_(pSetFW_.getParameter<double>("BField")),
        bFieldError_(pSetFW_.getParameter<double>("BFieldError")),
        outerRadius_(pSetFW_.getParameter<double>("OuterRadius")),
        innerRadius_(pSetFW_.getParameter<double>("InnerRadius")),
        halfLength_(pSetFW_.getParameter<double>("HalfLength")),
        maxPitch_(pSetFW_.getParameter<double>("MaxPitch")),
        // Parmeter specifying front-end
        pSetFE_(iConfig.getParameter<ParameterSet>("FrontEnd")),
        widthBend_(pSetFE_.getParameter<int>("WidthBend")),
        widthCol_(pSetFE_.getParameter<int>("WidthCol")),
        widthRow_(pSetFE_.getParameter<int>("WidthRow")),
        baseBend_(pSetFE_.getParameter<double>("BaseBend")),
        baseCol_(pSetFE_.getParameter<double>("BaseCol")),
        baseRow_(pSetFE_.getParameter<double>("BaseRow")),
        baseWindowSize_(pSetFE_.getParameter<double>("BaseWindowSize")),
        bendCut_(pSetFE_.getParameter<double>("BendCut")),
        // Parmeter specifying DTC
        pSetDTC_(iConfig.getParameter<ParameterSet>("DTC")),
        numRegions_(pSetDTC_.getParameter<int>("NumRegions")),
        numOverlappingRegions_(pSetDTC_.getParameter<int>("NumOverlappingRegions")),
        numATCASlots_(pSetDTC_.getParameter<int>("NumATCASlots")),
        numDTCsPerRegion_(pSetDTC_.getParameter<int>("NumDTCsPerRegion")),
        numModulesPerDTC_(pSetDTC_.getParameter<int>("NumModulesPerDTC")),
        dtcNumRoutingBlocks_(pSetDTC_.getParameter<int>("NumRoutingBlocks")),
        dtcDepthMemory_(pSetDTC_.getParameter<int>("DepthMemory")),
        dtcWidthRowLUT_(pSetDTC_.getParameter<int>("WidthRowLUT")),
        dtcWidthQoverPt_(pSetDTC_.getParameter<int>("WidthQoverPt")),
        offsetDetIdDSV_(pSetDTC_.getParameter<int>("OffsetDetIdDSV")),
        offsetDetIdTP_(pSetDTC_.getParameter<int>("OffsetDetIdTP")),
        offsetLayerDisks_(pSetDTC_.getParameter<int>("OffsetLayerDisks")),
        offsetLayerId_(pSetDTC_.getParameter<int>("OffsetLayerId")),
        // Parmeter specifying GeometricProcessor
        pSetGP_(iConfig.getParameter<ParameterSet>("GeometricProcessor")),
        numSectorsPhi_(pSetGP_.getParameter<int>("NumSectorsPhi")),
        chosenRofZ_(pSetGP_.getParameter<double>("ChosenRofZ")),
        neededRangeChiZ_(pSetGP_.getParameter<double>("RangeChiZ")),
        gpDepthMemory_(pSetGP_.getParameter<int>("DepthMemory")),
        boundariesEta_(pSetGP_.getParameter<vector<double>>("BoundariesEta")),
        // Parmeter specifying HoughTransform
        pSetHT_(iConfig.getParameter<ParameterSet>("HoughTransform")),
        htNumBinsQoverPt_(pSetHT_.getParameter<int>("NumBinsQoverPt")),
        htNumBinsPhiT_(pSetHT_.getParameter<int>("NumBinsPhiT")),
        htMinLayers_(pSetHT_.getParameter<int>("MinLayers")),
        htDepthMemory_(pSetHT_.getParameter<int>("DepthMemory")),
        // Parmeter specifying MiniHoughTransform
        pSetMHT_(iConfig.getParameter<ParameterSet>("MiniHoughTransform")),
        mhtNumBinsQoverPt_(pSetMHT_.getParameter<int>("NumBinsQoverPt")),
        mhtNumBinsPhiT_(pSetMHT_.getParameter<int>("NumBinsPhiT")),
        mhtNumDLB_(pSetMHT_.getParameter<int>("NumDLB")),
        mhtMinLayers_(pSetMHT_.getParameter<int>("MinLayers")),
        // Parmeter specifying SeedFilter
        pSetSF_(iConfig.getParameter<ParameterSet>("SeedFilter")),
        sfPowerBaseCot_(pSetSF_.getParameter<int>("PowerBaseCot")),
        sfBaseDiffZ_(pSetSF_.getParameter<int>("BaseDiffZ")),
        sfMinLayers_(pSetSF_.getParameter<int>("MinLayers")),
        // Parmeter specifying KalmanFilter
        pSetKF_(iConfig.getParameter<ParameterSet>("KalmanFilter")),
        kfWidthLutInvPhi_(pSetKF_.getParameter<int>("WidthLutInvPhi")),
        kfWidthLutInvZ_(pSetKF_.getParameter<int>("WidthLutInvZ")),
        kfNumTracks_(pSetKF_.getParameter<int>("NumTracks")),
        kfMinLayers_(pSetKF_.getParameter<int>("MinLayers")),
        kfMaxLayers_(pSetKF_.getParameter<int>("MaxLayers")),
        kfMaxStubsPerLayer_(pSetKF_.getParameter<int>("MaxStubsPerLayer")),
        kfMaxSkippedLayers_(pSetKF_.getParameter<int>("MaxSkippedLayers")),
        kfBaseShiftr0_(pSetKF_.getParameter<int>("BaseShiftr0")),
        kfBaseShiftr02_(pSetKF_.getParameter<int>("BaseShiftr02")),
        kfBaseShiftv0_(pSetKF_.getParameter<int>("BaseShiftv0")),
        kfBaseShiftS00_(pSetKF_.getParameter<int>("BaseShiftS00")),
        kfBaseShiftS01_(pSetKF_.getParameter<int>("BaseShiftS01")),
        kfBaseShiftK00_(pSetKF_.getParameter<int>("BaseShiftK00")),
        kfBaseShiftK10_(pSetKF_.getParameter<int>("BaseShiftK10")),
        kfBaseShiftR00_(pSetKF_.getParameter<int>("BaseShiftR00")),
        kfBaseShiftInvR00_(pSetKF_.getParameter<int>("BaseShiftInvR00")),
        kfBaseShiftChi20_(pSetKF_.getParameter<int>("BaseShiftChi20")),
        kfBaseShiftC00_(pSetKF_.getParameter<int>("BaseShiftC00")),
        kfBaseShiftC01_(pSetKF_.getParameter<int>("BaseShiftC01")),
        kfBaseShiftC11_(pSetKF_.getParameter<int>("BaseShiftC11")),
        kfBaseShiftr1_(pSetKF_.getParameter<int>("BaseShiftr1")),
        kfBaseShiftr12_(pSetKF_.getParameter<int>("BaseShiftr12")),
        kfBaseShiftv1_(pSetKF_.getParameter<int>("BaseShiftv1")),
        kfBaseShiftS12_(pSetKF_.getParameter<int>("BaseShiftS12")),
        kfBaseShiftS13_(pSetKF_.getParameter<int>("BaseShiftS13")),
        kfBaseShiftK21_(pSetKF_.getParameter<int>("BaseShiftK21")),
        kfBaseShiftK31_(pSetKF_.getParameter<int>("BaseShiftK31")),
        kfBaseShiftR11_(pSetKF_.getParameter<int>("BaseShiftR11")),
        kfBaseShiftInvR11_(pSetKF_.getParameter<int>("BaseShiftInvR11")),
        kfBaseShiftChi21_(pSetKF_.getParameter<int>("BaseShiftChi21")),
        kfBaseShiftC22_(pSetKF_.getParameter<int>("BaseShiftC22")),
        kfBaseShiftC23_(pSetKF_.getParameter<int>("BaseShiftC23")),
        kfBaseShiftC33_(pSetKF_.getParameter<int>("BaseShiftC33")),
        kfBaseShiftChi2_(pSetKF_.getParameter<int>("BaseShiftChi2")),
        // Parmeter specifying DuplicateRemoval
        pSetDR_(iConfig.getParameter<ParameterSet>("DuplicateRemoval")),
        drDepthMemory_(pSetDR_.getParameter<int>("DepthMemory")),
        drWidthPhi0_(pSetDR_.getParameter<int>("WidthPhi0")),
        drWidthQoverPt_(pSetDR_.getParameter<int>("WidthQoverPt")),
        drWidthCot_(pSetDR_.getParameter<int>("WidthCot")),
        drWidthZ0_(pSetDR_.getParameter<int>("WidthZ0")) {
    configurationSupported_ = true;
    // check if bField is supported
    checkMagneticField();
    // check if geometry is supported
    checkGeometry();
    if (!configurationSupported_)
      return;
    // derive constants
    calculateConstants();
    // convert configuration of TTStubAlgorithm
    consumeStubAlgorithm();
    // create all possible encodingsBend
    encodingsBendPS_.reserve(maxWindowSize_ + 1);
    encodingsBend2S_.reserve(maxWindowSize_ + 1);
    encodeBend(encodingsBendPS_, true);
    encodeBend(encodingsBend2S_, false);
    // create encodingsLayerId
    encodingsLayerId_.reserve(numDTCsPerRegion_);
    encodeLayerId();
    // create sensor modules
    produceSensorModules();
  }

  // checks current configuration vs input sample configuration
  void Setup::checkHistory(const ProcessHistory& processHistory) const {
    const pset::Registry* psetRegistry = pset::Registry::instance();
    // check used TTStubAlgorithm in input producer
    checkHistory(processHistory, psetRegistry, phTTStubAlgorithm_, pSetIdTTStubAlgorithm_);
    // check used GeometryConfiguration in input producer
    checkHistory(processHistory, psetRegistry, phGeometryConfiguration_, pSetIdGeometryConfiguration_);
  }

  // checks consitency between history and current configuration for a specific module
  void Setup::checkHistory(const ProcessHistory& ph,
                           const pset::Registry* pr,
                           const string& label,
                           const ParameterSetID& pSetId) const {
    vector<pair<string, ParameterSet>> pSets;
    pSets.reserve(ph.size());
    for (const ProcessConfiguration& pc : ph) {
      const ParameterSet* pSet = pr->getMapped(pc.parameterSetID());
      if (pSet && pSet->exists(label))
        pSets.emplace_back(pc.processName(), pSet->getParameterSet(label));
    }
    if (pSets.empty()) {
      cms::Exception exception("BadConfiguration");
      exception << label << " not found in process history.";
      exception.addContext("tt::Setup::checkHistory");
      throw exception;
    }
    auto consistent = [&pSetId](const pair<string, ParameterSet>& p) { return p.second.id() == pSetId; };
    if (!all_of(pSets.begin(), pSets.end(), consistent)) {
      const ParameterSet& pSetProcess = getParameterSet(pSetId);
      cms::Exception exception("BadConfiguration");
      exception.addContext("tt::Setup::checkHistory");
      exception << label << " inconsistent with History." << endl;
      exception << "Current Configuration:" << endl << pSetProcess.dump() << endl;
      for (const pair<string, ParameterSet>& p : pSets)
        if (!consistent(p))
          exception << "Process " << p.first << " Configuration:" << endl << dumpDiff(p.second, pSetProcess) << endl;
      throw exception;
    }
  }

  // dumps pSetHistory where incosistent lines with pSetProcess are highlighted
  string Setup::dumpDiff(const ParameterSet& pSetHistory, const ParameterSet& pSetProcess) const {
    stringstream ssHistory, ssProcess, ss;
    ssHistory << pSetHistory.dump();
    ssProcess << pSetProcess.dump();
    string lineHistory, lineProcess;
    for (; getline(ssHistory, lineHistory) && getline(ssProcess, lineProcess);)
      ss << (lineHistory != lineProcess ? "\033[1;31m" : "") << lineHistory << "\033[0m" << endl;
    return ss.str();
  }

  // converts tk layout id into dtc id
  int Setup::dtcId(int tkLayoutId) const {
    checkTKLayoutId(tkLayoutId);
    const int tkId = tkLayoutId - 1;
    const int side = tkId / (numRegions_ * numATCASlots_);
    const int region = (tkId % (numRegions_ * numATCASlots_)) / numATCASlots_;
    const int slot = tkId % numATCASlots_;
    return region * numDTCsPerRegion_ + side * numATCASlots_ + slot;
  }

  // converts dtc id into tk layout id
  int Setup::tkLayoutId(int dtcId) const {
    checkDTCId(dtcId);
    const int slot = dtcId % numATCASlots_;
    const int region = dtcId / numDTCsPerRegion_;
    const int side = (dtcId % numDTCsPerRegion_) / numATCASlots_;
    return (side * numRegions_ + region) * numATCASlots_ + slot + 1;
  }

  // converts TFP identifier (region[0-8], channel[0-47]) into dtc id
  int Setup::dtcId(int tfpRegion, int tfpChannel) const {
    checkTFPIdentifier(tfpRegion, tfpChannel);
    const int dtcChannel = numOverlappingRegions_ - (tfpChannel / numDTCsPerRegion_) - 1;
    const int dtcBoard = tfpChannel % numDTCsPerRegion_;
    const int dtcRegion = tfpRegion - dtcChannel >= 0 ? tfpRegion - dtcChannel : tfpRegion - dtcChannel + numRegions_;
    return dtcRegion * numDTCsPerRegion_ + dtcBoard;
  }

  // checks if given DTC id is connected to PS or 2S sensormodules
  bool Setup::psModule(int dtcId) const {
    checkDTCId(dtcId);
    // from tklayout: first 3 are 10 gbps PS, next 3 are 5 gbps PS and residual 6 are 5 gbps 2S modules
    return slot(dtcId) < numATCASlots_ / 2;
  }

  // checks if given dtcId is connected to -z (false) or +z (true)
  bool Setup::side(int dtcId) const {
    checkDTCId(dtcId);
    const int side = (dtcId % numDTCsPerRegion_) / numATCASlots_;
    // from tkLayout: first 12 +z, next 12 -z
    return side == 0;
  }

  // ATCA slot number [0-11] of given dtcId
  int Setup::slot(int dtcId) const {
    checkDTCId(dtcId);
    return dtcId % numATCASlots_;
  }

  // sensor module for det id
  SensorModule* Setup::sensorModule(const DetId& detId) const {
    const auto it = detIdToSensorModule_.find(detId);
    if (it == detIdToSensorModule_.end()) {
      cms::Exception exception("NullPtr");
      exception << "Unknown DetId used.";
      exception.addContext("tt::Setup::sensorModule");
      throw exception;
    }
    return it->second;
  }

  // index = encoded bend, value = decoded bend for given window size and module type
  const vector<double>& Setup::encodingBend(int windowSize, bool psModule) const {
    const vector<vector<double>>& encodingsBend = psModule ? encodingsBendPS_ : encodingsBend2S_;
    return encodingsBend.at(windowSize);
  }

  // index = encoded layerId, inner value = decoded layerId for given dtcId or tfp channel
  const vector<int>& Setup::encodingLayerId(int dtcId) const {
    const int index = dtcId % numDTCsPerRegion_;
    return encodingsLayerId_.at(index);
  }

  // check if bField is supported
  void Setup::checkMagneticField() {
    const double bFieldES = magneticField_->inTesla(GlobalPoint(0., 0., 0.)).z();
    if (abs(bField_ - bFieldES) > bFieldError_) {
      configurationSupported_ = false;
      LogWarning("ConfigurationNotSupported")
          << "Magnetic Field from EventSetup (" << bFieldES << ") differs more then " << bFieldError_
          << " from supported value (" << bField_ << "). ";
    }
  }

  // check if geometry is supported
  void Setup::checkGeometry() {
    const vector<string>& geomXMLFiles = pSetGC_->getParameter<vector<string>>(sgXMLLabel_);
    string version;
    for (const string& geomXMLFile : geomXMLFiles) {
      const auto begin = geomXMLFile.find(sgXMLPath_) + sgXMLPath_.size();
      const auto end = geomXMLFile.find(sgXMLFile_);
      if (begin != string::npos && end != string::npos)
        version = geomXMLFile.substr(begin, end - begin - 1);
    }
    if (version.empty()) {
      cms::Exception exception("LogicError");
      exception << "No " << sgXMLPath_ << "*/" << sgXMLFile_ << " found in GeometryConfiguration";
      exception.addContext("tt::Setup::checkGeometry");
      throw exception;
    }
    if (find(sgXMLVersions_.begin(), sgXMLVersions_.end(), version) == sgXMLVersions_.end()) {
      configurationSupported_ = false;
      LogWarning("ConfigurationNotSupported")
          << "Geometry Configuration " << sgXMLPath_ << version << "/" << sgXMLFile_ << " is not supported. ";
    }
  }

  // convert configuration of TTStubAlgorithm
  void Setup::consumeStubAlgorithm() {
    numTiltedLayerRings_ = pSetSA_->getParameter<vector<double>>("NTiltedRings");
    windowSizeBarrelLayers_ = pSetSA_->getParameter<vector<double>>("BarrelCut");
    const auto& pSetsTiltedLayer = pSetSA_->getParameter<vector<ParameterSet>>("TiltedBarrelCutSet");
    const auto& pSetsEncapDisks = pSetSA_->getParameter<vector<ParameterSet>>("EndcapCutSet");
    windowSizeTiltedLayerRings_.reserve(pSetsTiltedLayer.size());
    for (const auto& pSet : pSetsTiltedLayer)
      windowSizeTiltedLayerRings_.emplace_back(pSet.getParameter<vector<double>>("TiltedCut"));
    windowSizeEndcapDisksRings_.reserve(pSetsEncapDisks.size());
    for (const auto& pSet : pSetsEncapDisks)
      windowSizeEndcapDisksRings_.emplace_back(pSet.getParameter<vector<double>>("EndcapCut"));
    maxWindowSize_ = -1;
    for (const auto& windowss : {windowSizeTiltedLayerRings_, windowSizeEndcapDisksRings_, {windowSizeBarrelLayers_}})
      for (const auto& windows : windowss)
        for (const auto& window : windows)
          maxWindowSize_ = max(maxWindowSize_, (int)(window / baseWindowSize_));
  }

  // create bend encodings
  void Setup::encodeBend(vector<vector<double>>& encodings, bool ps) const {
    for (int window = 0; window < maxWindowSize_ + 1; window++) {
      set<double> encoding;
      for (int bend = 0; bend < window + 1; bend++)
        encoding.insert(stubAlgorithm_->degradeBend(ps, window, bend));
      encodings.emplace_back(encoding.begin(), encoding.end());
    }
  }

  // create encodingsLayerId
  void Setup::encodeLayerId() {
    vector<vector<DTCELinkId>> dtcELinkIds(numDTCs_);
    for (vector<DTCELinkId>& dtcELinkId : dtcELinkIds)
      dtcELinkId.reserve(numModulesPerDTC_);
    for (const DTCELinkId& dtcLinkId : cablingMap_->getKnownDTCELinkIds())
      dtcELinkIds[dtcId(dtcLinkId.dtc_id())].push_back(dtcLinkId);
    for (int dtcBoard = 0; dtcBoard < numDTCsPerRegion_; dtcBoard++) {
      set<int> encodingLayerId;
      for (int region = 0; region < numRegions_; region++) {
        const int dtcId = region * numDTCsPerRegion_ + dtcBoard;
        for (const DTCELinkId& dtcLinkId : dtcELinkIds[dtcId]) {
          const DetId& detId = cablingMap_->dtcELinkIdToDetId(dtcLinkId)->second;
          const bool barrel = detId.subdetId() == StripSubdetector::TOB;
          const int layerId =
              barrel ? trackerTopology_->layer(detId) : trackerTopology_->tidWheel(detId) + offsetLayerDisks_;
          encodingLayerId.insert(layerId);
        }
      }
      // check configuration
      if ((int)encodingLayerId.size() > hybridNumLayers_) {
        cms::Exception exception("overflow");
        exception << "Cabling map connects more than " << hybridNumLayers_ << " layers to a DTC.";
        exception.addContext("tt::Setup::Setup");
        throw exception;
      }
      encodingsLayerId_.emplace_back(encodingLayerId.begin(), encodingLayerId.end());
    }
  }

  // create sensor modules
  void Setup::produceSensorModules() {
    sensorModules_.reserve(numModules_);
    dtcModules_ = vector<vector<SensorModule*>>(numDTCs_);
    for (vector<SensorModule*>& dtcModules : dtcModules_)
      dtcModules.reserve(numModulesPerDTC_);
    // loop over all tracker modules
    for (const DetId& detId : trackerGeometry_->detIds()) {
      // skip pixel detector
      if (detId.subdetId() == PixelSubdetector::PixelBarrel || detId.subdetId() == PixelSubdetector::PixelEndcap)
        continue;
      // skip multiple detIds per module
      if (!trackerTopology_->isLower(detId))
        continue;
      // lowerDetId - 1 = tk layout det id
      const DetId detIdTkLayout = detId + offsetDetIdTP_;
      // tk layout dtc id, lowerDetId - 1 = tk lyout det id
      const int tklId = cablingMap_->detIdToDTCELinkId(detIdTkLayout).first->second.dtc_id();
      // track trigger dtc id [0-215]
      const int dtcId = Setup::dtcId(tklId);
      // collection of so far connected modules to this dtc
      vector<SensorModule*>& dtcModules = dtcModules_[dtcId];
      // construct sendor module
      sensorModules_.emplace_back(*this, detId, dtcId, dtcModules.size());
      SensorModule* sensorModule = &sensorModules_.back();
      // store connection between detId and sensor module
      detIdToSensorModule_.emplace(detId, sensorModule);
      // store connection between dtcId and sensor module
      dtcModules.push_back(sensorModule);
    }
    for (vector<SensorModule*>& dtcModules : dtcModules_) {
      dtcModules.shrink_to_fit();
      // check configuration
      if ((int)dtcModules.size() > numModulesPerDTC_) {
        cms::Exception exception("overflow");
        exception << "Cabling map connects more than " << numModulesPerDTC_ << " modules to a DTC.";
        exception.addContext("tt::Setup::Setup");
        throw exception;
      }
    }
  }

  // derive constants
  void Setup::calculateConstants() {
    // emp
    const int numFramesPerBX = freqBE_ / freqLHC_;
    numFrames_ = numFramesPerBX * tmpTFP_ - 1;
    numFramesIO_ = numFramesPerBX * tmpTFP_ - numFramesInfra_;
    numFramesFE_ = numFramesPerBX * tmpFE_ - numFramesInfra_;
    // common track finding
    invPtToDphi_ = speedOfLight_ * bField_ / 2000.;
    baseRegion_ = 2. * M_PI / numRegions_;
    // gp
    baseSector_ = baseRegion_ / numSectorsPhi_;
    maxCot_ = sinh(maxEta_);
    maxZT_ = maxCot_ * chosenRofZ_;
    numSectorsEta_ = boundariesEta_.size() - 1;
    widthSectorEta_ = ceil(log2(numSectorsEta_));
    widthChiZ_ = ceil(log2(neededRangeChiZ_ / baseZ_));
    // ht
    htWidthQoverPt_ = ceil(log2(htNumBinsQoverPt_));
    htWidthPhiT_ = ceil(log2(htNumBinsPhiT_));
    const double rangeQoverPt = 2. * invPtToDphi_ / minPt_;
    htBaseQoverPt_ = rangeQoverPt / htNumBinsQoverPt_;
    htBasePhiT_ = baseSector_ / htNumBinsPhiT_;
    // tmtt
    widthLayer_ = ceil(log2(numLayers_));
    const double baseRgen = htBasePhiT_ / htBaseQoverPt_;
    const double rangeR = 2. * max(abs(outerRadius_ - chosenRofPhi_), abs(innerRadius_ - chosenRofPhi_));
    const int baseShiftR = ceil(log2(rangeR / baseRgen / pow(2., widthR_)));
    baseR_ = baseRgen * pow(2., baseShiftR);
    const double rangeZ = 2. * halfLength_;
    const int baseShiftZ = ceil(log2(rangeZ / baseR_ / pow(2., widthZ_)));
    baseZ_ = baseR_ * pow(2., baseShiftZ);
    const double rangePhiDTC = baseRegion_ + rangeQoverPt * baseR_ * pow(2., widthR_) / 4.;
    widthPhiDTC_ = widthPhi_ + ceil(log2(rangePhiDTC / baseRegion_));
    const int baseShiftPhi = ceil(log2(rangePhiDTC / htBasePhiT_ / pow(2., widthPhiDTC_)));
    basePhi_ = htBasePhiT_ * pow(2., baseShiftPhi);
    const double neededRangeChiPhi = 2. * htBasePhiT_;
    widthChiPhi_ = ceil(log2(neededRangeChiPhi / basePhi_));
    // hybrid
    const double hybridRangeQoverPt = 2. * invPtToDphi_ / hybridMinPt_;
    const double hybridRangeR =
        2. * max(abs(outerRadius_ - hybridChosenRofPhi_), abs(innerRadius_ - hybridChosenRofPhi_));
    const double hybridRangePhi = baseRegion_ + hybridRangeR * hybridRangeQoverPt / 2.;
    hybridWidthLayer_ = ceil(log2(hybridNumLayers_));
    hybridBasesZ_.reserve(SensorModule::NumTypes);
    for (int type = 0; type < SensorModule::NumTypes; type++)
      hybridBasesZ_.emplace_back(hybridRangesZ_.at(type) / pow(2., hybridWidthsZ_.at(type)));
    hybridBasesR_.reserve(SensorModule::NumTypes);
    for (int type = 0; type < SensorModule::NumTypes; type++)
      hybridBasesR_.emplace_back(hybridRangesR_.at(type) / pow(2., hybridWidthsR_.at(type)));
    hybridBasesR_[SensorModule::Disk2S] = 1.;
    hybridBasesPhi_.reserve(SensorModule::NumTypes);
    for (int type = 0; type < SensorModule::NumTypes; type++)
      hybridBasesPhi_.emplace_back(hybridRangePhi / pow(2., hybridWidthsPhi_.at(type)));
    hybridBasesAlpha_.reserve(SensorModule::NumTypes);
    for (int type = 0; type < SensorModule::NumTypes; type++)
      hybridBasesAlpha_.emplace_back(hybridRangesAlpha_.at(type) / pow(2., hybridWidthsAlpha_.at(type)));
    hybridNumsUnusedBits_.reserve(SensorModule::NumTypes);
    for (int type = 0; type < SensorModule::NumTypes; type++)
      hybridNumsUnusedBits_.emplace_back(TTBV::S - hybridWidthsR_.at(type) - hybridWidthsZ_.at(type) -
                                         hybridWidthsPhi_.at(type) - hybridWidthsAlpha_.at(type) -
                                         hybridWidthsBend_.at(type) - hybridWidthLayer_ - 1);
    hybridMaxCot_ = sinh(hybridMaxEta_);
    disk2SRs_.reserve(hybridDisk2SRsSet_.size());
    for (const auto& pSet : hybridDisk2SRsSet_)
      disk2SRs_.emplace_back(pSet.getParameter<vector<double>>("Disk2SRs"));
    // dtc
    numDTCs_ = numRegions_ * numDTCsPerRegion_;
    numDTCsPerTFP_ = numDTCsPerRegion_ * numOverlappingRegions_;
    numModules_ = numDTCs_ * numModulesPerDTC_;
    dtcNumModulesPerRoutingBlock_ = numModulesPerDTC_ / dtcNumRoutingBlocks_;
    dtcNumMergedRows_ = pow(2, widthRow_ - dtcWidthRowLUT_);
    const double maxRangeQoverPt = max(rangeQoverPt, hybridRangeQoverPt);
    const int baseShiftQoverPt = htWidthQoverPt_ - dtcWidthQoverPt_ + ceil(log2(maxRangeQoverPt / rangeQoverPt));
    dtcBaseQoverPt_ = htBaseQoverPt_ * pow(2., baseShiftQoverPt);
    const int baseDiffM = dtcWidthRowLUT_ - widthRow_;
    dtcBaseM_ = basePhi_ * pow(2., baseDiffM);
    const double x1 = pow(2, widthRow_) * baseRow_ * maxPitch_ / 2.;
    const double x0 = x1 - pow(2, dtcWidthRowLUT_) * baseRow_ * maxPitch_;
    const double maxM = atan2(x1, innerRadius_) - atan2(x0, innerRadius_);
    dtcWidthM_ = ceil(log2(maxM / dtcBaseM_));
    dtcNumUnusedBits_ = TTBV::S - 1 - widthR_ - widthPhiDTC_ - widthZ_ - 2 * htWidthQoverPt_ - 2 * widthSectorEta_ -
                        numSectorsPhi_ - widthLayer_;
    // mht
    mhtNumCells_ = mhtNumBinsQoverPt_ * mhtNumBinsPhiT_;
    mhtWidthQoverPt_ = ceil(log2(htNumBinsQoverPt_ * mhtNumBinsQoverPt_));
    mhtWidthPhiT_ = ceil(log2(htNumBinsPhiT_ * mhtNumBinsPhiT_));
    mhtBaseQoverPt_ = htBaseQoverPt_ / mhtNumBinsQoverPt_;
    mhtBasePhiT_ = htBasePhiT_ / mhtNumBinsPhiT_;
    // SF
    sfBaseCot_ = pow(2, sfPowerBaseCot_);
    sfBaseZT_ = baseZ_ * pow(2, sfBaseDiffZ_);
    // DR
    drBaseQoverPt_ = htBaseQoverPt_ * pow(2, htWidthQoverPt_ - drWidthQoverPt_);
    drBasePhi0_ = basePhi_ * pow(2, widthPhiDTC_ - drWidthPhi0_);
    drBaseCot_ = floor(log2(2. * maxCot_ * pow(2, -drWidthCot_)));
    drBaseZ0_ = baseZ_ * pow(2, ceil(log2(2. * beamWindowZ_ / baseZ_)) - drWidthZ0_);
    // KF
    kfBasex0_ = drBaseQoverPt_;
    kfBasex1_ = drBasePhi0_;
    kfBasex2_ = drBaseCot_;
    kfBasex3_ = drBaseZ0_;
    kfBasem0_ = basePhi_;
    kfBasem1_ = baseZ_;
    kfBaseH00_ = baseR_;
    kfBaseH12_ = baseR_;
    kfBaseChi2_ = pow(2, kfBaseShiftChi2_);
    kfBaser0_ = pow(2, kfBaseShiftr0_) * kfBasex1_;
    kfBaser02_ = pow(2, kfBaseShiftr02_) * kfBasex1_ * kfBasex1_;
    kfBasev0_ = pow(2, kfBaseShiftv0_) * kfBasex1_ * kfBasex1_;
    kfBaseS00_ = pow(2, kfBaseShiftS00_) * kfBasex0_ * kfBasex1_;
    kfBaseS01_ = pow(2, kfBaseShiftS01_) * kfBasex1_ * kfBasex1_;
    kfBaseK00_ = pow(2, kfBaseShiftK00_) * kfBasex0_ / kfBasex1_;
    kfBaseK10_ = pow(2, kfBaseShiftK10_);
    kfBaseR00_ = pow(2, kfBaseShiftR00_) * kfBasex1_ * kfBasex1_;
    kfBaseInvR00_ = pow(2, kfBaseShiftInvR00_) / kfBasex1_ / kfBasex1_;
    kfBaseChi20_ = pow(2, kfBaseShiftChi20_);
    kfBaseC00_ = pow(2, kfBaseShiftC00_) * kfBasex0_ * kfBasex0_;
    kfBaseC01_ = pow(2, kfBaseShiftC01_) * kfBasex0_ * kfBasex1_;
    kfBaseC11_ = pow(2, kfBaseShiftC11_) * kfBasex1_ * kfBasex1_;
    kfBaser1_ = pow(2, kfBaseShiftr1_) * kfBasex3_;
    kfBaser12_ = pow(2, kfBaseShiftr12_) * kfBasex3_ * kfBasex3_;
    kfBasev1_ = pow(2, kfBaseShiftv1_) * kfBasex3_ * kfBasex3_;
    kfBaseS12_ = pow(2, kfBaseShiftS12_) * kfBasex2_ * kfBasex3_;
    kfBaseS13_ = pow(2, kfBaseShiftS13_) * kfBasex3_ * kfBasex3_;
    kfBaseK21_ = pow(2, kfBaseShiftK21_) * kfBasex2_ / kfBasex3_;
    kfBaseK31_ = pow(2, kfBaseShiftK31_);
    kfBaseR11_ = pow(2, kfBaseShiftR11_) * kfBasex3_ * kfBasex3_;
    kfBaseInvR11_ = pow(2, kfBaseShiftInvR11_) / kfBasex3_ / kfBasex3_;
    kfBaseChi21_ = pow(2, kfBaseShiftChi21_);
    kfBaseC22_ = pow(2, kfBaseShiftC22_) * kfBasex2_ * kfBasex2_;
    kfBaseC23_ = pow(2, kfBaseShiftC23_) * kfBasex2_ * kfBasex3_;
    kfBaseC33_ = pow(2, kfBaseShiftC33_) * kfBasex3_ * kfBasex3_;
  }

  // returns bit accurate position of a stub from a given tfp identifier region [0-8] channel [0-47]
  GlobalPoint Setup::stubPos(bool hybrid, const TTDTC::Frame& frame, int tfpRegion, int tfpChannel) const {
    GlobalPoint p;
    if (frame.first.isNull())
      return p;
    TTBV bv(frame.second);
    if (hybrid) {
      const DetId& detId = frame.first->getDetId();
      const int dtcId = Setup::dtcId(tfpRegion, tfpChannel);
      const bool barrel = detId.subdetId() == StripSubdetector::TOB;
      const bool psModule = Setup::psModule(dtcId);
      const int layerId =
          (barrel ? trackerTopology_->layer(detId) : trackerTopology_->tidWheel(detId)) - offsetLayerId_;
      const bool side = Setup::side(dtcId);
      SensorModule::Type type;
      if (barrel && psModule)
        type = SensorModule::BarrelPS;
      if (barrel && !psModule)
        type = SensorModule::Barrel2S;
      if (!barrel && psModule)
        type = SensorModule::DiskPS;
      if (!barrel && !psModule)
        type = SensorModule::Disk2S;
      const int widthBend = hybridWidthsBend_.at(type);
      const int widthAlpha = hybridWidthsAlpha_.at(type);
      const int widthPhi = hybridWidthsPhi_.at(type);
      const int widthZ = hybridWidthsZ_.at(type);
      const int widthR = hybridWidthsR_.at(type);
      const double basePhi = hybridBasesPhi_.at(type);
      const double baseZ = hybridBasesZ_.at(type);
      const double baseR = hybridBasesR_.at(type);
      // parse bit vector
      bv >>= 1 + hybridWidthLayer_ + widthBend + widthAlpha;
      double phi = (bv.val(widthPhi, 0, true) + .5) * basePhi;
      bv >>= widthPhi;
      double z = (bv.val(widthZ, 0, true) + .5) * baseZ;
      bv >>= widthZ;
      double r = (bv.val(widthR, 0, barrel) + .5) * baseR;
      if (barrel) {
        r += hybridLayerRs_.at(layerId);
      } else {
        z += hybridDiskZs_.at(layerId) * (side ? 1. : -1.);
      }
      phi = deltaPhi(phi + tfpRegion * baseRegion_);
      if (type == SensorModule::Disk2S) {
        r = bv.val(widthR);
        r = disk2SRs_.at(layerId).at((int)r);
      }
      p = GlobalPoint(GlobalPoint::Cylindrical(r, phi, z));
    } else {
      bv >>= 2 * htWidthQoverPt_ + 2 * widthSectorEta_ + numSectorsPhi_ + widthLayer_;
      double z = (bv.val(widthZ_, 0, true) + .5) * baseZ_;
      bv >>= widthZ_;
      double phi = (bv.val(widthPhiDTC_, 0, true) + .5) * basePhi_;
      bv >>= widthPhiDTC_;
      double r = (bv.val(widthR_, 0, true) + .5) * baseR_;
      bv >>= widthR_;
      r = r + chosenRofPhi_;
      phi = deltaPhi(phi + tfpRegion * baseRegion_);
      p = GlobalPoint(GlobalPoint::Cylindrical(r, phi, z));
    }
    return p;
  }

  // returns global TTStub position
  GlobalPoint Setup::stubPos(const TTStubRef& ttStubRef) const {
    const DetId detId = ttStubRef->getDetId() + offsetDetIdDSV_;
    const GeomDetUnit* det = trackerGeometry_->idToDetUnit(detId);
    const PixelTopology* topol =
        dynamic_cast<const PixelTopology*>(&(dynamic_cast<const PixelGeomDetUnit*>(det)->specificTopology()));
    const Plane& plane = dynamic_cast<const PixelGeomDetUnit*>(det)->surface();
    const MeasurementPoint& mp = ttStubRef->clusterRef(0)->findAverageLocalCoordinatesCentered();
    return plane.toGlobal(topol->localPosition(mp));
  }

  // range check of dtc id
  void Setup::checkDTCId(int dtcId) const {
    if (dtcId < 0 || dtcId >= numDTCsPerRegion_ * numRegions_) {
      cms::Exception exception("out_of_range");
      exception.addContext("trackerDTC::Setup::checkDTCId");
      exception << "Used DTC Id (" << dtcId << ") "
                << "is out of range 0 to " << numDTCsPerRegion_ * numRegions_ - 1 << ".";
      throw exception;
    }
  }

  // range check of tklayout id
  void Setup::checkTKLayoutId(int tkLayoutId) const {
    if (tkLayoutId <= 0 || tkLayoutId > numDTCsPerRegion_ * numRegions_) {
      cms::Exception exception("out_of_range");
      exception.addContext("trackerDTC::Setup::checkTKLayoutId");
      exception << "Used TKLayout Id (" << tkLayoutId << ") "
                << "is out of range 1 to " << numDTCsPerRegion_ * numRegions_ << ".";
      throw exception;
    }
  }

  // range check of tfp identifier
  void Setup::checkTFPIdentifier(int tfpRegion, int tfpChannel) const {
    const bool oorRegion = tfpRegion >= numRegions_ || tfpRegion < 0;
    const bool oorChannel = tfpChannel >= numDTCsPerTFP_ || tfpChannel < 0;
    if (oorRegion || oorChannel) {
      cms::Exception exception("out_of_range");
      exception.addContext("trackerDTC::Setup::checkTFPIdentifier");
      if (oorRegion)
        exception << "Requested Processing Region "
                  << "(" << tfpRegion << ") "
                  << "is out of range 0 to " << numRegions_ - 1 << ".";
      if (oorChannel)
        exception << "Requested TFP Channel "
                  << "(" << tfpChannel << ") "
                  << "is out of range 0 to " << numDTCsPerTFP_ - 1 << ".";
      throw exception;
    }
  }

}  // namespace trackerDTC