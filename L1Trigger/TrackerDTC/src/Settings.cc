#include "L1Trigger/TrackerDTC/interface/Settings.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/Registry.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTDTC.h"
#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_official.h"
#include "L1Trigger/TrackerDTC/interface/Module.h"

#include <cmath>
#include <iterator>
#include <algorithm>
#include <set>
#include <vector>
#include <memory>
#include <unordered_map>
#include <utility>

using namespace std;
using namespace edm;

namespace trackerDTC {

  Settings::Settings(const ParameterSet& iConfig)
      :  //TrackerDTCProducer parameter sets
        paramsED_(iConfig.getParameter<ParameterSet>("ParamsED")),
        paramsRouter_(iConfig.getParameter<ParameterSet>("ParamsRouter")),
        paramsConverter_(iConfig.getParameter<ParameterSet>("ParamsConverter")),
        paramsTracker_(iConfig.getParameter<ParameterSet>("ParamsTracker")),
        paramsFW_(iConfig.getParameter<ParameterSet>("ParamsFW")),
        paramsFormat_(iConfig.getParameter<ParameterSet>("ParamsFormat")),
        paramsAnalyzer_(iConfig.getParameter<ParameterSet>("ParamsAnalyzer")),
        paramsTP_(iConfig.getParameter<ParameterSet>("ParamsTP")),
        // ED parameter
        inputTagTTStubDetSetVec_(paramsED_.getParameter<InputTag>("InputTagTTStubDetSetVec")),
        inputTagMagneticField_(paramsED_.getParameter<ESInputTag>("InputTagMagneticField")),
        inputTagTrackerGeometry_(paramsED_.getParameter<ESInputTag>("InputTagTrackerGeometry")),
        inputTagTrackerTopology_(paramsED_.getParameter<ESInputTag>("InputTagTrackerTopology")),
        inputTagCablingMap_(paramsED_.getParameter<ESInputTag>("InputTagCablingMap")),
        inputTagTTStubAlgorithm_(paramsED_.getParameter<ESInputTag>("InputTagTTStubAlgorithm")),
        inputTagGeometryConfiguration_(paramsED_.getParameter<ESInputTag>("InputTagGeometryConfiguration")),
        supportedTrackerXMLPSet_(paramsED_.getParameter<string>("SupportedTrackerXMLPSet")),
        supportedTrackerXMLPath_(paramsED_.getParameter<string>("SupportedTrackerXMLPath")),
        supportedTrackerXMLFile_(paramsED_.getParameter<string>("SupportedTrackerXMLFile")),
        supportedTrackerXMLVersions_(paramsED_.getParameter<vector<string>>("SupportedTrackerXMLVersions")),
        productBranchAccepted_(paramsED_.getParameter<string>("ProductBranchAccepted")),
        productBranchLost_(paramsED_.getParameter<string>("ProductBranchLost")),
        dataFormat_(paramsED_.getParameter<string>("DataFormat")),
        offsetDetIdDSV_(paramsED_.getParameter<int>("OffsetDetIdDSV")),
        offsetDetIdTP_(paramsED_.getParameter<int>("OffsetDetIdTP")),
        offsetLayerDisks_(paramsED_.getParameter<int>("OffsetLayerDisks")),
        offsetLayerId_(paramsED_.getParameter<int>("OffsetLayerId")),
        checkHistory_(paramsED_.getParameter<bool>("CheckHistory")),
        processName_(paramsED_.getParameter<string>("ProcessName")),
        productLabel_(paramsED_.getParameter<string>("ProductLabel") + "@"),
        // Router parameter
        enableTruncation_(paramsRouter_.getParameter<bool>("EnableTruncation")),
        freqDTC_(paramsRouter_.getParameter<double>("FreqDTC")),
        tmpTFP_(paramsRouter_.getParameter<int>("TMP_TFP")),
        numFramesInfra_(paramsRouter_.getParameter<int>("NumFramesInfra")),
        numRoutingBlocks_(paramsRouter_.getParameter<int>("NumRoutingBlocks")),
        sizeStack_(paramsRouter_.getParameter<int>("SizeStack")),
        // Converter parameter
        widthRowLUT_(paramsConverter_.getParameter<int>("WidthRowLUT")),
        widthQoverPt_(paramsConverter_.getParameter<int>("WidthQoverPt")),
        // Tracker parameter
        numRegions_(paramsTracker_.getParameter<int>("NumRegions")),
        numOverlappingRegions_(paramsTracker_.getParameter<int>("NumOverlappingRegions")),
        numDTCsPerRegion_(paramsTracker_.getParameter<int>("NumDTCsPerRegion")),
        numModulesPerDTC_(paramsTracker_.getParameter<int>("NumModulesPerDTC")),
        tmpFE_(paramsTracker_.getParameter<int>("TMP_FE")),
        widthBend_(paramsTracker_.getParameter<int>("WidthBend")),
        widthCol_(paramsTracker_.getParameter<int>("WidthCol")),
        widthRow_(paramsTracker_.getParameter<int>("WidthRow")),
        baseBend_(paramsTracker_.getParameter<double>("BaseBend")),
        baseCol_(paramsTracker_.getParameter<double>("BaseCol")),
        baseRow_(paramsTracker_.getParameter<double>("BaseRow")),
        bendCut_(paramsTracker_.getParameter<double>("BendCut")),
        freqLHC_(paramsTracker_.getParameter<double>("FreqLHC")),
        // f/w constants
        speedOfLight_(paramsFW_.getParameter<double>("SpeedOfLight")),
        bField_(paramsFW_.getParameter<double>("BField")),
        bFieldError_(paramsFW_.getParameter<double>("BFieldError")),
        outerRadius_(paramsFW_.getParameter<double>("OuterRadius")),
        innerRadius_(paramsFW_.getParameter<double>("InnerRadius")),
        maxPitch_(paramsFW_.getParameter<double>("MaxPitch")),
        // Format specific parameter
        maxEta_(paramsFormat_.getParameter<double>("MaxEta")),
        minPt_(paramsFormat_.getParameter<double>("MinPt")),
        chosenRofPhi_(paramsFormat_.getParameter<double>("ChosenRofPhi")),
        numLayers_(paramsFormat_.getParameter<int>("NumLayers")),
        // Analyzer
        useMCTruth_(paramsAnalyzer_.getParameter<bool>("UseMCTruth")),
        inputTagTTClusterAssMap_(paramsAnalyzer_.getParameter<InputTag>("InputTagTTClusterAssMap")),
        producerLabel_(paramsAnalyzer_.getParameter<string>("ProducerLabel")),
        // TP
        tpMinPt_(paramsTP_.getParameter<double>("MinPt")),
        tpMaxEta_(paramsTP_.getParameter<double>("MaxEta")),
        tpMaxVertR_(paramsTP_.getParameter<double>("MaxVertR")),
        tpMaxVertZ_(paramsTP_.getParameter<double>("MaxVertZ")),
        tpMaxD0_(paramsTP_.getParameter<double>("MaxD0")),
        tpMinLayers_(paramsTP_.getParameter<int>("MinLayers")),
        tpMinLayersPS_(paramsTP_.getParameter<int>("MinLayersPS")) {
    // derived Router parameter
    numDTCs_ = numRegions_ * numDTCsPerRegion_;
    numDTCsPerTFP_ = numDTCsPerRegion_ * numOverlappingRegions_;
    numModules_ = numRegions_ * numDTCsPerRegion_ * numModulesPerDTC_;
    numModulesPerRoutingBlock_ = numModulesPerDTC_ / numRoutingBlocks_;

    const int numFramesPerBX = freqDTC_ / freqLHC_;
    maxFramesChannelInput_ = numFramesPerBX * tmpFE_ - numFramesInfra_;
    maxFramesChannelOutput_ = numFramesPerBX * tmpTFP_ - numFramesInfra_;

    numMergedRows_ = pow(2, widthRow_ - widthRowLUT_);
    maxCot_ = sinh(maxEta_);

    invPtToDphi_ = speedOfLight_ * bField_ / 2000.;
    rangeQoverPt_ = 2. * invPtToDphi_ / minPt_;

    widthLayer_ = ceil(log2(numLayers_));

    baseRegion_ = 2. * M_PI / numRegions_;

    if (dataFormat_ == "TMTT")
      tmtt_ = make_unique<SettingsTMTT>(iConfig, this);
    else if (dataFormat_ == "Hybrid")
      hybrid_ = make_unique<SettingsHybrid>(iConfig, this);
    else {
      cms::Exception exception("Configuration");
      exception << "unknown data format requested (" << dataFormat_ << ").";
      exception.addContext("trackerDTC::Settings::Settings");
      throw exception;
    }

    maxQoverPt_ = (rangeQoverPt_ - baseQoverPt_) / 2.;

    const int baseDiffM = widthRowLUT_ - widthRow_;

    baseM_ = basePhi_ * pow(2., baseDiffM);
    baseC_ = basePhi_;

    widthC_ = widthPhi_;

    const double x1 = pow(2, widthRow_) * baseRow_ * maxPitch_ / 2.;
    const double x0 = x1 - pow(2, widthRowLUT_) * baseRow_ * maxPitch_;
    const double maxM = atan2(x1, innerRadius_) - atan2(x0, innerRadius_);

    widthM_ = ceil(log2(maxM / baseM_));

    // event setup
    trackerGeometry_ = nullptr;
    trackerTopology_ = nullptr;
    magneticField_ = nullptr;
    // derived event setup
    dtcModules_ = vector<vector<Module*>>(numDTCs_);
    configurationSupported_ = true;
  }

  // store TrackerGeometry
  void Settings::setTrackerGeometry(const TrackerGeometry* trackerGeometry) { trackerGeometry_ = trackerGeometry; }
  // store TrackerTopology
  void Settings::setTrackerTopology(const TrackerTopology* trackerTopology) { trackerTopology_ = trackerTopology; }
  // store MagneticField
  void Settings::setMagneticField(const MagneticField* magneticField) { magneticField_ = magneticField; }
  // store TrackerDetToDTCELinkCablingMap
  void Settings::setCablingMap(const TrackerDetToDTCELinkCablingMap* cablingMap) { ttCablingMap_ = cablingMap; }
  // store TTStubAlgorithm handle
  void Settings::setTTStubAlgorithm(
      const edm::ESHandle<TTStubAlgorithm<Ref_Phase2TrackerDigi_>>& handleTTStubAlgorithm) {
    handleTTStubAlgorithm_ = handleTTStubAlgorithm;
  }
  // store GeometryConfiguration
  void Settings::setGeometryConfiguration(const ESHandle<DDCompactView>& handleGeometryConfiguration) {
    handleGeometryConfiguration_ = handleGeometryConfiguration;
  }
  // store ProcessHistory
  void Settings::setProcessHistory(const ProcessHistory& processHistory) { processHistory_ = processHistory; }

  // check current coniguration consistency with input configuration
  void Settings::checkConfiguration() {
    // check if bField is supported
    const double bField = magneticField_->inTesla(GlobalPoint(0., 0., 0.)).z();
    if (abs(bField - bField_) > bFieldError_) {
      configurationSupported_ = false;
      LogWarning("ConfigurationNotSupported") << "Magnetic Field from EventSetup (" << bField << ") differs more then "
                                              << bFieldError_ << " from supported value (" << bField_ << "). ";
    }
    // check if geometry is supported
    const ParameterSet& pSetGeometryConfiguration = getParameterSet(handleGeometryConfiguration_.description()->pid_);
    const vector<string>& geomXMLFiles =
        pSetGeometryConfiguration.getParameter<vector<string>>(supportedTrackerXMLPSet_);
    string trackerXMLVersion;
    for (const string& geomXMLFile : geomXMLFiles) {
      const auto begin = geomXMLFile.find(supportedTrackerXMLPath_) + supportedTrackerXMLPath_.size();
      const auto end = geomXMLFile.find(supportedTrackerXMLFile_);
      if (begin != string::npos && end != string::npos)
        trackerXMLVersion = geomXMLFile.substr(begin, end - begin - 1);
    }
    if (trackerXMLVersion.empty()) {
      cms::Exception exception("LogicError");
      exception << "No " << supportedTrackerXMLPath_ << "*/" << supportedTrackerXMLFile_
                << " found in GeometryConfiguration";
      exception.addContext("trackerDTC::Settings::checkConfiguration");
      throw exception;
    }
    if (find(supportedTrackerXMLVersions_.begin(), supportedTrackerXMLVersions_.end(), trackerXMLVersion) ==
        supportedTrackerXMLVersions_.end()) {
      configurationSupported_ = false;
      LogWarning("ConfigurationNotSupported")
          << "Geometry Configuration " << supportedTrackerXMLPath_ << trackerXMLVersion << "/"
          << supportedTrackerXMLFile_ << " is not supported. ";
    }
    if (!configurationSupported_)
      return;
    // check history
    if (!checkHistory_)
      return;
    // get iConfig of used GeometryConfiguration in input producer
    const ParameterSet* historyGeometryConfigurationPSet = nullptr;
    const pset::Registry* psetRegistry = pset::Registry::instance();
    for (const ProcessConfiguration& pc : processHistory_) {
      if (processName_ != pc.processName())
        continue;
      const ParameterSet* processPset = psetRegistry->getMapped(pc.parameterSetID());
      if (processPset && processPset->exists(productLabel_))
        historyGeometryConfigurationPSet = &processPset->getParameterSet(productLabel_);
    }
    if (!historyGeometryConfigurationPSet) {
      cms::Exception exception("Configuration");
      exception << "GeometryConfiguration not found in process history.";
      exception << "Searched for process " << processName_ << " and label " << productLabel_ << ".";
      exception.addContext("trackerDTC::Settings::checkConfiguration");
      throw exception;
    }
    if (handleGeometryConfiguration_.description()->pid_ != historyGeometryConfigurationPSet->id()) {
      cms::Exception exception("Configuration");
      exception
          << "Configured GeometryConfiguration inconsistent with used GeometryConfiguration during stub production.";
      exception.addContext("trackerDTC::Settings::checkConfiguration");
      throw exception;
    }
    // check data format specific history
    if (dataFormat_ == "Hybrid")
      hybrid_->checkConfiguration(this);
  }

  // convert ES Products into handy objects
  void Settings::beginRun() {
    // convert cabling map
    // DTC product used to convert between different dtc id schemes
    TTDTC ttDTC(numRegions_, numOverlappingRegions_, numDTCsPerRegion_);
    // module counter for each DTC
    vector<int> modIds(numDTCs_, 0);
    // loop over all tracker modules
    for (const DetId& detId : trackerGeometry_->detIds()) {
      // skip pixel detector
      if (detId.subdetId() == pixelBarrel || detId.subdetId() == pixelDisks)
        continue;
      // skip multiple detIds per module
      if (!trackerTopology_->isLower(detId))
        continue;
      // tk layout dtc id, lowerDetId - 1 = tk lyout det id
      const int tklId = ttCablingMap_->detIdToDTCELinkId(detId.rawId() + offsetDetIdTP()).first->second.dtc_id();
      // track trigger dtc id [0-215]
      const int dtcId = ttDTC.dtcId(tklId);
      // DTC module id [0-71]
      int& modId = modIds[dtcId];
      // track trigger module id [0-15551]
      const int ModId = dtcId * numModulesPerDTC_ + modId;
      // store connection between global module id and detId
      cablingMap_.emplace(detId, ModId);
      // check configuration
      if (modId++ > numModulesPerDTC_) {
        cms::Exception exception("overflow");
        exception << "Cabling map connects more than " << numModulesPerDTC_ << " modules to a DTC.";
        exception.addContext("trackerDTC::Settings::convertCablingMap");
        throw exception;
      }
    }
    // hybrid specific conversions
    if (dataFormat_ == "Hybrid") {
      hybrid_->createEncodingsBend(this);
      hybrid_->createEncodingsLayer(this);
    }
    //convert tracker geometry
    for (int dtcId = 0; dtcId < numDTCs_; dtcId++)
      dtcModules_[dtcId] = vector<Module*>(modIds.at(dtcId), nullptr);
    modules_.reserve(cablingMap_.size());
    for (const auto& map : cablingMap_) {
      const int dtcId = map.second / numModulesPerDTC_;
      const int modId = map.second % numModulesPerDTC_;
      modules_.emplace_back(this, map.first, dtcId);
      dtcModules_[dtcId][modId] = &modules_.back();
    }
  }

  // convert DetId to module id [0:15551]
  int Settings::modId(const DetId& detId) const {
    auto it = cablingMap_.find(detId);
    if (it == cablingMap_.end()) {
      cms::Exception exception("LogicError");
      exception << "Unknown DetID (" << detId << ") received from TTStub.";
      exception.addAdditionalInfo("Please check consistency between chosen cabling map and chosen tracker geometry.");
      exception.addContext("trackerDTC::Settings::modId");
      throw exception;
    }
    return it->second;
  }

  // collection of modules connected to a specific dtc
  const vector<Module*>& Settings::modules(int dtcId) const { return dtcModules_.at(dtcId); }

}  // namespace trackerDTC
