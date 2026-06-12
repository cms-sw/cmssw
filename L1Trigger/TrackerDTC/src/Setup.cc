#include "L1Trigger/TrackerDTC/interface/Setup.h"

#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <fstream>
#include <ostream>
#include <string>

namespace trackerDTC {

  Setup::Setup(const Config& config,
               const TrackerGeometry& trackerGeometry,
               const TrackerTopology& trackerTopology,
               const TrackerDetToDTCELinkCablingMap& cablingMap,
               const StubAlgorithmOfficial& stubAlgorithm,
               const ConfigTTStubAlgorithm& configTTStubAlgorithm)
      : trackerGeometry_(&trackerGeometry),
        trackerTopology_(&trackerTopology),
        nTiltedRings_(configTTStubAlgorithm.nTiltedRings),
        config_(config),
        printIDs_(config.printIDs) {
    // calc max window size
    feMaxWindowSize_ = -1;
    for (const auto& windowss : {configTTStubAlgorithm.tiltedBarrelCutSet,
                                 configTTStubAlgorithm.endcapCutSet,
                                 {configTTStubAlgorithm.barrelCut}})
      for (const auto& windows : windowss)
        for (const auto& window : windows)
          feMaxWindowSize_ = std::max(feMaxWindowSize_, static_cast<int>(window / config_.feBaseBend));
    // create bend encodings
    degradedBendPS_ = std::vector<std::vector<double>>(feMaxWindowSize_ + 1);
    degradedBend2S_ = std::vector<std::vector<double>>(feMaxWindowSize_ + 1);
    for (int window = 0; window < feMaxWindowSize_ + 1; window++) {
      std::vector<double>& degradedPS = degradedBendPS_[window];
      std::vector<double>& degraded2S = degradedBend2S_[window];
      degradedPS.reserve(window + 1);
      degraded2S.reserve(window + 1);
      for (int bend = 0; bend < window + 1; bend++) {
        degradedPS.push_back(stubAlgorithm.degradeBend(true, window, bend));
        degraded2S.push_back(stubAlgorithm.degradeBend(false, window, bend));
      }
    }
    encodingsBendPS_ = degradedBendPS_;
    for (std::vector<double>& encoding : encodingsBendPS_)
      encoding.erase(std::unique(encoding.begin(), encoding.end()), encoding.end());
    encodingsBend2S_ = degradedBend2S_;
    for (std::vector<double>& encoding : encodingsBend2S_)
      encoding.erase(std::unique(encoding.begin(), encoding.end()), encoding.end());
    // calulcate constants
    const int numFramesPerBX = config_.dtcFreq / config_.sysLhcFreq;
    sysNumFrames_ = numFramesPerBX * config_.regNumTFP - config_.sysNumFramesInfra;
    dtcNumTFP_ = config_.regNumTFP * config_.sysNumOverlap;
    sysNumDTC_ = config_.sysNumRegion * config_.regNumDTC;
    feWidthBend_ = std::max(config_.mpaWidthBend, config_.cbcWidthBend);
    feWidthBX_ = tt::ilog2(config_.cicNumBX);
    feWidthCIC_ = tt::ilog2(config_.smNumCIC);
    feWidthCol_ = tt::ilog2(std::max(config_.mpaNumCol, config_.cbcNumCol) / config_.feBaseCol);
    feWidthFEC_ = tt::ilog2(config_.cicNumFEC);
    feWidthRow_ = tt::ilog2(std::max(config_.mpaNumRow, config_.cbcNumRow) / config_.feBaseRow);
    fePosValid_ = feWidthBX_ + feWidthFEC_ + feWidthRow_ + feWidthBend_ + feWidthCol_;
    feNumFrames_ = numFramesPerBX * config_.cicNumBX - config_.sysNumFramesInfra;
    tmp8NumNodes_ = config_.unNumNode;
    tmp8NumInputs_ = config_.dtcNumModule / config_.unNumNode;
    tmp8NumOutputs_ = config_.cicNumBX;
    tmp8NumChannel_ = tmp8NumNodes_ * tmp8NumOutputs_;
    tmp8NumFrames_ = numFramesPerBX * tmp8NumOutputs_ - 1;
    tmp12NumNodes_ = tmp8NumNodes_ / config_.reIn;
    tmp12NumInputs_ = config_.reIn * tmp8NumOutputs_;
    tmp12NumOutputs_ = tmp8NumOutputs_;
    tmp12NumChannel_ = tmp12NumNodes_ * tmp12NumOutputs_;
    tmp12NumFrames_ = numFramesPerBX * tmp8NumOutputs_ * config_.reOut / config_.reIn - 1;
    tmp18NumNodes_ = tmp12NumNodes_ / config_.reIn;
    tmp18NumInputs_ = config_.reIn * tmp12NumOutputs_;
    tmp18NumOutputs_ = tmp12NumOutputs_;
    tmp18NumChannel_ = tmp18NumNodes_ * tmp18NumOutputs_;
    tmp18NumFrames_ = numFramesPerBX * tmp12NumOutputs_ * config_.reOut / config_.reIn - 1;
    fwWidthDSPab_ = config_.fwWidthDSPa - 1;
    fwWidthDSPau_ = config_.fwWidthDSPa - 2;
    fwWidthDSPbb_ = config_.fwWidthDSPb - 1;
    fwWidthDSPbu_ = config_.fwWidthDSPb - 2;
    sysInvPtToDphi_ = config_.sysSpeedOfLight * config_.sysBField / 2000.;
    regMaxInv2R_ = sysInvPtToDphi_ / config_.regMinPt;
    regMaxZT_ = std::sinh(config_.regMaxEta) * config_.regChosenRofZ;
    const double rangeInv2R = 2. * regMaxInv2R_;
    const double rangeZT = 2. * regMaxZT_;
    regRangePhiT_ = 2. * M_PI / config_.sysNumRegion;
    glWidthInv2R_ = config_.fwWidthDSPb - 1;
    glBaseInv2R_ = rangeInv2R * std::pow(2, -glWidthInv2R_);
    const double maxRphi = std::max(std::abs(config_.sysOuterRadius - config_.regChosenRofPhi),
                                    std::abs(config_.sysInnerRadius - config_.regChosenRofPhi));
    const double rangeR = 2. * maxRphi;
    const double rangePhi = regRangePhiT_ + rangeInv2R * maxRphi;
    const double rangeZ = 2. * config_.sysHalfLength;
    const double significantR = regRangePhiT_ / rangeInv2R;
    const double significantPhi = regRangePhiT_;
    const double significantZ = rangeZT;
    glBaseR_ = significantR * std::pow(2, tt::ilog2(rangeR / significantR) - config_.glWidthR);
    glBasePhi_ = significantPhi * std::pow(2, tt::ilog2(rangePhi / significantPhi) - config_.glWidthPhi);
    glBaseZ_ = significantZ * std::pow(2, tt::ilog2(rangeZ / significantZ) - config_.glWidthZ);
    const int widthRM = config_.fwWidthDSPa - 1;
    const double rangeRM =
        std::sqrt(std::pow(config_.sysInnerRadius, 2) + std::pow(config_.mpaPitch * config_.mpaNumRow, 2)) -
        config_.sysInnerRadius;
    glBaseRC_ = .5 * glBaseR_;
    glBaseRM_ = glBaseR_ * std::pow(2, tt::ilog2(rangeRM / glBaseR_) - widthRM);
    const int widthPhiM = config_.fwWidthDSPa - 1;
    const double rangePhiM = config_.mpaPitch * config_.mpaNumRow / config_.sysInnerRadius;
    glBasePhiC_ = .5 * glBasePhi_;
    glBasePhiM_ = glBasePhi_ * std::pow(2, tt::ilog2(rangePhiM / glBasePhi_) - widthPhiM);
    const double stubRangeInv2R = 2. * sysInvPtToDphi_ / config_.stubMinPt;
    stubRangePhi_ = regRangePhiT_ + stubRangeInv2R * maxRphi;
    stubWidthLayerId_ = tt::ilog2(config_.dtcNumLayer);
    stubBasesZ_.reserve(SensorModule::NumTypes);
    for (int type = 0; type < SensorModule::NumTypes; type++)
      stubBasesZ_.emplace_back(config_.stubRangesZ.at(type) / std::pow(2., config_.stubWidthsZ.at(type)));
    stubBasesR_.reserve(SensorModule::NumTypes);
    for (int type = 0; type < SensorModule::NumTypes; type++)
      stubBasesR_.emplace_back(config_.stubRangesR.at(type) / std::pow(2., config_.stubWidthsR.at(type)));
    stubBasesR_[SensorModule::Disk2S] = 1.;
    stubBasesPhi_.reserve(SensorModule::NumTypes);
    for (int type = 0; type < SensorModule::NumTypes; type++)
      stubBasesPhi_.emplace_back(stubRangePhi_ / std::pow(2., config_.stubWidthsPhi.at(type)));
    stubBasesAlpha_.reserve(SensorModule::NumTypes);
    for (int type = 0; type < SensorModule::NumTypes; type++)
      stubBasesAlpha_.emplace_back(config_.stubRangesAlpha.at(type) / std::pow(2., config_.stubWidthsAlpha.at(type)));
    stubNumsUnusedBits_.reserve(SensorModule::NumTypes);
    for (int type = 0; type < SensorModule::NumTypes; type++)
      stubNumsUnusedBits_.emplace_back(TTBV::S_ - config_.stubWidthsND.at(type) - config_.stubWidthsR.at(type) -
                                       config_.stubWidthsZ.at(type) - config_.stubWidthsPhi.at(type) -
                                       config_.stubWidthsAlpha.at(type) - config_.stubWidthsBend.at(type) -
                                       stubWidthLayerId_ - 1);
    const double significantStubR = *std::min_element(stubBasesR_.begin(), stubBasesR_.end());
    const double significantStubPhi = *std::min_element(stubBasesPhi_.begin(), stubBasesPhi_.end());
    const double significantStubZ = *std::min_element(stubBasesZ_.begin(), stubBasesZ_.end());
    stubBaseR_ = significantStubR * std::pow(2, tt::ilog2(glBaseR_ / significantStubR));
    stubBasePhi_ = significantStubPhi * std::pow(2, tt::ilog2(glBasePhi_ / significantStubPhi));
    stubBaseZ_ = significantStubZ * std::pow(2, tt::ilog2(glBaseZ_ / significantStubZ));
    unDepth_ = std::pow(2, config_.unWidthAddr);
    // encode layer ids
    enum SubDetId { pixelBarrel = 1, pixelDisks = 2 };
    std::vector<std::set<int>> layerIds(config_.regNumDTC);
    // loop over all tracker modules
    for (const DetId& detId : trackerGeometry_->detIds()) {
      // skip pixel detector
      if (detId.subdetId() == pixelBarrel || detId.subdetId() == pixelDisks)
        continue;
      // skip multiple detIds per module
      if (!trackerTopology_->isLower(detId))
        continue;
      // lowerDetId - 1 = tk layout det id
      const DetId detIdTkLayout = detId - 1;
      // tk layout dtc id, lowerDetId - 1 = tk lyout det id
      const int tklId = cablingMap.detIdToDTCELinkId(detIdTkLayout).first->second.dtc_id();
      // track trigger dtc id [0-215]
      const int dtcId = this->dtcId(tklId);
      // barrel or endcap
      const bool barrel = detId.subdetId() == StripSubdetector::TOB;
      // layer id [barrel: 1-6, endcap: 11-15]
      const int layerId = (barrel ? trackerTopology_->layer(detId) : trackerTopology_->tidWheel(detId) + 10);
      // store layer id
      layerIds[dtcId % config_.regNumDTC].insert(layerId);
    }
    // create sensor modules
    sensorModules_.reserve(config_.sysNumModule);
    dtcModules_ = std::vector<std::vector<const SensorModule*>>(
        sysNumDTC_, std::vector<const SensorModule*>(config_.dtcNumModule, nullptr));
    // loop over all tracker modules
    for (const DetId& detId : trackerGeometry_->detIds()) {
      // skip pixel detector
      if (detId.subdetId() == pixelBarrel || detId.subdetId() == pixelDisks)
        continue;
      // skip multiple detIds per module
      if (!trackerTopology_->isLower(detId))
        continue;
      // lowerDetId - 1 = tk layout det id
      const DetId detIdTkLayout = detId - 1;
      // tk layout dtc id, lowerDetId - 1 = tk lyout det id
      const int tklId = cablingMap.detIdToDTCELinkId(detIdTkLayout).first->second.dtc_id();
      // track trigger dtc id [0-215]
      const int dtcId = this->dtcId(tklId);
      // dtc channel id [0-71]
      const int modId = cablingMap.detIdToDTCELinkId(detIdTkLayout).first->second.gbtlink_id();
      // get layer encoding for this module
      const std::set<int>& encodingLayer = layerIds[dtcId % config_.regNumDTC];
      // getting bend window size
      const bool barrel = detId.subdetId() == StripSubdetector::TOB;
      const int index = barrel ? trackerTopology.layer(detId) : trackerTopology.tidWheel(detId);
      double ws;
      if (barrel) {
        const SensorModule::TypeTilt typeTilt = static_cast<SensorModule::TypeTilt>(trackerTopology.tobSide(detId));
        if (typeTilt == SensorModule::TypeTilt::flat)
          ws = configTTStubAlgorithm.barrelCut[index];
        else {
          int ladder = trackerTopology.tobRod(detId);
          if (typeTilt == SensorModule::TypeTilt::tiltedMinus)
            // Corrected ring number, bet 0 and barrelNTilt.at(layerIndex_), in ascending |z|
            ladder = 1 + configTTStubAlgorithm.nTiltedRings[index] - ladder;
          ws = configTTStubAlgorithm.tiltedBarrelCutSet[index][ladder];
        }
      } else {
        const int ring = trackerTopology.tidRing(detId);
        ws = configTTStubAlgorithm.endcapCutSet[index][ring];
      }
      ws = tt::floor(ws / config_.feBaseRow);
      // getting bend endcoding and degradation
      const bool psModule = trackerGeometry.getDetectorType(detId) == TrackerGeometry::ModuleType::Ph2PSP;
      const std::vector<double>& encodingBend = psModule ? encodingsBendPS_[ws] : encodingsBend2S_[ws];
      const std::vector<double>& degradedBend = psModule ? degradedBendPS_[ws] : degradedBend2S_[ws];
      // construct sendor module
      sensorModules_.emplace_back(
          this, detId, dtcId, modId, barrel, psModule, index, ws, encodingBend, degradedBend, encodingLayer);
      // store connection between dtc and sensor module
      dtcModules_[dtcId][modId] = &sensorModules_.back();
      // store connection between detId and sensor module
      detIdToSensorModule_.emplace(detId, &sensorModules_.back());
    }
    if (config_.printConstants) {
      if (printIDs_.empty()) {
        printIDs_ = std::vector<int>(sysNumDTC_);
        std::iota(printIDs_.begin(), printIDs_.end(), 0);
      }
      for (int dtcId : printIDs_) {
        const std::string path = config_.printPath + "DTC" + std::to_string(dtcId) + ".dat";
        std::stringstream ss;
        for (const SensorModule* sm : dtcModules_[dtcId]) {
          if (!sm) {
            ss << std::string(1 + 5 + feWidthBend_ + stubWidthLayerId_ + 7 * TTBV::S_, '0') << std::endl;
            ;
            continue;
          }
          ss << "1";
          ss << std::to_string(sm->barrel());
          ss << std::to_string(sm->psModule());
          ss << std::to_string(sm->signRow());
          ss << std::to_string(sm->signCol());
          ss << std::to_string(sm->signBend());
          ss << TTBV(sm->windowSize(), feWidthBend_).str();
          ss << TTBV(sm->encodedLayer(), stubWidthLayerId_).str();
          ss << TTBV(sm->sep()).str();
          ss << TTBV(sm->tilt()).str();
          ss << TTBV(sm->r()).str();
          ss << TTBV(sm->phi()).str();
          ss << TTBV(sm->z()).str();
          ss << TTBV(sm->offsetR()).str();
          ss << TTBV(sm->offsetZ()).str();
          ss << std::endl;
        }
        std::fstream fs;
        fs.open(path, std::fstream::out);
        fs << ss.rdbuf();
        fs.close();
      }
    }
    if (config_.printEncodingBend) {
      std::string pathPS = config_.printPath + "EncodingBendPS.dat";
      std::stringstream ssPS;
      for (int windowSize = std::pow(2, config_.mpaWidthBend - 1); windowSize <= feMaxWindowSize_; windowSize++)
        for (double d : encodingsBendPS_[windowSize])
          ssPS << TTBV(d).str() << std::endl;
      std::fstream fs;
      fs.open(pathPS, std::fstream::out);
      fs << ssPS.rdbuf();
      fs.close();
      const std::string path2S = config_.printPath + "EncodingBend2S.dat";
      std::stringstream ss2S;
      for (int windowSize = std::pow(2, config_.cbcWidthBend - 1); windowSize <= feMaxWindowSize_; windowSize++)
        for (double d : encodingsBend2S_[windowSize])
          ss2S << TTBV(d).str() << std::endl;
      fs.open(path2S, std::fstream::out);
      fs << ss2S.rdbuf();
      fs.close();
    }
  }

  // converts tk layout id into dtc id
  int Setup::dtcId(int tkLayoutId) const {
    const int tkId = tkLayoutId - 1;
    const int side = tkId / (config_.sysNumRegion * config_.sysNumATCASlot);
    const int region = (tkId % (config_.sysNumRegion * config_.sysNumATCASlot)) / config_.sysNumATCASlot;
    const int slot = tkId % config_.sysNumATCASlot;
    return region * config_.regNumDTC + side * config_.sysNumATCASlot + slot;
  }

  // returns global TTStub position
  GlobalPoint Setup::stubPosTT(const TTStubRef& ttStubRef) const {
    const DetId detId = ttStubRef->getDetId() + 1;
    const GeomDetUnit* det = trackerGeometry_->idToDetUnit(detId);
    const PixelTopology* topol =
        dynamic_cast<const PixelTopology*>(&(dynamic_cast<const PixelGeomDetUnit*>(det)->specificTopology()));
    const Plane& plane = dynamic_cast<const PixelGeomDetUnit*>(det)->surface();
    const std::vector<int>& rows = ttStubRef->clusterRef(0)->getRows();
    const std::vector<int>& cols = ttStubRef->clusterRef(0)->getCols();
    const double row = (rows.front() + rows.back()) / 2. + .5;
    const double col = *std::min_element(cols.begin(), cols.end()) + .5;
    MeasurementPoint fe(row, col);
    return plane.toGlobal(topol->localPosition(fe));
  }

  // returns bit accurate position of a stub from a given tfp region [0-8]
  GlobalPoint Setup::stubPosDTC(const tt::FrameStub& frame, int region) const {
    if (frame.first.isNull())
      return GlobalPoint();
    const SensorModule* sm = sensorModule(frame.first->getDetId() + 1);
    const SensorModule::Type type = sm->type();
    // parse bit vector
    TTBV bv(frame.second);
    bv >>= 1 + stubWidthLayerId_ + stubWidthBend(type);
    TTBV alpha;
    if (type == SensorModule::Disk2S)
      alpha = TTBV(bv, config_.stubWidthsAlpha[type], 0, true);
    bv >>= config_.stubWidthsAlpha[type];
    // stub phi w.r.t. to nonant edge in rad
    double phi = bv.val(stubWidthPhi(type)) * stubBasePhi(type);
    bv >>= stubWidthPhi(type);
    // stub z w.r.t. an offset in cm
    double z = bv.val(stubWidthZ(type), 0, true) * stubBaseZ(type);
    bv >>= stubWidthZ(type);
    // stub r w.r.t. an offset in cm
    double r = bv.val(stubWidthR(type), 0, sm->barrel()) * stubBaseR(type);
    // stub r in cm
    if (type == SensorModule::Disk2S) {
      const double row = alpha.val(stubBasesAlpha_[type]);
      r = stubDiskR(sm->layerIndex(), r);
      r = std::sqrt(std::pow(r, 2) + std::pow(row * config_.cbcPitch, 2));
    } else
      r += sm->offsetR() + config_.regChosenRofPhi;
    // stub z in cm
    z += sm->offsetZ();
    // stub phi in rad
    phi = tt::deltaPhi(phi - stubRangePhi_ / 2. + region * regRangePhiT_);
    return GlobalPoint(GlobalPoint::Cylindrical(r, phi, z));
  }

}  // namespace trackerDTC
