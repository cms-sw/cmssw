// -*- C++ -*-
//
// Package:    CondTools/SiPhase2BadComponentsByHandBuilder
// Class:      SiPhase2BadComponentsByHandBuilder
//
/**\class SiPhase2BadComponentsByHandBuilder SiPhase2BadComponentsByHandBuilder.cc CondTools/SiPhase2Tracker/plugins/SiPhase2BadComponentsByHandBuilder.cc

 Description: Translates hand-specified text lists of Phase-2 DetIds into a SiPixelQuality payload and writes it to
 sqlite. badModuleListFile entries are expanded to their full physical module;
 badSensorListFile entries are written as they are.

 Implementation: Each badModuleListFile DetId is expanded to its full physical module: OT stacks to both sensors via
 StackGeomDet, IT Ph2PXB3D pairs to the partner DetId by parity. Each badSensorListFile DetId is written as is;
 a stack DetId among them is rejected. Entries that belong to the other subsystem are skipped based on
 targetRecord.
*/
//
// Original Author:  Stef Duponcheel
//         Created:  Thu, 08 Oct 2026 14:41:19 GMT
//
//

// system include files
#include <vector>
#include <memory>
#include <map>
#include <set>
#include <fstream>

// user include files
#include "CommonTools/ConditionDBWriter/interface/ConditionDBWriter.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"

#include "DataFormats/DetId/interface/DetId.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/CommonTopologies/interface/StackGeomDet.h"

//
// class declaration
//

class SiPhase2BadComponentsByHandBuilder : public ConditionDBWriter<SiPixelQuality> {
public:
  explicit SiPhase2BadComponentsByHandBuilder(const edm::ParameterSet&);
  ~SiPhase2BadComponentsByHandBuilder() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void algoBeginRun(const edm::Run& run, const edm::EventSetup& es) override;
  std::vector<uint32_t> selectDetIds(uint32_t id, bool expandToModule) const;
  std::unique_ptr<SiPixelQuality> getNewObject() override;

  const bool printdebug_;
  const TrackerGeometry* tkGeom_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const std::string badModuleListFile_;
  const std::string badSensorListFile_;
  const std::string targetRecord_;
  std::map<uint32_t, const StackGeomDet*> sensorToStack_;
};
//
// constructor
//
SiPhase2BadComponentsByHandBuilder::SiPhase2BadComponentsByHandBuilder(const edm::ParameterSet& iConfig)
    : ConditionDBWriter<SiPixelQuality>(iConfig),
      printdebug_(iConfig.getUntrackedParameter<bool>("printDebug", true)),
      tkGeom_(nullptr),
      geomToken_(esConsumes<edm::Transition::BeginRun>()),
      badModuleListFile_(iConfig.getUntrackedParameter<std::string>("badModuleListFile", "")),
      badSensorListFile_(iConfig.getUntrackedParameter<std::string>("badSensorListFile", "")),
      targetRecord_(iConfig.getUntrackedParameter<std::string>("targetRecord", "")) {
  if (targetRecord_ != "SiPhase2OuterTrackerBadModuleRcd" && targetRecord_ != "SiPhase2InnerTrackerBadModuleRcd") {
    throw cms::Exception("SiPhase2BadComponentsByHandBuilder")
        << "targetRecord must be SiPhase2OuterTrackerBadModuleRcd or SiPhase2InnerTrackerBadModuleRcd, got '"
        << targetRecord_ << "'";
  }
  if (badModuleListFile_.empty() && badSensorListFile_.empty()) {
    throw cms::Exception("SiPhase2BadComponentsByHandBuilder")
        << "At least one of badModuleListFile or badSensorListFile must be set.";
  }
}
//
// member functions
//
SiPhase2BadComponentsByHandBuilder::~SiPhase2BadComponentsByHandBuilder() = default;

void SiPhase2BadComponentsByHandBuilder::algoBeginRun(const edm::Run& run, const edm::EventSetup& es) {
  if (!tkGeom_) {
    tkGeom_ = &es.getData(geomToken_);
    // map each OT sensor to its stack
    for (auto const* det : tkGeom_->dets()) {
      if (const auto* stack = dynamic_cast<const StackGeomDet*>(det)) {
        sensorToStack_[stack->lowerDet()->geographicalId().rawId()] = stack;
        sensorToStack_[stack->upperDet()->geographicalId().rawId()] = stack;
      }
    }
  }
}

std::vector<uint32_t> SiPhase2BadComponentsByHandBuilder::selectDetIds(uint32_t id, bool expandToModule) const {
  const GeomDet* det = tkGeom_->idToDet(DetId(id));
  if (!det) {
    throw cms::Exception("SiPhase2BadComponentsByHandBuilder") << "DetId " << id << " does not exist in the geometry";
  }

  const StackGeomDet* directStack = dynamic_cast<const StackGeomDet*>(det);
  if (directStack && !expandToModule) {
    throw cms::Exception("SiPhase2BadComponentsByHandBuilder")
        << "DetId " << id << " is an OT stack, not a sensor. List it in badModuleListFile to disable a whole module.";
  }

  // the stack itself, or (if id is a sensor) the stack it belongs to
  const StackGeomDet* stack = directStack;
  if (!stack) {
    // a sensor of an OT stack: idToDet returns the sensor, not the stack
    if (auto it = sensorToStack_.find(id); it != sensorToStack_.end()) {
      stack = it->second;
    }
  }
  bool isOTModule = stack != nullptr;
  bool isOTRecord = (targetRecord_ == "SiPhase2OuterTrackerBadModuleRcd");
  if (isOTModule != isOTRecord) {
    if (printdebug_) {
      edm::LogInfo("SiPhase2BadComponentsByHandBuilder")
          << "DetId " << id << " belongs to the other subsystem, skipped for " << targetRecord_;
    }
    return {};
  }

  if (!expandToModule) {
    return {id};  // write the listed sensor
  }

  if (isOTModule) {
    // OT stack: both sensors make up one physical module
    return {stack->lowerDet()->geographicalId().rawId(), stack->upperDet()->geographicalId().rawId()};
  }

  if (tkGeom_->getDetectorType(DetId(id)) == TrackerGeometry::ModuleType::Ph2PXB3D) {
    // IT 3D sensor pair: partner is id+1 if id is odd, id-1 if even
    uint32_t partnerId = (id % 2 == 1) ? (id + 1) : (id - 1);
    if (!tkGeom_->idToDet(DetId(partnerId))) {
      throw cms::Exception("SiPhase2BadComponentsByHandBuilder")
          << "DetId " << id << " claims to be Ph2PXB3D but its expected partner " << partnerId
          << " doesn't exist in the geometry.";
    }
    return {id, partnerId};
  }

  return {id};  // Ph2PXB or Ph2PXF: single-sensor module, nothing to expand
}

std::unique_ptr<SiPixelQuality> SiPhase2BadComponentsByHandBuilder::getNewObject() {
  auto obj = std::make_unique<SiPixelQuality>();
  std::set<uint32_t> badDetIds;
  uint32_t id;

  if (!badModuleListFile_.empty()) {
    std::ifstream infile(badModuleListFile_);
    if (!infile.is_open()) {
      throw cms::Exception("SiPhase2BadComponentsByHandBuilder")
          << "Could not open badModuleListFile: " << badModuleListFile_;
    }
    while (infile >> id) {
      for (uint32_t detId : selectDetIds(id, true)) {
        badDetIds.insert(detId);
      }
    }
  }

  if (!badSensorListFile_.empty()) {
    std::ifstream sensorFile(badSensorListFile_);
    if (!sensorFile.is_open()) {
      throw cms::Exception("SiPhase2BadComponentsByHandBuilder")
          << "Could not open badSensorListFile: " << badSensorListFile_;
    }
    while (sensorFile >> id) {
      for (uint32_t detId : selectDetIds(id, false)) {
        badDetIds.insert(detId);
      }
    }
  }

  for (uint32_t detId : badDetIds) {
    SiPixelQuality::disabledModuleType badModule;
    badModule.DetID = detId;
    badModule.errorType = 0;
    badModule.BadRocs = 65535;  // Ph1 relic, currently not used.
    obj->addDisabledModule(badModule);
    if (printdebug_) {
      edm::LogInfo("SiPhase2BadComponentsByHandBuilder") << "Added disabled DetId " << detId;
    }
  }

  edm::Service<cond::service::PoolDBOutputService> mydbservice;
  if (mydbservice.isAvailable()) {
    if (mydbservice->isNewTagRequest(targetRecord_)) {
      mydbservice->createOneIOV<SiPixelQuality>(*obj, mydbservice->beginOfTime(), targetRecord_);
    } else {
      mydbservice->appendOneIOV<SiPixelQuality>(*obj, mydbservice->currentTime(), targetRecord_);
    }
  } else {
    edm::LogError("SiPhase2BadComponentsByHandBuilder") << "PoolDBOutputService not available";
  }

  return obj;
}

void SiPhase2BadComponentsByHandBuilder::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Builds a SiPixelQuality payload from a hand-specified list of Phase-2 DetIds.");
  ConditionDBWriter::fillPSetDescription(desc);
  desc.addUntracked<bool>("printDebug", true);
  desc.addUntracked<std::string>("badModuleListFile", "")
      ->setComment("Path to a plain text file, one DetId per line, each expanded to its full module; optional");
  desc.addUntracked<std::string>("badSensorListFile", "")
      ->setComment("Path to a plain text file, one DetId per line, each written as it is; optional");
  desc.addUntracked<std::string>("targetRecord")
      ->setComment(
          "Record to write this payload under, e.g. SiPhase2OuterTrackerBadModuleRcd or "
          "SiPhase2InnerTrackerBadModuleRcd");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiPhase2BadComponentsByHandBuilder);
