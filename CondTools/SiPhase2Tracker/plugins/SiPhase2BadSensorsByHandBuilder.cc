// -*- C++ -*-
//
// Package:    CondTools/SiPhase2BadSensorsByHandBuilder
// Class:      SiPhase2BadSensorsByHandBuilder
//
/**\class SiPhase2BadSensorsByHandBuilder SiPhase2BadSensorsByHandBuilder.cc CondTools/SiPhase2Tracker/plugins/SiPhase2BadSensorsByHandBuilder.cc

 Description: Translates a hand-specified text list of Phase-2 sensor DetIds into a SiPixelQuality payload and
 writes it to sqlite. One job instance per subsystem.
 To disable a whole module, list both of its sensor DetIds.

 Implementation: Each listed DetId must exist in the geometry, must be a sensor (not a bare OT stack DetId), and
 must belong to the subsystem selected by Record.
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

class SiPhase2BadSensorsByHandBuilder : public ConditionDBWriter<SiPixelQuality> {
public:
  explicit SiPhase2BadSensorsByHandBuilder(const edm::ParameterSet&);
  ~SiPhase2BadSensorsByHandBuilder() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void algoBeginRun(const edm::Run& run, const edm::EventSetup& es) override;
  uint32_t validateDetId(uint32_t id) const;
  std::unique_ptr<SiPixelQuality> getNewObject() override;

  const bool printdebug_;
  const TrackerGeometry* tkGeom_;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
  const std::string badSensorListFile_;
  const std::string targetRecord_;
  std::map<uint32_t, const StackGeomDet*> sensorToStack_;
};
//
// constructor
//
SiPhase2BadSensorsByHandBuilder::SiPhase2BadSensorsByHandBuilder(const edm::ParameterSet& iConfig)
    : ConditionDBWriter<SiPixelQuality>(iConfig),
      printdebug_(iConfig.getUntrackedParameter<bool>("printDebug")),
      tkGeom_(nullptr),
      geomToken_(esConsumes<edm::Transition::BeginRun>()),
      badSensorListFile_(iConfig.getParameter<std::string>("badSensorListFile")),
      targetRecord_(iConfig.getParameter<std::string>("Record")) {
  if (targetRecord_ != "SiPhase2OuterTrackerBadSensorRcd" && targetRecord_ != "SiPhase2InnerTrackerBadSensorRcd") {
    throw cms::Exception("SiPhase2BadSensorsByHandBuilder")
        << "Record must be SiPhase2OuterTrackerBadSensorRcd or SiPhase2InnerTrackerBadSensorRcd, got '" << targetRecord_
        << "'";
  }
  if (badSensorListFile_.empty()) {
    throw cms::Exception("SiPhase2BadSensorsByHandBuilder") << "badSensorListFile must be set.";
  }
}
//
// member functions
//
SiPhase2BadSensorsByHandBuilder::~SiPhase2BadSensorsByHandBuilder() = default;

void SiPhase2BadSensorsByHandBuilder::algoBeginRun(const edm::Run& run, const edm::EventSetup& es) {
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

uint32_t SiPhase2BadSensorsByHandBuilder::validateDetId(uint32_t id) const {
  const GeomDet* det = tkGeom_->idToDet(DetId(id));  // throws if id doesn't exist in the geometry

  if (dynamic_cast<const StackGeomDet*>(det)) {
    throw cms::Exception("SiPhase2BadSensorsByHandBuilder")
        << "DetId " << id << " is an OT stack, not a sensor. List both of its sensor DetIds instead.";
  }

  // a sensor of an OT stack is in sensorToStack_; an IT sensor is not
  bool isOTModule = sensorToStack_.find(id) != sensorToStack_.end();
  bool isOTRecord = (targetRecord_ == "SiPhase2OuterTrackerBadSensorRcd");
  if (isOTModule != isOTRecord) {
    throw cms::Exception("SiPhase2BadSensorsByHandBuilder") << "DetId " << id << " belongs to the other subsystem";
  }

  return id;
}

std::unique_ptr<SiPixelQuality> SiPhase2BadSensorsByHandBuilder::getNewObject() {
  auto obj = std::make_unique<SiPixelQuality>();
  std::set<uint32_t> badDetIds;
  uint32_t id;

  edm::FileInPath sensorListPath(badSensorListFile_);
  std::ifstream sensorFile;
  sensorFile.open(sensorListPath.fullPath().c_str());
  if (!sensorFile.is_open()) {
    throw cms::Exception("SiPhase2BadSensorsByHandBuilder")
        << "Could not open badSensorListFile: " << badSensorListFile_;
  }
  while (sensorFile >> id) {
    badDetIds.insert(validateDetId(id));
  }

  edm::LogInfo("SiPhase2BadSensorsByHandBuilder")
      << "Writing " << badDetIds.size() << " disabled sensors to " << targetRecord_;

  for (uint32_t detId : badDetIds) {
    SiPixelQuality::disabledModuleType badModule;
    badModule.DetID = detId;
    badModule.errorType = 0;
    badModule.BadRocs = 65535;  // Ph1 relic, currently not used.
    obj->addDisabledModule(badModule);
    if (printdebug_) {
      edm::LogInfo("SiPhase2BadSensorsByHandBuilder") << "Added disabled DetId " << detId;
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
    edm::LogError("SiPhase2BadSensorsByHandBuilder") << "PoolDBOutputService not available";
  }

  return obj;
}

void SiPhase2BadSensorsByHandBuilder::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setComment("Builds a SiPixelQuality payload from a hand-specified list of Phase-2 DetIds.");
  ConditionDBWriter::fillPSetDescription(desc);
  desc.addUntracked<bool>("printDebug", false);
  desc.add<std::string>("badSensorListFile")
      ->setComment(
          "Path to a plain text file, one sensor DetId per line, for the subsystem selected by Record; to "
          "disable a whole module, list both of its sensor DetIds");
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiPhase2BadSensorsByHandBuilder);
