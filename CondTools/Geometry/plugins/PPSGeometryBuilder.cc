/****************************************************************************
 *
 *  DB builder for PPS geometry
 *
 *  Author: Wagner Carvalho (wcarvalh@cern.ch)
 *  Moved out common functionalities to DetGeomDesc(Builder) + support both old DD and DD4hep, by Gabrielle Hugo.
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESWatcher.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/VeryForwardIdealGeometryRecord.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

#include "CondFormats/GeometryObjects/interface/PDetGeomDesc.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDescBuilder.h"

class PPSGeometryBuilder : public edm::one::EDAnalyzer<> {
public:
  explicit PPSGeometryBuilder(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  void buildSerializableDataFromGeoInfo(PDetGeomDesc* serializableData, const DetGeomDesc* geoInfo, int& counter);
  PDetGeomDesc::Item buildItemFromDetGeomDesc(const DetGeomDesc* geoInfo);

  bool fromDD4hep_;
  std::string compactViewTag_;
  bool isRun2_;
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> ddToken_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4hepToken_;
  edm::ESWatcher<IdealGeometryRecord> watcherIdealGeometry_;
  edm::Service<cond::service::PoolDBOutputService> dbService_;
};

PPSGeometryBuilder::PPSGeometryBuilder(const edm::ParameterSet& iConfig)
    : fromDD4hep_(iConfig.getUntrackedParameter<bool>("fromDD4hep", false)),
      compactViewTag_(iConfig.getUntrackedParameter<std::string>("compactViewTag", "XMLIdealGeometryESSource_CTPPS")),
      isRun2_(iConfig.getUntrackedParameter<bool>("isRun2", false)),
      ddToken_(esConsumes(edm::ESInputTag("", compactViewTag_))),
      dd4hepToken_(esConsumes(edm::ESInputTag("", compactViewTag_))) {}

/*
 * Save PPS geo to DB.
 */
void PPSGeometryBuilder::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get DetGeomDesc tree
  std::unique_ptr<DetGeomDesc> geoInfoRoot = nullptr;
  if (watcherIdealGeometry_.check(iSetup)) {
    edm::LogInfo("PPSGeometryBuilder") << "Got IdealGeometryRecord ";
    // old DD
    if (!fromDD4hep_) {
      // Get CompactView from IdealGeometryRecord
      auto const& myCompactView = iSetup.getData(ddToken_);

      // Build geometry
      geoInfoRoot = detgeomdescbuilder::buildDetGeomDescFromCompactView(myCompactView, isRun2_);
    }
    // DD4hep
    else {
      // Get CompactView from IdealGeometryRecord
      auto const& myCompactView = iSetup.getData(dd4hepToken_);

      // Build geometry
      geoInfoRoot = detgeomdescbuilder::buildDetGeomDescFromCompactView(myCompactView, isRun2_);
    }
  }

  // Build persistent geometry data from geometry
  PDetGeomDesc* serializableData =
      new PDetGeomDesc();  // cond::service::PoolDBOutputService::writeOne interface requires raw pointer.
  int counter = 0;
  if (geoInfoRoot) {
    buildSerializableDataFromGeoInfo(serializableData, geoInfoRoot.get(), counter);
  }

  // Save geometry in the database
  if (serializableData->container_.empty()) {
    throw cms::Exception("PPSGeometryBuilder") << "PDetGeomDesc is empty, no geometry to save in the database.";
  } else {
    if (dbService_.isAvailable()) {
      dbService_->writeOne(serializableData, dbService_->beginOfTime(), "VeryForwardIdealGeometryRecord");
      edm::LogInfo("PPSGeometryBuilder") << "Successfully wrote DB, with " << serializableData->container_.size()
                                         << " PDetGeomDesc items.";
    } else {
      throw cms::Exception("PPSGeometryBuilder") << "PoolDBService required.";
    }
  }
}

/*
 * Build persistent data items to be stored in DB (PDetGeomDesc) from geo info (DetGeomDesc).
 * Recursive, depth-first search.
 */
void PPSGeometryBuilder::buildSerializableDataFromGeoInfo(PDetGeomDesc* serializableData,
                                                          const DetGeomDesc* geoInfo,
                                                          int& counter) {
  PDetGeomDesc::Item serializableItem = buildItemFromDetGeomDesc(geoInfo);
  counter++;

  // Store item in serializableData
  if ((!fromDD4hep_ && counter >= 2)       // Old DD: Skip CMSE
      || (fromDD4hep_ && counter >= 4)) {  // DD4hep: Skip world + OCMS + CMSE
    serializableData->container_.emplace_back(serializableItem);
  }

  // Recursive calls on children
  for (auto& child : geoInfo->components()) {
    buildSerializableDataFromGeoInfo(serializableData, child, counter);
  }
}

/*
 * Build Item from DetGeomDesc info.
 */
PDetGeomDesc::Item PPSGeometryBuilder::buildItemFromDetGeomDesc(const DetGeomDesc* geoInfo) {
  PDetGeomDesc::Item result;
  result.dx_ = geoInfo->translation().X();
  result.dy_ = geoInfo->translation().Y();
  result.dz_ = geoInfo->translation().Z();

  const DDRotationMatrix& rot = geoInfo->rotation();
  rot.GetComponents(result.axx_,
                    result.axy_,
                    result.axz_,
                    result.ayx_,
                    result.ayy_,
                    result.ayz_,
                    result.azx_,
                    result.azy_,
                    result.azz_);
  result.name_ = geoInfo->name();
  result.params_ = geoInfo->params();
  result.copy_ = geoInfo->copyno();
  result.z_ = geoInfo->parentZPosition();
  result.sensorType_ = geoInfo->sensorType();
  result.geographicalID_ = geoInfo->geographicalID();

  return result;
}

DEFINE_FWK_MODULE(PPSGeometryBuilder);
