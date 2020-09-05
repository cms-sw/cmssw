/****************************************************************************
 *
 *  DB builder for PPS geometry
 *
 *  Author: Wagner Carvalho (wcarvalh@cern.ch)
 *  Rewritten / Moved out common functionalities to DetGeomDesc(Builder) by Gabrielle Hugo.
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
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "CondFormats/GeometryObjects/interface/PDetGeomDesc.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDescBuilder.h"


class PPSGeometryBuilderDD4hep : public edm::one::EDAnalyzer<> {
public:
  explicit PPSGeometryBuilderDD4hep(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  void buildSerializableDataFromGeoInfo(PDetGeomDesc* serializableData, const DetGeomDesc* geoInfo, int& counter);
  PDetGeomDesc::Item buildItemFromDetGeomDesc(const DetGeomDesc* geoInfo);

  std::string compactViewTag_;
  edm::ESWatcher<IdealGeometryRecord> watcherIdealGeometry_;
  edm::Service<cond::service::PoolDBOutputService> dbService_;
};

PPSGeometryBuilderDD4hep::PPSGeometryBuilderDD4hep(const edm::ParameterSet& iConfig)
    : compactViewTag_(iConfig.getUntrackedParameter<std::string>("compactViewTag", "XMLIdealGeometryESSource_CTPPS")) {}

/*
 * Save PPS geo to DB.
 */
void PPSGeometryBuilderDD4hep::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<cms::DDCompactView> myCompactView;

  if (watcherIdealGeometry_.check(iSetup)) {
    edm::LogInfo("PPSGeometryBuilderDD4hep") << "Got IdealGeometryRecord ";
    iSetup.get<IdealGeometryRecord>().get(compactViewTag_.c_str(), myCompactView);
  }
  // Build geometry
  auto geoInfoRoot = detgeomdescbuilder::buildDetGeomDescFromCompactView(*myCompactView);

  // Build persistent geometry data from geometry
  PDetGeomDesc* serializableData =
    new PDetGeomDesc();  // cond::service::PoolDBOutputService::writeOne interface requires raw pointer.
  int counter = 0;
  buildSerializableDataFromGeoInfo(serializableData, geoInfoRoot.get(), counter);

  // Save geometry in the database
  if (serializableData->container_.empty()) {
    throw cms::Exception("PPSGeometryBuilderDD4hep") << "PDetGeomDesc is empty, no geometry to save in the database.";
  } else {
    if (dbService_.isAvailable()) {
      dbService_->writeOne(serializableData, dbService_->beginOfTime(), "VeryForwardIdealGeometryRecord");
      edm::LogInfo("PPSGeometryBuilderDD4hep") << "Successfully wrote DB, with " << serializableData->container_.size()
                                         << " PDetGeomDesc items.";
    } else {
      throw cms::Exception("PPSGeometryBuilderDD4hep") << "PoolDBService required.";
    }
  }
}

/*
 * Build persistent data items to be stored in DB (PDetGeomDesc) from geo info (DetGeomDesc).
 * Recursive, depth-first search.
 */
void PPSGeometryBuilderDD4hep::buildSerializableDataFromGeoInfo(PDetGeomDesc* serializableData,
                                                          const DetGeomDesc* geoInfo,
                                                          int& counter) {
  PDetGeomDesc::Item serializableItem = buildItemFromDetGeomDesc(geoInfo);  
  counter++;

  // Store item in serializableData
  if (counter >= 4) {  // Skip world + OCMS + CMSE
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
PDetGeomDesc::Item PPSGeometryBuilderDD4hep::buildItemFromDetGeomDesc(const DetGeomDesc* geoInfo) {
  PDetGeomDesc::Item result;
  result.dx_ = geoInfo->translation().X();
  result.dy_ = geoInfo->translation().Y();
  result.dz_ = geoInfo->translation().Z();

  const DDRotationMatrix& rot = geoInfo->rotation();
  rot.GetComponents(result.axx_, result.axy_, result.axz_, 
		    result.ayx_, result.ayy_, result.ayz_, 
		    result.azx_, result.azy_, result.azz_);
  result.name_ = geoInfo->name();
  result.params_ = geoInfo->params();
  result.copy_ = geoInfo->copyno();
  result.z_ = geoInfo->parentZPosition();
  result.sensorType_ = geoInfo->sensorType();
  result.geographicalID_ = geoInfo->geographicalID();

  return result;
}

DEFINE_FWK_MODULE(PPSGeometryBuilderDD4hep);
