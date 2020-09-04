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

class PPSGeometryBuilder : public edm::one::EDAnalyzer<> {
public:
  explicit PPSGeometryBuilder(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  void buildSerializableDataFromGeoInfo(PDetGeomDesc* serializableData, const DetGeomDesc* geoInfo, int& counter);

  std::string compactViewTag_;
  edm::ESWatcher<IdealGeometryRecord> watcherIdealGeometry_;
  edm::Service<cond::service::PoolDBOutputService> dbService_;
};

PPSGeometryBuilder::PPSGeometryBuilder(const edm::ParameterSet& iConfig)
    : compactViewTag_(iConfig.getUntrackedParameter<std::string>("compactViewTag", "XMLIdealGeometryESSource_CTPPS")) {}

/*
 * Save PPS geo to DB.
 */
void PPSGeometryBuilder::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<cms::DDCompactView> myCompactView;

  if (watcherIdealGeometry_.check(iSetup)) {
    edm::LogInfo("PPSGeometryBuilder") << "Got IdealGeometryRecord ";
    iSetup.get<IdealGeometryRecord>().get(compactViewTag_.c_str(), myCompactView);
  }
  // Build geometry
  auto geoInfoSentinel = detgeomdescbuilder::buildDetGeomDescFromCompactView(*myCompactView);

  // Build persistent geometry data from geometry
  PDetGeomDesc* serializableData =
      new PDetGeomDesc;  // cond::service::PoolDBOutputService::writeOne interface requires raw pointer.
  int counter = 0;
  buildSerializableDataFromGeoInfo(serializableData, geoInfoSentinel.get(), counter);

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
  PDetGeomDesc::Item serializableItem(geoInfo);
  counter++;

  if (counter >= 4) {  // Skip sentinel + OCMS + CMSE
    serializableData->container_.emplace_back(serializableItem);
  }

  for (auto& child : geoInfo->components()) {
    buildSerializableDataFromGeoInfo(serializableData, child, counter);
  }
}

DEFINE_FWK_MODULE(PPSGeometryBuilder);
