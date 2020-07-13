/****************************************************************************
 *
 * Author:
 *
 *  Wagner Carvalho (wcarvalh@cern.ch)
 *
 *  DB builder for PPS geometry
 *
 ****************************************************************************/

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
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
#include "Geometry/VeryForwardGeometryBuilder/plugins/dd4hep/PPSGeometryESProducer.cc"

class PPSGeometryESProducer;


class PPSGeometryBuilder : public edm::one::EDAnalyzer<> {
public:
  explicit PPSGeometryBuilder(const edm::ParameterSet&);
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  void buildPDetFromDetGeomDesc(const DetGeomDesc* geoInfo, PDetGeomDesc* gd, int& counter);

  std::string compactViewTag_;
  edm::ESWatcher<IdealGeometryRecord> watcherIdealGeometry_;
  edm::Service<cond::service::PoolDBOutputService> dbservice_;
};


PPSGeometryBuilder::PPSGeometryBuilder(const edm::ParameterSet& iConfig)
    : compactViewTag_(iConfig.getUntrackedParameter<std::string>("compactViewTag", "XMLIdealGeometryESSource_CTPPS")) {
}


void PPSGeometryBuilder::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::ESHandle<cms::DDCompactView> cpv;

  if (watcherIdealGeometry_.check(iSetup)) {
    std::cout << "Got IdealGeometryRecord" << std::endl;
    iSetup.get<IdealGeometryRecord>().get(compactViewTag_.c_str(), cpv);
  }

  auto sentinel = PPSGeometryESProducer::buildDetGeomDescFromCompactView(*cpv);

  // Persistent geometry data
  PDetGeomDesc* pdet = new PDetGeomDesc;
  int counter = 0;
  // Build geometry
  buildPDetFromDetGeomDesc(sentinel.get(), pdet, counter);

  // Save geometry in the database
  if (pdet->container_.empty()) {
    throw cms::Exception("PPSGeometryBuilder") << "PDetGeomDesc is empty, no geometry to save in the database.";
  } else {
    if (dbservice_.isAvailable()) {
      dbservice_->writeOne(pdet, dbservice_->beginOfTime(), "VeryForwardIdealGeometryRecord");
    } else {
      throw cms::Exception("PoolDBService required.");
    }
  }
}


void PPSGeometryBuilder::buildPDetFromDetGeomDesc(const DetGeomDesc* geoInfo, PDetGeomDesc* gd, int& counter) {
  PDetGeomDesc::Item item(geoInfo);
  counter++;

  if (counter >= 4) {  // sentinel + OCMS + CMSE
    gd->container_.emplace_back(item); 
  }

  for (auto& child : geoInfo->components()) {
    buildPDetFromDetGeomDesc(child, gd, counter);
  }
}


DEFINE_FWK_MODULE(PPSGeometryBuilder);
