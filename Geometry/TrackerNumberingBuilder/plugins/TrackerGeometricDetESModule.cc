#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DetectorDescription/Core/interface/DDCompactView.h"
#include "DetectorDescription/Core/interface/DDVectorGetter.h"
#include "DetectorDescription/Core/interface/DDutils.h"
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"
#include "Geometry/TrackerNumberingBuilder/plugins/DDDCmsTrackerContruction.h"
#include "Geometry/TrackerNumberingBuilder/plugins/CondDBCmsTrackerConstruction.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "CondFormats/GeometryObjects/interface/PGeometricDet.h"

#include <memory>

class TrackerGeometricDetESModule : public edm::ESProducer {
public:
  TrackerGeometricDetESModule(const edm::ParameterSet& p);
  ~TrackerGeometricDetESModule(void) override;
  std::unique_ptr<GeometricDet> produce(const IdealGeometryRecord&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> ddToken_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4hepToken_;
  edm::ESGetToken<PGeometricDet, IdealGeometryRecord> pgToken_;
  bool fromDDD_;
  bool fromDD4hep_;
};

using namespace edm;

TrackerGeometricDetESModule::TrackerGeometricDetESModule(const edm::ParameterSet& p)
    : fromDDD_(p.getParameter<bool>("fromDDD")), fromDD4hep_(p.getParameter<bool>("fromDD4hep")) {
  auto cc = setWhatProduced(this);
  if (fromDDD_) {
    ddToken_ = cc.consumes<DDCompactView>(edm::ESInputTag());
  } else if (fromDD4hep_) {
    dd4hepToken_ = cc.consumes<cms::DDCompactView>(edm::ESInputTag());
  } else {
    pgToken_ = cc.consumes<PGeometricDet>(edm::ESInputTag());
  }
}

TrackerGeometricDetESModule::~TrackerGeometricDetESModule(void) {}

void TrackerGeometricDetESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription descDB;
  descDB.add<bool>("fromDDD", false);
  descDB.add<bool>("fromDD4hep", false);
  descriptions.add("trackerNumberingGeometryDB", descDB);

  edm::ParameterSetDescription desc;
  desc.add<bool>("fromDDD", true);
  desc.add<bool>("fromDD4hep", false);
  descriptions.add("trackerNumberingGeometry", desc);

  edm::ParameterSetDescription descDD4hep;
  descDD4hep.add<bool>("fromDDD", false);
  descDD4hep.add<bool>("fromDD4hep", true);
  descriptions.add("DD4hep_trackerNumberingGeometry", descDD4hep);
}

std::unique_ptr<GeometricDet> TrackerGeometricDetESModule::produce(const IdealGeometryRecord& iRecord) {
  if (fromDDD_) {
    edm::ESTransientHandle<DDCompactView> cpv = iRecord.getTransientHandle(ddToken_);

    return DDDCmsTrackerContruction::construct(*cpv, dbl_to_int(DDVectorGetter::get("detIdShifts")));
  } else if (fromDD4hep_) {
    edm::ESTransientHandle<cms::DDCompactView> cpv = iRecord.getTransientHandle(dd4hepToken_);

    return DDDCmsTrackerContruction::construct(*cpv, cpv->getVector<int>("detIdShifts"));
  } else {
    auto const& pgd = iRecord.get(pgToken_);

    return CondDBCmsTrackerConstruction::construct(pgd);
  }
}

DEFINE_FWK_EVENTSETUP_MODULE(TrackerGeometricDetESModule);
