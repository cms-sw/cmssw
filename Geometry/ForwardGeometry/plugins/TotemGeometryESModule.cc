/****************************************************************************
*
* This is a part of TOTEM offline software.
* Author:
*   Laurent Forthomme
*
****************************************************************************/

#include <memory>

#include "FWCore/Framework/interface/ModuleFactory.h"
#include "FWCore/Framework/interface/ESProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DetectorDescription/DDCMS/interface/DDCompactView.h"

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TotemGeometryRcd.h"
#include "Geometry/ForwardGeometry/interface/TotemGeometry.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDescBuilder.h"

class TotemGeometryESModule : public edm::ESProducer {
public:
  TotemGeometryESModule(const edm::ParameterSet&);

  std::unique_ptr<DetGeomDesc> produceGeomDesc(const IdealGeometryRecord&);
  std::unique_ptr<TotemGeometry> produceGeometry(const TotemGeometryRcd&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  const edm::ESGetToken<DetGeomDesc, TotemGeometryRcd> detGeomDescToken_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4hepToken_;
};

TotemGeometryESModule::TotemGeometryESModule(const edm::ParameterSet& iConfig)
    : detGeomDescToken_{setWhatProduced(this, &TotemGeometryESModule::produceGeometry).consumes<DetGeomDesc>(edm::ESInputTag())},
      dd4hepToken_{setWhatProduced(this, &TotemGeometryESModule::produceGeomDesc).consumes<cms::DDCompactView>(edm::ESInputTag("", iConfig.getParameter<std::string>("compactViewTag")))} {
}

std::unique_ptr<DetGeomDesc> TotemGeometryESModule::produceGeomDesc(const IdealGeometryRecord& iRecord) {
  const auto& dd4hep = iRecord.get(dd4hepToken_);
  return detgeomdescbuilder::buildDetGeomDescFromCompactView(dd4hep, false);
}

std::unique_ptr<TotemGeometry> TotemGeometryESModule::produceGeometry(const TotemGeometryRcd& iRecord) {
  const auto& geom_desc = iRecord.get(detGeomDescToken_);
  return std::make_unique<TotemGeometry>(&geom_desc);
}

void TotemGeometryESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("compactViewTag", std::string("XMLIdealGeometryESSource"));
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(TotemGeometryESModule);
