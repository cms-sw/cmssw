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

#include "DetectorDescription/Core/interface/DDCompactView.h"   // DDL
#include "DetectorDescription/DDCMS/interface/DDCompactView.h"  // DD4hep

#include "Geometry/Records/interface/IdealGeometryRecord.h"
#include "Geometry/Records/interface/TotemGeometryRcd.h"
#include "Geometry/ForwardGeometry/interface/TotemGeometry.h"
#include "Geometry/VeryForwardGeometryBuilder/interface/DetGeomDescBuilder.h"

class TotemGeometryESModule : public edm::ESProducer {
public:
  TotemGeometryESModule(const edm::ParameterSet&);

  std::unique_ptr<TotemGeometry> produce(const TotemGeometryRcd&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  edm::ESGetToken<DDCompactView, IdealGeometryRecord> ddlToken_;
  edm::ESGetToken<cms::DDCompactView, IdealGeometryRecord> dd4hepToken_;

  const bool useDDL_;
  const bool useDD4hep_;
};

TotemGeometryESModule::TotemGeometryESModule(const edm::ParameterSet& iConfig)
    : useDDL_(iConfig.getParameter<bool>("useDDL")), useDD4hep_(iConfig.getParameter<bool>("useDD4hep")) {
  auto cc = setWhatProduced(this);
  if (useDDL_)
    ddlToken_ = cc.consumes();
  else if (useDD4hep_)
    dd4hepToken_ = cc.consumes();
  else
    throw cms::Exception("TotemGeometryESModule") << "Geometry must either be retrieved from a DDL or DD4hep payload!";
}

std::unique_ptr<TotemGeometry> TotemGeometryESModule::produce(const TotemGeometryRcd& iRecord) {
  if (useDDL_) {
    edm::ESTransientHandle<DDCompactView> hnd = iRecord.getTransientHandle(ddlToken_);
    return std::make_unique<TotemGeometry>(detgeomdescbuilder::buildDetGeomDescFromCompactView(*hnd, false).get());
  } else {
    edm::ESTransientHandle<cms::DDCompactView> hnd = iRecord.getTransientHandle(dd4hepToken_);
    return std::make_unique<TotemGeometry>(detgeomdescbuilder::buildDetGeomDescFromCompactView(*hnd, false).get());
  }
}

void TotemGeometryESModule::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("useDDL", true);
  desc.add<bool>("useDD4hep", false);
  descriptions.addWithDefaultLabel(desc);
}

//define this as a plug-in
DEFINE_FWK_EVENTSETUP_MODULE(TotemGeometryESModule);
