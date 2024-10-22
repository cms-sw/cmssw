#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "MagneticField/Engine/interface/MagneticField.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "CondFormats/SiStripObjects/interface/SiStripLorentzAngle.h"
#include "CalibTracker/Records/interface/SiStripDependentRecords.h"
#include "Geometry/TrackerGeometryBuilder/interface/StripGeomDetUnit.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"

#include "CalibTracker/SiStripCommon/interface/ShallowTools.h"

class SiStripLorentzAngleRunInfoTableProducer : public edm::global::EDProducer<edm::BeginRunProducer> {
public:
  explicit SiStripLorentzAngleRunInfoTableProducer(const edm::ParameterSet& params)
      : m_name{params.getParameter<std::string>("name")},
        m_magFieldName{params.getParameter<std::string>("magFieldName")},
        m_doc{params.existsAs<std::string>("doc") ? params.getParameter<std::string>("doc") : ""},
        m_tkGeomToken{esConsumes<edm::Transition::BeginRun>()},
        m_magFieldToken{esConsumes<edm::Transition::BeginRun>()},
        m_lorentzAngleToken{esConsumes<edm::Transition::BeginRun>()} {
    produces<nanoaod::FlatTable, edm::Transition::BeginRun>();
    produces<nanoaod::FlatTable, edm::Transition::BeginRun>("magField");
  }

  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::string>("name", "Det");
    desc.add<std::string>("magFieldName", "magField");
    desc.add<std::string>("doc", "Run info for the Lorentz angle measurement");
    descriptions.add("siStripLorentzAngleRunInfoTable", desc);
  }

  void globalBeginRunProduce(edm::Run& iRun, edm::EventSetup const& iSetup) const override;

private:
  const std::string m_name, m_magFieldName;
  const std::string m_doc;
  edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> m_tkGeomToken;
  edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> m_magFieldToken;
  edm::ESGetToken<SiStripLorentzAngle, SiStripLorentzAngleDepRcd> m_lorentzAngleToken;
};

namespace {
  template <typename VALUES>
  void addColumn(nanoaod::FlatTable* table, const std::string& name, VALUES&& values, const std::string& doc) {
    using value_type = typename std::remove_reference<VALUES>::type::value_type;
    table->template addColumn<value_type>(name, values, doc);
  }
}  // namespace

void SiStripLorentzAngleRunInfoTableProducer::globalBeginRunProduce(edm::Run& iRun,
                                                                    edm::EventSetup const& iSetup) const {
  const auto& tkGeom = iSetup.getData(m_tkGeomToken);
  const auto& magField = iSetup.getData(m_magFieldToken);
  const auto& lorentzAngle = iSetup.getData(m_lorentzAngleToken);
  std::vector<uint32_t> c_rawid;
  std::vector<float> c_globalZofunitlocalY, c_localB, c_BdotY, c_driftx, c_drifty, c_driftz, c_lorentzAngle;

  auto dets = tkGeom.detsTIB();
  dets.insert(dets.end(), tkGeom.detsTID().begin(), tkGeom.detsTID().end());
  dets.insert(dets.end(), tkGeom.detsTOB().begin(), tkGeom.detsTOB().end());
  dets.insert(dets.end(), tkGeom.detsTEC().begin(), tkGeom.detsTEC().end());
  for (auto det : dets) {
    auto detid = det->geographicalId().rawId();
    const StripGeomDetUnit* stripDet = dynamic_cast<const StripGeomDetUnit*>(tkGeom.idToDet(det->geographicalId()));
    if (stripDet) {
      c_rawid.push_back(detid);
      c_globalZofunitlocalY.push_back(stripDet->toGlobal(LocalVector(0, 1, 0)).z());
      const auto locB = magField.inTesla(stripDet->surface().position());
      c_localB.push_back(locB.mag());
      c_BdotY.push_back(stripDet->surface().toLocal(locB).y());
      const auto drift = shallow::drift(stripDet, magField, lorentzAngle);
      c_driftx.push_back(drift.x());
      c_drifty.push_back(drift.y());
      c_driftz.push_back(drift.z());
      c_lorentzAngle.push_back(lorentzAngle.getLorentzAngle(detid));
    }
  }
  auto out = std::make_unique<nanoaod::FlatTable>(c_rawid.size(), m_name, false, false);
  addColumn(out.get(), "rawid", c_rawid, "DetId");
  addColumn(out.get(), "globalZofunitlocalY", c_globalZofunitlocalY, "z component of a local unit vector along y");
  addColumn(out.get(), "localB", c_localB, "Local magnitude of the magnetic field");
  addColumn(out.get(), "BdotY", c_BdotY, "Magnetic field projection on the local y axis");
  addColumn(out.get(), "driftx", c_driftx, "x component of the drift vector");
  addColumn(out.get(), "drifty", c_drifty, "y component of the drift vector");
  addColumn(out.get(), "driftz", c_driftz, "z component of the drift vector");
  addColumn(out.get(), "lorentzAngle", c_lorentzAngle, "Lorentz angle from database");
  iRun.put(std::move(out));

  auto out2 = std::make_unique<nanoaod::FlatTable>(1, m_magFieldName, true, false);
  out2->addColumnValue<float>(
      "origin", magField.inTesla(GlobalPoint(0, 0, 0)).z(), "z-component of the magnetic field at (0,0,0) in Tesla");
  iRun.put(std::move(out2), "magField");
}

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(SiStripLorentzAngleRunInfoTableProducer);
