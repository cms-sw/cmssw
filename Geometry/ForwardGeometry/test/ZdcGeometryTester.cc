#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESTransientHandle.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/ForwardGeometry/interface/ZdcGeometry.h"
#include "DataFormats/HcalDetId/interface/HcalZDCDetId.h"

class ZdcGeometryTester : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  explicit ZdcGeometryTester(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override {}
  void beginRun(edm::Run const&, edm::EventSetup const&) override {}
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void doTest(const ZdcGeometry& geometry);

  // ----------member data ---------------------------
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> tokGeom_;
};

ZdcGeometryTester::ZdcGeometryTester(const edm::ParameterSet&)
    : tokGeom_{esConsumes<CaloGeometry, CaloGeometryRecord>()} {}

void ZdcGeometryTester::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.add("zdcGeometryTester", desc);
}

void ZdcGeometryTester::analyze(edm::Event const&, edm::EventSetup const& iSetup) {
  auto geo = &iSetup.getData(tokGeom_);
  const ZdcGeometry* geom =
      dynamic_cast<const ZdcGeometry*>(geo->getSubdetectorGeometry(DetId::Calo, HcalZDCDetId::SubdetectorId));
  doTest(*geom);
}

void ZdcGeometryTester::doTest(const ZdcGeometry& geom) {
  // Total number of valid cells
  std::vector<DetId> valids = geom.getValidDetIds();
  CaloSubdetectorGeometry::TrVec tVec;
  CaloSubdetectorGeometry::IVec iVec;
  CaloSubdetectorGeometry::DimVec dVec;
  CaloSubdetectorGeometry::IVec dins;
  geom.getSummary(tVec, iVec, dVec, dins);
  edm::LogVerbatim("HCalGeom") << "\nTotal number of dense indices: " << dins.size() << " valid de IDs "
                               << valids.size() << std::endl;
  std::vector<int> ndet(4, 0);
  std::vector<std::string> dets = {"EM", "HAD", "LUM", "RPD"};
  int unknown(0);
  for (auto& di : valids) {
    HcalZDCDetId id = HcalZDCDetId(di);
    HcalZDCDetId::Section section = id.section();
    if (section == HcalZDCDetId::EM)
      ++ndet[0];
    else if (section == HcalZDCDetId::HAD)
      ++ndet[1];
    else if (section == HcalZDCDetId::LUM)
      ++ndet[2];
    else if (section == HcalZDCDetId::RPD)
      ++ndet[3];
    else
      ++unknown;
  }
  std::ostringstream st1;
  st1 << "Number of IDs for";
  for (unsigned int k = 0; k < ndet.size(); ++k)
    st1 << " " << dets[k] << ": " << ndet[k];
  edm::LogVerbatim("HCalGeom") << st1.str() << " and unknown section:" << unknown;

  // Positions and Get closest cell
  edm::LogVerbatim("HCalGeom") << "\nTest on Positions and getclosest cell"
                               << "\n=====================================";
  for (auto& di : valids) {
    HcalZDCDetId id = HcalZDCDetId(di);
    auto cell = geom.getGeometry(static_cast<DetId>(id));
    if (cell) {
      GlobalPoint pos = cell->getPosition();
      DetId idn = geom.getClosestCell(pos);
      std::string found = (id.rawId() == idn.rawId()) ? " Matched " : " ***** ERROR No Match *****";
      edm::LogVerbatim("HCalGeom") << id << " at " << pos << found;
    } else {
      edm::LogVerbatim("HcalGeom") << id << " ***** ERROR No Cell *****";
    }
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(ZdcGeometryTester);
