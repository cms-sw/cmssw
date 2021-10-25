#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/EcalAlgo/interface/EcalBarrelGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalEndcapGeometry.h"
#include "Geometry/EcalAlgo/interface/EcalPreshowerGeometry.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/HcalGeometry.h"
#include "Geometry/HcalTowerAlgo/interface/CaloTowerGeometry.h"
#include "Geometry/ForwardGeometry/interface/ZdcGeometry.h"
#include "Geometry/ForwardGeometry/interface/CastorGeometry.h"

class PCaloGeometryBuilder : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  PCaloGeometryBuilder(const edm::ParameterSet& pset)
      : m_ecalE(pset.getUntrackedParameter<bool>("EcalE", true)),
        m_ecalP(pset.getUntrackedParameter<bool>("EcalP", true)),
        m_hgcal(pset.getUntrackedParameter<bool>("HGCal", false)) {
    const std::string toDB("_toDB");
    ebGeomToken_ = esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", EcalBarrelGeometry::producerTag() + toDB));
    eeGeomToken_ = esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", EcalEndcapGeometry::producerTag() + toDB));
    esGeomToken_ =
        esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", EcalPreshowerGeometry::producerTag() + toDB));
    hgcalGeomToken_ = esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", HGCalGeometry::producerTag() + toDB));
    hcalGeomToken_ = esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", HcalGeometry::producerTag() + toDB));
    ctGeomToken_ = esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", CaloTowerGeometry::producerTag() + toDB));
    zdcGeomToken_ = esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", ZdcGeometry::producerTag() + toDB));
    castGeomToken_ = esConsumes<edm::Transition::BeginRun>(edm::ESInputTag("", CastorGeometry::producerTag() + toDB));
  }

  void beginRun(edm::Run const& iEvent, edm::EventSetup const&) override;
  void analyze(edm::Event const& iEvent, edm::EventSetup const&) override {}
  void endRun(edm::Run const& iEvent, edm::EventSetup const&) override {}

private:
  bool m_ecalE;
  bool m_ecalP;
  bool m_hgcal;
  edm::ESGetToken<CaloSubdetectorGeometry, EcalBarrelGeometry::AlignedRecord> ebGeomToken_;
  edm::ESGetToken<CaloSubdetectorGeometry, EcalEndcapGeometry::AlignedRecord> eeGeomToken_;
  edm::ESGetToken<CaloSubdetectorGeometry, EcalPreshowerGeometry::AlignedRecord> esGeomToken_;
  edm::ESGetToken<CaloSubdetectorGeometry, HGCalGeometry::AlignedRecord> hgcalGeomToken_;
  edm::ESGetToken<CaloSubdetectorGeometry, HcalGeometry::AlignedRecord> hcalGeomToken_;
  edm::ESGetToken<CaloSubdetectorGeometry, CaloTowerGeometry::AlignedRecord> ctGeomToken_;
  edm::ESGetToken<CaloSubdetectorGeometry, ZdcGeometry::AlignedRecord> zdcGeomToken_;
  edm::ESGetToken<CaloSubdetectorGeometry, CastorGeometry::AlignedRecord> castGeomToken_;
};

void PCaloGeometryBuilder::beginRun(const edm::Run&, edm::EventSetup const& es) {
  edm::LogInfo("PCaloGeometryBuilder") << "Writing out " << EcalBarrelGeometry::producerTag() << std::endl;
  auto pGeb = es.getHandle(ebGeomToken_);
  if (m_ecalE) {
    edm::LogInfo("PCaloGeometryBuilder") << "Writing out " << EcalEndcapGeometry::producerTag() << std::endl;
    auto pGeb = es.getHandle(eeGeomToken_);
  }

  if (m_ecalP) {
    edm::LogInfo("PCaloGeometryBuilder") << "Writing out " << EcalPreshowerGeometry::producerTag() << std::endl;
    auto pGes = es.getHandle(esGeomToken_);
  }

  if (m_hgcal) {
    edm::LogInfo("PCaloGeometryBuilder") << "Writing out " << HGCalGeometry::producerTag() << std::endl;
    auto pGhgcal = es.getHandle(hgcalGeomToken_);
    ;
  }

  edm::LogInfo("PCaloGeometryBuilder") << "Writing out " << HcalGeometry::producerTag() << std::endl;
  auto pGhcal = es.getHandle(hcalGeomToken_);

  edm::LogInfo("PCaloGeometryBuilder") << "Writing out " << CaloTowerGeometry::producerTag() << std::endl;
  auto pGct = es.getHandle(ctGeomToken_);

  edm::LogInfo("PCaloGeometryBuilder") << "Writing out " << ZdcGeometry::producerTag() << std::endl;
  auto pGzdc = es.getHandle(zdcGeomToken_);

  edm::LogInfo("PCaloGeometryBuilder") << "Writing out " << CastorGeometry::producerTag() << std::endl;
  auto pGcast = es.getHandle(castGeomToken_);
}

DEFINE_FWK_MODULE(PCaloGeometryBuilder);
