#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"

#include "SimDataFormats/CaloHit/interface/PCaloHit.h"
#include "SimDataFormats/CaloHit/interface/PCaloHitContainer.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/HGCalTopology.h"
#include "Geometry/HGCalGeometry/interface/HGCalGeometry.h"
#include "Geometry/HGCalCommonData/interface/HGCalDDDConstants.h"
#include "Geometry/HGCalCommonData/interface/HGCalParameters.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"

#include "RecoLocalCalo/HGCalRecAlgos/interface/RecHitTools.h"

#include <string>
#include <vector>

class HGCalToolTesterPartialWafer : public edm::one::EDAnalyzer<edm::one::WatchRuns> {
public:
  HGCalToolTesterPartialWafer(const edm::ParameterSet& ps);
  ~HGCalToolTesterPartialWafer() override = default;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void beginRun(edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override {}
  void beginJob() override {}
  void endJob() override {}

private:
  const std::string g4Label_, caloHitSource_, nameSense_;
  const edm::EDGetTokenT<edm::PCaloHitContainer> tok_calo_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  const HGCalGeometry* geom_;
  CaloGeometry geo_;
  hgcal::RecHitTools tool_;
};

HGCalToolTesterPartialWafer::HGCalToolTesterPartialWafer(const edm::ParameterSet& ps)
    : g4Label_(ps.getParameter<std::string>("moduleLabel")),
      caloHitSource_(ps.getParameter<std::string>("caloHitSource")),
      nameSense_(ps.getParameter<std::string>("nameSense")),
      tok_calo_(consumes<edm::PCaloHitContainer>(edm::InputTag(g4Label_, caloHitSource_))),
      geomToken_(esConsumes<CaloGeometry, CaloGeometryRecord, edm::Transition::BeginRun>()),
      geom_(nullptr) {
  edm::LogVerbatim("HGCalSim") << "Test Hit ID using SimHits for " << nameSense_ << " with module Label: " << g4Label_
                               << "   Hits: " << caloHitSource_;
}

void HGCalToolTesterPartialWafer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("moduleLabel", "g4SimHits");
  desc.add<std::string>("caloHitSource", "HGCHitsEE");
  desc.add<std::string>("nameSense", "HGCalEESensitive");
  descriptions.add("hgcalToolTesterPartialWaferEE", desc);
}

void HGCalToolTesterPartialWafer::beginRun(const edm::Run&, const edm::EventSetup& iSetup) {
  //Setup the tool
  geo_ = iSetup.getData(geomToken_);
  geom_ = (nameSense_ == "HGCalEESensitive") ? static_cast<const HGCalGeometry*>(geo_.getSubdetectorGeometry(
                                                   DetId::HGCalEE, ForwardSubdetector::ForwardEmpty))
                                             : static_cast<const HGCalGeometry*>(geo_.getSubdetectorGeometry(
                                                   DetId::HGCalHSi, ForwardSubdetector::ForwardEmpty));
  edm::LogVerbatim("HGCalSim") << "HGCalToolTesterPartialWafer: beginRun Called for " << nameSense_;
}

void HGCalToolTesterPartialWafer::analyze(const edm::Event& e, const edm::EventSetup& iS) {
  // get HGCal geometry constant
  tool_.setGeometry(geo_);
  const HGCalDDDConstants& hgc = geom_->topology().dddConstants();

  // get the hit collection
  const edm::Handle<edm::PCaloHitContainer>& hitsCalo = e.getHandle(tok_calo_);
  bool getHits = (hitsCalo.isValid());
  uint32_t nhits = (getHits) ? hitsCalo->size() : 0;
  uint32_t good(0), allSi(0), all(0);
  edm::LogVerbatim("HGCalSim") << "HGCalToolTesterPartialWafer: Input flags Hits " << getHits << " with " << nhits
                               << " hits";

  if (getHits) {
    std::vector<PCaloHit> hits;
    hits.insert(hits.end(), hitsCalo->begin(), hitsCalo->end());
    if (!hits.empty()) {
      // Loop over all hits
      for (auto hit : hits) {
        ++all;
        DetId id(hit.id());
        if ((id.det() == DetId::HGCalEE) || (id.det() == DetId::HGCalHSi)) {
          ++allSi;
          HGCSiliconDetId hid(id);
          const auto& info = hgc.waferInfo(hid.layer(), hid.waferU(), hid.waferV());
          // Only partial wafers
          if (info.part != HGCalTypes::WaferFull) {
            ++good;
            GlobalPoint pos1 = geom_->getPosition(id);
            GlobalPoint pos2 = tool_.getPosition(id);
            edm::LogVerbatim("HGCalSim") << "Hit[" << all << ":" << allSi << ":" << good << "]" << HGCSiliconDetId(id)
                                         << " Wafer Type:Part:Orient:Cassette " << info.type << ":" << info.part << ":"
                                         << info.orient << ":" << info.cassette << " at (" << pos1.x() << ", "
                                         << pos1.y() << ", " << pos1.z() << ") or (" << pos2.x() << ", " << pos2.y()
                                         << ", " << pos2.z() << ")";
          }
        }
      }
    }
  }
  edm::LogVerbatim("HGCalSim") << "Total hits = " << all << ":" << nhits << " Good DetIds = " << allSi << ":" << good;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalToolTesterPartialWafer);
