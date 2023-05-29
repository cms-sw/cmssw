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

#include "DataFormats/HGCDigi/interface/HGCDigiCollections.h"
#include "DataFormats/HGCRecHit/interface/HGCRecHitCollections.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"

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

class HGCalRecHitToolsPartialWafer : public edm::one::EDAnalyzer<> {
public:
  HGCalRecHitToolsPartialWafer(const edm::ParameterSet& ps);
  ~HGCalRecHitToolsPartialWafer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  void beginJob() override {}
  void endJob() override {}

private:
  template <class T>
  void analyze(const T& collection, const HGCalGeometry* geom);

  const edm::InputTag source_;
  const std::string nameSense_;
  const bool checkDigi_;
  const edm::EDGetTokenT<HGCalDigiCollection> digiToken_;
  const edm::EDGetTokenT<HGCRecHitCollection> recHitToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> geomToken_;
  hgcal::RecHitTools tool_;
};

HGCalRecHitToolsPartialWafer::HGCalRecHitToolsPartialWafer(const edm::ParameterSet& ps)
    : source_(ps.getParameter<edm::InputTag>("source")),
      nameSense_(ps.getParameter<std::string>("nameSense")),
      checkDigi_(ps.getParameter<bool>("checkDigi")),
      digiToken_(consumes<HGCalDigiCollection>(source_)),
      recHitToken_(consumes<HGCRecHitCollection>(source_)),
      geomToken_(esConsumes<CaloGeometry, CaloGeometryRecord>()) {
  edm::LogVerbatim("HGCalSim") << "Test Hit ID using Digi(1)/RecHit(0): " << checkDigi_ << " for " << nameSense_
                               << " with module Label: " << source_;
}

void HGCalRecHitToolsPartialWafer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("source", edm::InputTag("simHGCalUnsuppressedDigis", "EE"));
  desc.add<std::string>("nameSense", "HGCalEESensitive");
  desc.add<bool>("checkDigi", true);
  descriptions.add("hgcalCheckToolDigiEE", desc);
}

void HGCalRecHitToolsPartialWafer::analyze(const edm::Event& e, const edm::EventSetup& iS) {
  // get HGCal geometry constant
  const CaloGeometry geo = iS.getData(geomToken_);
  const HGCalGeometry* geom = (nameSense_ == "HGCalEESensitive")
                                  ? static_cast<const HGCalGeometry*>(
                                        geo.getSubdetectorGeometry(DetId::HGCalEE, ForwardSubdetector::ForwardEmpty))
                                  : static_cast<const HGCalGeometry*>(
                                        geo.getSubdetectorGeometry(DetId::HGCalHSi, ForwardSubdetector::ForwardEmpty));
  //Setup the tool
  tool_.setGeometry(geo);

  // get the hit collection
  if (checkDigi_) {
    const auto& collection = e.getHandle(digiToken_);
    if (collection.isValid()) {
      edm::LogVerbatim("HGCalSim") << "HGCalRecHitToolsPartialWafer: Finds Digi Collection for " << nameSense_
                                   << " with " << collection->size() << " hits";
      analyze(*(collection.product()), geom);
    } else {
      edm::LogVerbatim("HGCalSim") << "HGCalRecHitToolsPartialWafer: Cannot find Digi collection for " << nameSense_;
    }
  } else {
    const auto& collection = e.getHandle(recHitToken_);
    if (collection.isValid()) {
      edm::LogVerbatim("HGCalSim") << "HGCalRecHitToolsPartialWafer: Finds RecHit Collection for " << nameSense_
                                   << " with " << collection->size() << " hits";
      analyze(*(collection.product()), geom);
    } else {
      edm::LogVerbatim("HGCalSim") << "HGCalRecHitToolsPartialWafer: Cannot find RecHit collection for " << nameSense_;
    }
  }
}

template <class T>
void HGCalRecHitToolsPartialWafer::analyze(const T& collection, const HGCalGeometry* geom) {
  const HGCalDDDConstants& hgc = geom->topology().dddConstants();
  uint32_t nhits = collection.size();
  uint32_t good(0), allSi(0), all(0);

  // Loop over all hits
  for (const auto& it : collection) {
    ++all;
    DetId id(it.id());
    if ((id.det() == DetId::HGCalEE) || (id.det() == DetId::HGCalHSi)) {
      ++allSi;
      HGCSiliconDetId hid(id);
      const auto& info = hgc.waferInfo(hid.layer(), hid.waferU(), hid.waferV());
      // Only partial wafers
      if (info.part != HGCalTypes::WaferFull) {
        ++good;
        GlobalPoint pos1 = geom->getPosition(id);
        GlobalPoint pos2 = tool_.getPosition(id);
        edm::LogVerbatim("HGCalSim") << "Hit[" << all << ":" << allSi << ":" << good << "]" << HGCSiliconDetId(id)
                                     << " Wafer Type:Part:Orient:Cassette " << info.type << ":" << info.part << ":"
                                     << info.orient << ":" << info.cassette << " at (" << pos1.x() << ", " << pos1.y()
                                     << ", " << pos1.z() << ") or (" << pos2.x() << ", " << pos2.y() << ", " << pos2.z()
                                     << ")";
      }
    }
  }
  edm::LogVerbatim("HGCalSim") << "Total hits = " << all << ":" << nhits << " Good DetIds = " << allSi << ":" << good;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HGCalRecHitToolsPartialWafer);
