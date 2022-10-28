// system include files
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "CondFormats/EcalObjects/interface/EcalPFRecHitThresholds.h"
#include "CondFormats/DataRecord/interface/EcalPFRecHitThresholdsRcd.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/Records/interface/CaloGeometryRecord.h"

//#define EDM_ML_DEBUG

class HcalTestThreshold : public edm::one::EDAnalyzer<> {
public:
  explicit HcalTestThreshold(edm::ParameterSet const&);
  ~HcalTestThreshold() override {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;
  double eThreshold(const DetId& id, const CaloGeometry* geo) const;

  const int etaMin_, etaMax_, phiValue_;
  const std::vector<int> ixNumbers_, iyNumbers_;
  const double hitEthrEB_, hitEthrEE0_, hitEthrEE1_;
  const double hitEthrEE2_, hitEthrEE3_;
  const double hitEthrEELo_, hitEthrEEHi_;

  const edm::ESGetToken<EcalPFRecHitThresholds, EcalPFRecHitThresholdsRcd> ecalPFRecHitThresholdsToken_;
  const edm::ESGetToken<CaloGeometry, CaloGeometryRecord> caloGeometryToken_;
};

HcalTestThreshold::HcalTestThreshold(const edm::ParameterSet& iConfig)
    : etaMin_(iConfig.getParameter<int>("etaMin")),
      etaMax_(iConfig.getParameter<int>("etaMax")),
      phiValue_(iConfig.getParameter<int>("phiValue")),
      ixNumbers_(iConfig.getParameter<std::vector<int>>("ixEENumbers")),
      iyNumbers_(iConfig.getParameter<std::vector<int>>("iyEENumbers")),
      hitEthrEB_(iConfig.getParameter<double>("EBHitEnergyThreshold")),
      hitEthrEE0_(iConfig.getParameter<double>("EEHitEnergyThreshold0")),
      hitEthrEE1_(iConfig.getParameter<double>("EEHitEnergyThreshold1")),
      hitEthrEE2_(iConfig.getParameter<double>("EEHitEnergyThreshold2")),
      hitEthrEE3_(iConfig.getParameter<double>("EEHitEnergyThreshold3")),
      hitEthrEELo_(iConfig.getParameter<double>("EEHitEnergyThresholdLow")),
      hitEthrEEHi_(iConfig.getParameter<double>("EEHitEnergyThresholdHigh")),
      ecalPFRecHitThresholdsToken_(esConsumes()),
      caloGeometryToken_(esConsumes()) {
  edm::LogVerbatim("HcalIsoTrack") << "Parameters read from config file \n"
                                   << "\tThreshold for EB " << hitEthrEB_ << "\t for EE " << hitEthrEE0_ << ":"
                                   << hitEthrEE1_ << ":" << hitEthrEE2_ << ":" << hitEthrEE3_ << ":" << hitEthrEELo_
                                   << ":" << hitEthrEEHi_;
  edm::LogVerbatim("HcalIsoTrack") << "\tRange in eta for EB " << etaMin_ << ":" << etaMax_ << "\tPhi value "
                                   << phiValue_;
  std::ostringstream st1, st2;
  st1 << "\t" << ixNumbers_.size() << " EE ix Numbers";
  for (const auto& ix : ixNumbers_)
    st1 << ": " << ix;
  edm::LogVerbatim("HcalIsoTrack") << st1.str();
  st2 << "\t" << iyNumbers_.size() << " EE iy Numbers";
  for (const auto& iy : iyNumbers_)
    st2 << ": " << iy;
  edm::LogVerbatim("HcalIsoTrack") << st2.str();
}

// ------------ method called when starting to processes a run  ------------
void HcalTestThreshold::analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup) {
  edm::LogVerbatim("HcalIsoTrack") << "Run " << iEvent.id().run() << " Event " << iEvent.id().event();

  auto const& thresholds = iSetup.getData(ecalPFRecHitThresholdsToken_);
  auto const& geo = &iSetup.getData(caloGeometryToken_);

  // First EB
  edm::LogVerbatim("HcalIsoTrack") << "\n\nThresholds for EB Towers";
  for (int eta = etaMin_; eta < etaMax_; ++eta) {
    if (eta != 0) {
      EBDetId id(eta, phiValue_, EBDetId::ETAPHIMODE);
      edm::LogVerbatim("HcalIsoTrack") << id << " Eta " << (geo->getPosition(id)).eta() << " Thresholds:: old "
                                       << eThreshold(id, geo) << " new " << thresholds[id];
    }
  }

  // Next EE
#ifdef EDM_ML_DEBUG
  auto const& validId = geo->getValidDetIds(DetId::Ecal, 2);
  edm::LogVerbatim("HcalIsoTrack") << "\n\nList of " << validId.size() << " valid DetId's of EE";
  for (const auto& id : validId)
    edm::LogVerbatim("HcalIsoTrack") << EEDetId(id) << " SC " << EEDetId(id).isc() << " CR " << EEDetId(id).ic()
                                     << " Eta " << (geo->getPosition(id)).eta();
#endif
  edm::LogVerbatim("HcalIsoTrack") << "\n\nThresholds for EE Towers";
  for (int zside = 0; zside < 2; ++zside) {
    int iz = 2 * zside - 1;
    for (const auto& ix : ixNumbers_) {
      for (const auto& iy : iyNumbers_) {
        EEDetId id(ix, iy, iz, EEDetId::XYMODE);
        edm::LogVerbatim("HcalIsoTrack") << id << " Eta " << (geo->getPosition(id)).eta() << " Thresholds:: old "
                                         << eThreshold(id, geo) << " new " << thresholds[id];
      }
    }
  }
}

void HcalTestThreshold::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  std::vector<int> ixNumb = {1, 10, 20, 30, 39};
  std::vector<int> iyNumb = {41, 43, 45, 47};
  desc.add<int>("etaMin", -85);
  desc.add<int>("etaMax", 85);
  desc.add<int>("phiValue", 1);
  desc.add<std::vector<int>>("ixEENumbers", ixNumb);
  desc.add<std::vector<int>>("iyEENumbers", iyNumb);
  // energy thershold for ECAL (from Egamma group)
  desc.add<double>("EBHitEnergyThreshold", 0.08);
  desc.add<double>("EEHitEnergyThreshold0", 0.30);
  desc.add<double>("EEHitEnergyThreshold1", 0.00);
  desc.add<double>("EEHitEnergyThreshold2", 0.00);
  desc.add<double>("EEHitEnergyThreshold3", 0.00);
  desc.add<double>("EEHitEnergyThresholdLow", 0.30);
  desc.add<double>("EEHitEnergyThresholdHigh", 0.30);
  descriptions.add("hcalTestThreshold", desc);
}

double HcalTestThreshold::eThreshold(const DetId& id, const CaloGeometry* geo) const {
  const GlobalPoint& pos = geo->getPosition(id);
  double eta = std::abs(pos.eta());
  double eThr(hitEthrEB_);
  if (id.subdetId() != EcalBarrel) {
    eThr = (((eta * hitEthrEE3_ + hitEthrEE2_) * eta + hitEthrEE1_) * eta + hitEthrEE0_);
    if (eThr < hitEthrEELo_)
      eThr = hitEthrEELo_;
    else if (eThr > hitEthrEEHi_)
      eThr = hitEthrEEHi_;
  }
  return eThr;
}

//define this as a plug-in
DEFINE_FWK_MODULE(HcalTestThreshold);
