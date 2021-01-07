#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHit.h"
#include "RecoEgamma/EgammaTools/interface/HGCalShowerShapeHelper.h"
#include "RecoEgamma/EgammaTools/interface/HGCalClusterTools.h"

class EgammaHLTHGCalIDVarProducer : public edm::stream::EDProducer<> {
public:
  explicit EgammaHLTHGCalIDVarProducer(const edm::ParameterSet&);
  ~EgammaHLTHGCalIDVarProducer() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
  void produce(edm::Event&, const edm::EventSetup&) override;

  class PCAAssocMap {
  public:
    PCAAssocMap(double HGCalShowerShapeHelper::ShowerWidths::*var, const std::string& name) : var_(var), name_(name) {}

    void initMap(const edm::Handle<reco::RecoEcalCandidateCollection>& candHandle) {
      assocMap_ = std::make_unique<reco::RecoEcalCandidateIsolationMap>(candHandle);
    }

    void insert(reco::RecoEcalCandidateRef& ref, const HGCalShowerShapeHelper::ShowerWidths& showerWidths) {
      assocMap_->insert(ref, showerWidths.*var_);
    }

    std::unique_ptr<reco::RecoEcalCandidateIsolationMap> releaseMap() { return std::move(assocMap_); }
    const std::string& name() const { return name_; }

  private:
    double HGCalShowerShapeHelper::ShowerWidths::*var_;
    std::string name_;
    std::unique_ptr<reco::RecoEcalCandidateIsolationMap> assocMap_;
  };

private:
  // ----------member data ---------------------------
  float rCylinder_;
  float hOverECone_;
  std::vector<PCAAssocMap> pcaAssocMaps_;
  const edm::EDGetTokenT<reco::RecoEcalCandidateCollection> recoEcalCandidateToken_;
  const edm::EDGetTokenT<reco::PFRecHitCollection> hgcalRecHitToken_;
  const edm::EDGetTokenT<reco::CaloClusterCollection> layerClusterToken_;
  HGCalShowerShapeHelper ssCalc_;
};

EgammaHLTHGCalIDVarProducer::EgammaHLTHGCalIDVarProducer(const edm::ParameterSet& config)
    : rCylinder_(config.getParameter<double>("rCylinder")),
      hOverECone_(config.getParameter<double>("hOverECone")),
      recoEcalCandidateToken_(
          consumes<reco::RecoEcalCandidateCollection>(config.getParameter<edm::InputTag>("recoEcalCandidateProducer"))),
      hgcalRecHitToken_(consumes<reco::PFRecHitCollection>(config.getParameter<edm::InputTag>("hgcalRecHits"))),
      layerClusterToken_(consumes<reco::CaloClusterCollection>(config.getParameter<edm::InputTag>("layerClusters"))),
      ssCalc_(consumesCollector()) {
  pcaAssocMaps_.emplace_back(PCAAssocMap(&HGCalShowerShapeHelper::ShowerWidths::sigma2xx, "sigma2xx"));
  pcaAssocMaps_.emplace_back(PCAAssocMap(&HGCalShowerShapeHelper::ShowerWidths::sigma2yy, "sigma2yy"));
  pcaAssocMaps_.emplace_back(PCAAssocMap(&HGCalShowerShapeHelper::ShowerWidths::sigma2zz, "sigma2zz"));
  pcaAssocMaps_.emplace_back(PCAAssocMap(&HGCalShowerShapeHelper::ShowerWidths::sigma2xy, "sigma2xy"));
  pcaAssocMaps_.emplace_back(PCAAssocMap(&HGCalShowerShapeHelper::ShowerWidths::sigma2yz, "sigma2yz"));
  pcaAssocMaps_.emplace_back(PCAAssocMap(&HGCalShowerShapeHelper::ShowerWidths::sigma2zx, "sigma2zx"));
  pcaAssocMaps_.emplace_back(PCAAssocMap(&HGCalShowerShapeHelper::ShowerWidths::sigma2uu, "sigma2uu"));
  pcaAssocMaps_.emplace_back(PCAAssocMap(&HGCalShowerShapeHelper::ShowerWidths::sigma2vv, "sigma2vv"));
  pcaAssocMaps_.emplace_back(PCAAssocMap(&HGCalShowerShapeHelper::ShowerWidths::sigma2ww, "sigma2ww"));

  produces<reco::RecoEcalCandidateIsolationMap>("rVar");
  produces<reco::RecoEcalCandidateIsolationMap>("hForHOverE");
  for (auto& var : pcaAssocMaps_) {
    produces<reco::RecoEcalCandidateIsolationMap>(var.name());
  }
}

EgammaHLTHGCalIDVarProducer::~EgammaHLTHGCalIDVarProducer() {}

void EgammaHLTHGCalIDVarProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("recoEcalCandidateProducer", edm::InputTag("hltL1SeededRecoEcalCandidate"));
  desc.add<edm::InputTag>("hgcalRecHits", edm::InputTag("hgcalRecHits"));
  desc.add<edm::InputTag>("layerClusters", edm::InputTag("layerClusters"));
  desc.add<double>("rCylinder", 2.8);
  desc.add<double>("hOverECone", 0.15);
  descriptions.add(("hltEgammaHLTHGCalIDVarProducer"), desc);
}

void EgammaHLTHGCalIDVarProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  auto recoEcalCandHandle = iEvent.getHandle(recoEcalCandidateToken_);
  const auto& hgcalRecHits = iEvent.get(hgcalRecHitToken_);
  const auto& layerClusters = iEvent.get(layerClusterToken_);

  ssCalc_.initPerEvent(iSetup, hgcalRecHits);

  auto rVarMap = std::make_unique<reco::RecoEcalCandidateIsolationMap>(recoEcalCandHandle);
  auto hForHoverEMap = std::make_unique<reco::RecoEcalCandidateIsolationMap>(recoEcalCandHandle);
  for (auto& pcaMap : pcaAssocMaps_) {
    pcaMap.initMap(recoEcalCandHandle);
  }

  for (size_t candNr = 0; candNr < recoEcalCandHandle->size(); candNr++) {
    reco::RecoEcalCandidateRef candRef(recoEcalCandHandle, candNr);
    ssCalc_.initPerObject(candRef->superCluster()->hitsAndFractions());
    rVarMap->insert(candRef, ssCalc_.getRvar(rCylinder_, candRef->superCluster()->energy()));

    float hForHoverE = HGCalClusterTools::hadEnergyInCone(
        candRef->superCluster()->eta(), candRef->superCluster()->phi(), layerClusters, 0., hOverECone_, 0., 0.);
    hForHoverEMap->insert(candRef, hForHoverE);
    auto pcaWidths = ssCalc_.getPCAWidths(rCylinder_);
    for (auto& pcaMap : pcaAssocMaps_) {
      pcaMap.insert(candRef, pcaWidths);
    }
  }
  iEvent.put(std::move(rVarMap), "rVar");
  iEvent.put(std::move(hForHoverEMap), "hForHOverE");
  for (auto& pcaMap : pcaAssocMaps_) {
    iEvent.put(pcaMap.releaseMap(), pcaMap.name());
  }
}

DEFINE_FWK_MODULE(EgammaHLTHGCalIDVarProducer);
