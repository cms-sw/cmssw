#include <iostream>
#include <vector>
#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateIsolation.h"

#include "DataFormats/CaloRecHit/interface/CaloCluster.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

#include "RecoEgamma/EgammaTools/interface/HGCalClusterTools.h"

template <typename T1>
class HLTHGCalLayerClusterIsolationProducer : public edm::stream::EDProducer<> {
  typedef std::vector<T1> T1Collection;
  typedef edm::Ref<T1Collection> T1Ref;
  typedef edm::AssociationMap<edm::OneToValue<std::vector<T1>, float>> T1IsolationMap;

public:
  explicit HLTHGCalLayerClusterIsolationProducer(const edm::ParameterSet&);
  ~HLTHGCalLayerClusterIsolationProducer() override = default;

  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<T1Collection> recoCandidateProducer_;
  const edm::EDGetTokenT<reco::CaloClusterCollection> layerClusterProducer_;
  const edm::EDGetTokenT<double> rhoProducer_;

  const double drMax_;
  const double drVetoEM_;
  const double drVetoHad_;
  const double minEnergyEM_;
  const double minEnergyHad_;
  const double minEtEM_;
  const double minEtHad_;
  const bool useEt_;
  const bool doRhoCorrection_;
  const double rhoMax_;
  const double rhoScale_;
  const std::vector<double> effectiveAreas_;
};

template <typename T1>
HLTHGCalLayerClusterIsolationProducer<T1>::HLTHGCalLayerClusterIsolationProducer(const edm::ParameterSet& config)
    : layerClusterProducer_(
          consumes<reco::CaloClusterCollection>(config.getParameter<edm::InputTag>("layerClusterProducer"))),
      rhoProducer_(consumes<double>(config.getParameter<edm::InputTag>("rhoProducer"))),
      drMax_(config.getParameter<double>("drMax")),
      drVetoEM_(config.getParameter<double>("drVetoEM")),
      drVetoHad_(config.getParameter<double>("drVetoHad")),
      minEnergyEM_(config.getParameter<double>("minEnergyEM")),
      minEnergyHad_(config.getParameter<double>("minEnergyHad")),
      minEtEM_(config.getParameter<double>("minEtEM")),
      minEtHad_(config.getParameter<double>("minEtHad")),
      useEt_(config.getParameter<bool>("useEt")),
      doRhoCorrection_(config.getParameter<bool>("doRhoCorrection")),
      rhoMax_(config.getParameter<double>("rhoMax")),
      rhoScale_(config.getParameter<double>("rhoScale")),
      effectiveAreas_(config.getParameter<std::vector<double>>("effectiveAreas")) {
  if (doRhoCorrection_) {
    if (effectiveAreas_.size() != 2)
      throw cms::Exception("IncompatibleVects")
          << "effectiveAreas should have two elements for em and had components. \n";
  }

  std::string recoCandidateProducerName = "recoCandidateProducer";
  if ((typeid(HLTHGCalLayerClusterIsolationProducer<T1>) ==
       typeid(HLTHGCalLayerClusterIsolationProducer<reco::RecoEcalCandidate>)))
    recoCandidateProducerName = "recoEcalCandidateProducer";

  recoCandidateProducer_ = consumes<T1Collection>(config.getParameter<edm::InputTag>(recoCandidateProducerName));
  produces<T1IsolationMap>();
  produces<T1IsolationMap>("em");
  produces<T1IsolationMap>("had");
}

template <typename T1>
void HLTHGCalLayerClusterIsolationProducer<T1>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  std::string recoCandidateProducerName = "recoCandidateProducer";
  if ((typeid(HLTHGCalLayerClusterIsolationProducer<T1>) ==
       typeid(HLTHGCalLayerClusterIsolationProducer<reco::RecoEcalCandidate>)))
    recoCandidateProducerName = "recoEcalCandidateProducer";

  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(recoCandidateProducerName, edm::InputTag("hltL1SeededRecoEcalCandidatePF"));
  desc.add<edm::InputTag>("layerClusterProducer", edm::InputTag("hltParticleFlowClusterECAL"));
  desc.add<edm::InputTag>("rhoProducer", edm::InputTag("fixedGridRhoFastjetAllCalo"));
  desc.add<bool>("doRhoCorrection", false);
  desc.add<bool>("useEt", false);
  desc.add<double>("rhoMax", 9.9999999E7);
  desc.add<double>("rhoScale", 1.0);
  desc.add<double>("drMax", 0.3);
  desc.add<double>("drVetoEM", 0.0);
  desc.add<double>("drVetoHad", 0.0);
  desc.add<double>("minEnergyEM", 0.0);
  desc.add<double>("minEnergyHad", 0.0);
  desc.add<double>("minEtEM", 0.0);
  desc.add<double>("minEtHad", 0.0);
  desc.add<std::vector<double>>("effectiveAreas", {0.0, 0.0});  // for em and had components
  descriptions.add(defaultModuleLabel<HLTHGCalLayerClusterIsolationProducer<T1>>(), desc);
}

template <typename T1>
void HLTHGCalLayerClusterIsolationProducer<T1>::produce(edm::Event& iEvent, const edm::EventSetup&) {
  edm::Handle<double> rhoHandle;
  double rho = 0.0;
  if (doRhoCorrection_) {
    iEvent.getByToken(rhoProducer_, rhoHandle);
    rho = *(rhoHandle.product());
  }

  rho = std::min(rho, rhoMax_);
  rho = rho * rhoScale_;

  edm::Handle<T1Collection> recoCandHandle;
  edm::Handle<reco::CaloClusterCollection> clusterHandle;

  iEvent.getByToken(recoCandidateProducer_, recoCandHandle);
  iEvent.getByToken(layerClusterProducer_, clusterHandle);

  const std::vector<reco::CaloCluster> layerClusters = *(clusterHandle.product());

  T1IsolationMap recoCandMap(recoCandHandle);
  T1IsolationMap recoCandMapEm(recoCandHandle);
  T1IsolationMap recoCandMapHad(recoCandHandle);

  for (unsigned int iReco = 0; iReco < recoCandHandle->size(); iReco++) {
    T1Ref candRef(recoCandHandle, iReco);

    float sumEm =
        HGCalClusterTools::emEnergyInCone(candRef->eta(),
                                          candRef->phi(),
                                          layerClusters,
                                          drVetoEM_,
                                          drMax_,
                                          minEtEM_,
                                          minEnergyEM_,
                                          useEt_ ? HGCalClusterTools::EType::ET : HGCalClusterTools::EType::ENERGY);

    float sumHad =
        HGCalClusterTools::hadEnergyInCone(candRef->eta(),
                                           candRef->phi(),
                                           layerClusters,
                                           drVetoHad_,
                                           drMax_,
                                           minEtHad_,
                                           minEnergyHad_,
                                           useEt_ ? HGCalClusterTools::EType::ET : HGCalClusterTools::EType::ENERGY);

    if (doRhoCorrection_) {
      sumEm = sumEm - rho * effectiveAreas_.at(0);
      sumHad = sumHad - rho * effectiveAreas_.at(1);
    }

    float sum = sumEm + sumHad;

    recoCandMap.insert(candRef, sum);
    recoCandMapEm.insert(candRef, sumEm);
    recoCandMapHad.insert(candRef, sumHad);
  }

  iEvent.put(std::make_unique<T1IsolationMap>(recoCandMap));
  iEvent.put(std::make_unique<T1IsolationMap>(recoCandMapEm), "em");
  iEvent.put(std::make_unique<T1IsolationMap>(recoCandMapHad), "had");
}

typedef HLTHGCalLayerClusterIsolationProducer<reco::RecoEcalCandidate> EgammaHLTHGCalLayerClusterIsolationProducer;
typedef HLTHGCalLayerClusterIsolationProducer<reco::RecoChargedCandidate> MuonHLTHGCalLayerClusterIsolationProducer;

DEFINE_FWK_MODULE(EgammaHLTHGCalLayerClusterIsolationProducer);
DEFINE_FWK_MODULE(MuonHLTHGCalLayerClusterIsolationProducer);
