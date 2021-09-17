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

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateIsolation.h"

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "RecoEgamma/EgammaIsolationAlgos/interface/EcalPFClusterIsolation.h"
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

#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"

template <typename T1>
class HLTEcalPFClusterIsolationProducer : public edm::stream::EDProducer<> {
  typedef std::vector<T1> T1Collection;
  typedef edm::Ref<T1Collection> T1Ref;
  typedef edm::AssociationMap<edm::OneToValue<std::vector<T1>, float>> T1IsolationMap;

public:
  explicit HLTEcalPFClusterIsolationProducer(const edm::ParameterSet&);
  ~HLTEcalPFClusterIsolationProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  bool computedRVeto(T1Ref candRef, reco::PFClusterRef pfclu);

  edm::EDGetTokenT<T1Collection> recoCandidateProducer_;
  const edm::EDGetTokenT<reco::PFClusterCollection> pfClusterProducer_;
  const edm::EDGetTokenT<double> rhoProducer_;

  const double drMax_;
  const double drVetoBarrel_;
  const double drVetoEndcap_;
  const double etaStripBarrel_;
  const double etaStripEndcap_;
  const double energyBarrel_;
  const double energyEndcap_;

  const bool doRhoCorrection_;
  const double rhoMax_;
  const double rhoScale_;
  const std::vector<double> effectiveAreas_;
  const std::vector<double> absEtaLowEdges_;
};

template <typename T1>
HLTEcalPFClusterIsolationProducer<T1>::HLTEcalPFClusterIsolationProducer(const edm::ParameterSet& config)
    : pfClusterProducer_(consumes<reco::PFClusterCollection>(config.getParameter<edm::InputTag>("pfClusterProducer"))),
      rhoProducer_(consumes<double>(config.getParameter<edm::InputTag>("rhoProducer"))),
      drMax_(config.getParameter<double>("drMax")),
      drVetoBarrel_(config.getParameter<double>("drVetoBarrel")),
      drVetoEndcap_(config.getParameter<double>("drVetoEndcap")),
      etaStripBarrel_(config.getParameter<double>("etaStripBarrel")),
      etaStripEndcap_(config.getParameter<double>("etaStripEndcap")),
      energyBarrel_(config.getParameter<double>("energyBarrel")),
      energyEndcap_(config.getParameter<double>("energyEndcap")),
      doRhoCorrection_(config.getParameter<bool>("doRhoCorrection")),
      rhoMax_(config.getParameter<double>("rhoMax")),
      rhoScale_(config.getParameter<double>("rhoScale")),
      effectiveAreas_(config.getParameter<std::vector<double>>("effectiveAreas")),
      absEtaLowEdges_(config.getParameter<std::vector<double>>("absEtaLowEdges")) {
  if (doRhoCorrection_) {
    if (absEtaLowEdges_.size() != effectiveAreas_.size())
      throw cms::Exception("IncompatibleVects") << "absEtaLowEdges and effectiveAreas should be of the same size. \n";

    if (absEtaLowEdges_.at(0) != 0.0)
      throw cms::Exception("IncompleteCoverage") << "absEtaLowEdges should start from 0. \n";

    for (unsigned int aIt = 0; aIt < absEtaLowEdges_.size() - 1; aIt++) {
      if (!(absEtaLowEdges_.at(aIt) < absEtaLowEdges_.at(aIt + 1)))
        throw cms::Exception("ImproperBinning") << "absEtaLowEdges entries should be in increasing order. \n";
    }
  }

  std::string recoCandidateProducerName = "recoCandidateProducer";
  if ((typeid(HLTEcalPFClusterIsolationProducer<T1>) ==
       typeid(HLTEcalPFClusterIsolationProducer<reco::RecoEcalCandidate>)))
    recoCandidateProducerName = "recoEcalCandidateProducer";

  recoCandidateProducer_ = consumes<T1Collection>(config.getParameter<edm::InputTag>(recoCandidateProducerName));
  produces<T1IsolationMap>();
}

template <typename T1>
HLTEcalPFClusterIsolationProducer<T1>::~HLTEcalPFClusterIsolationProducer() {}

template <typename T1>
void HLTEcalPFClusterIsolationProducer<T1>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  std::string recoCandidateProducerName = "recoCandidateProducer";
  if ((typeid(HLTEcalPFClusterIsolationProducer<T1>) ==
       typeid(HLTEcalPFClusterIsolationProducer<reco::RecoEcalCandidate>)))
    recoCandidateProducerName = "recoEcalCandidateProducer";

  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(recoCandidateProducerName, edm::InputTag("hltL1SeededRecoEcalCandidatePF"));
  desc.add<edm::InputTag>("pfClusterProducer", edm::InputTag("hltParticleFlowClusterECAL"));
  desc.add<edm::InputTag>("rhoProducer", edm::InputTag("fixedGridRhoFastjetAllCalo"));
  desc.add<bool>("doRhoCorrection", false);
  desc.add<double>("rhoMax", 9.9999999E7);
  desc.add<double>("rhoScale", 1.0);
  desc.add<double>("drMax", 0.3);
  desc.add<double>("drVetoBarrel", 0.0);
  desc.add<double>("drVetoEndcap", 0.0);
  desc.add<double>("etaStripBarrel", 0.0);
  desc.add<double>("etaStripEndcap", 0.0);
  desc.add<double>("energyBarrel", 0.0);
  desc.add<double>("energyEndcap", 0.0);
  desc.add<std::vector<double>>("effectiveAreas", {0.29, 0.21});  // 2016 post-ichep sinEle default
  desc.add<std::vector<double>>("absEtaLowEdges", {0.0, 1.479});  // Barrel, Endcap
  descriptions.add(defaultModuleLabel<HLTEcalPFClusterIsolationProducer<T1>>(), desc);
}

template <typename T1>
void HLTEcalPFClusterIsolationProducer<T1>::produce(edm::Event& iEvent, const edm::EventSetup&) {
  edm::Handle<double> rhoHandle;
  double rho = 0.0;
  if (doRhoCorrection_) {
    iEvent.getByToken(rhoProducer_, rhoHandle);
    rho = *(rhoHandle.product());
  }

  if (rho > rhoMax_)
    rho = rhoMax_;

  rho = rho * rhoScale_;

  edm::Handle<T1Collection> recoCandHandle;
  edm::Handle<reco::PFClusterCollection> clusterHandle;

  iEvent.getByToken(recoCandidateProducer_, recoCandHandle);
  iEvent.getByToken(pfClusterProducer_, clusterHandle);

  EcalPFClusterIsolation<T1> isoAlgo(
      drMax_, drVetoBarrel_, drVetoEndcap_, etaStripBarrel_, etaStripEndcap_, energyBarrel_, energyEndcap_);
  T1IsolationMap recoCandMap(recoCandHandle);

  for (unsigned int iReco = 0; iReco < recoCandHandle->size(); iReco++) {
    T1Ref candRef(recoCandHandle, iReco);

    float sum = isoAlgo.getSum(candRef, clusterHandle);

    if (doRhoCorrection_) {
      int iEA = -1;
      auto cEta = std::abs(candRef->eta());
      for (int bIt = absEtaLowEdges_.size() - 1; bIt > -1; bIt--) {
        if (cEta > absEtaLowEdges_.at(bIt)) {
          iEA = bIt;
          break;
        }
      }

      sum = sum - rho * effectiveAreas_.at(iEA);
    }

    recoCandMap.insert(candRef, sum);
  }

  iEvent.put(std::make_unique<T1IsolationMap>(recoCandMap));
}

typedef HLTEcalPFClusterIsolationProducer<reco::RecoEcalCandidate> EgammaHLTEcalPFClusterIsolationProducer;
typedef HLTEcalPFClusterIsolationProducer<reco::RecoChargedCandidate> MuonHLTEcalPFClusterIsolationProducer;

DEFINE_FWK_MODULE(EgammaHLTEcalPFClusterIsolationProducer);
DEFINE_FWK_MODULE(MuonHLTEcalPFClusterIsolationProducer);
