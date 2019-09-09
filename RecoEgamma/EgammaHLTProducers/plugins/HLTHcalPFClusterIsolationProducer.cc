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

#include "RecoEgamma/EgammaIsolationAlgos/interface/HcalPFClusterIsolation.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"

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
class HLTHcalPFClusterIsolationProducer : public edm::global::EDProducer<> {
  typedef std::vector<T1> T1Collection;
  typedef edm::Ref<T1Collection> T1Ref;
  typedef edm::AssociationMap<edm::OneToValue<std::vector<T1>, float>> T1IsolationMap;

public:
  explicit HLTHcalPFClusterIsolationProducer(const edm::ParameterSet&);
  ~HLTHcalPFClusterIsolationProducer() override;

  void produce(edm::StreamID sid, edm::Event&, const edm::EventSetup&) const override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  edm::EDGetTokenT<T1Collection> recoCandidateProducer_;
  const edm::EDGetTokenT<reco::PFClusterCollection> pfClusterProducerHCAL_;
  const edm::EDGetTokenT<double> rhoProducer_;
  const edm::EDGetTokenT<reco::PFClusterCollection> pfClusterProducerHFEM_;
  const edm::EDGetTokenT<reco::PFClusterCollection> pfClusterProducerHFHAD_;

  const bool useHF_;

  const double drMax_;
  const double drVetoBarrel_;
  const double drVetoEndcap_;
  const double etaStripBarrel_;
  const double etaStripEndcap_;
  const double energyBarrel_;
  const double energyEndcap_;
  const bool useEt_;

  const bool doRhoCorrection_;
  const double rhoMax_;
  const double rhoScale_;
  const std::vector<double> effectiveAreas_;
  const std::vector<double> absEtaLowEdges_;
};

template <typename T1>
HLTHcalPFClusterIsolationProducer<T1>::HLTHcalPFClusterIsolationProducer(const edm::ParameterSet& config)
    : pfClusterProducerHCAL_(
          consumes<reco::PFClusterCollection>(config.getParameter<edm::InputTag>("pfClusterProducerHCAL"))),
      rhoProducer_(consumes<double>(config.getParameter<edm::InputTag>("rhoProducer"))),
      pfClusterProducerHFEM_(
          consumes<reco::PFClusterCollection>(config.getParameter<edm::InputTag>("pfClusterProducerHFEM"))),
      pfClusterProducerHFHAD_(
          consumes<reco::PFClusterCollection>(config.getParameter<edm::InputTag>("pfClusterProducerHFHAD"))),
      useHF_(config.getParameter<bool>("useHF")),
      drMax_(config.getParameter<double>("drMax")),
      drVetoBarrel_(config.getParameter<double>("drVetoBarrel")),
      drVetoEndcap_(config.getParameter<double>("drVetoEndcap")),
      etaStripBarrel_(config.getParameter<double>("etaStripBarrel")),
      etaStripEndcap_(config.getParameter<double>("etaStripEndcap")),
      energyBarrel_(config.getParameter<double>("energyBarrel")),
      energyEndcap_(config.getParameter<double>("energyEndcap")),
      useEt_(config.getParameter<bool>("useEt")),
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
  if ((typeid(HLTHcalPFClusterIsolationProducer<T1>) ==
       typeid(HLTHcalPFClusterIsolationProducer<reco::RecoEcalCandidate>)))
    recoCandidateProducerName = "recoEcalCandidateProducer";
  recoCandidateProducer_ = consumes<T1Collection>(config.getParameter<edm::InputTag>(recoCandidateProducerName));

  produces<T1IsolationMap>();
}

template <typename T1>
HLTHcalPFClusterIsolationProducer<T1>::~HLTHcalPFClusterIsolationProducer() {}

template <typename T1>
void HLTHcalPFClusterIsolationProducer<T1>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  std::string recoCandidateProducerName = "recoCandidateProducer";
  if ((typeid(HLTHcalPFClusterIsolationProducer<T1>) ==
       typeid(HLTHcalPFClusterIsolationProducer<reco::RecoEcalCandidate>)))
    recoCandidateProducerName = "recoEcalCandidateProducer";

  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>(recoCandidateProducerName, edm::InputTag("hltL1SeededRecoEcalCandidatePF"));
  desc.add<edm::InputTag>("pfClusterProducerHCAL", edm::InputTag("hltParticleFlowClusterHCAL"));
  desc.ifValue(
      edm::ParameterDescription<bool>("useHF", false, true),
      true >> (edm::ParameterDescription<edm::InputTag>(
                   "pfClusterProducerHFEM", edm::InputTag("hltParticleFlowClusterHFEM"), true) and
               edm::ParameterDescription<edm::InputTag>(
                   "pfClusterProducerHFHAD", edm::InputTag("hltParticleFlowClusterHFHAD"), true)) or
          false >> (edm::ParameterDescription<edm::InputTag>("pfClusterProducerHFEM", edm::InputTag(""), true) and
                    edm::ParameterDescription<edm::InputTag>("pfClusterProducerHFHAD", edm::InputTag(""), true)));
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
  desc.add<bool>("useEt", true);
  desc.add<std::vector<double>>("effectiveAreas", {0.2, 0.25});   // 2016 post-ichep sinEle default
  desc.add<std::vector<double>>("absEtaLowEdges", {0.0, 1.479});  // Barrel, Endcap
  descriptions.add(defaultModuleLabel<HLTHcalPFClusterIsolationProducer<T1>>(), desc);
}

template <typename T1>
void HLTHcalPFClusterIsolationProducer<T1>::produce(edm::StreamID sid,
                                                    edm::Event& iEvent,
                                                    const edm::EventSetup& iSetup) const {
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

  std::vector<edm::Handle<reco::PFClusterCollection>> clusterHandles;
  edm::Handle<reco::PFClusterCollection> clusterHcalHandle;
  edm::Handle<reco::PFClusterCollection> clusterHfemHandle;
  edm::Handle<reco::PFClusterCollection> clusterHfhadHandle;

  iEvent.getByToken(recoCandidateProducer_, recoCandHandle);
  iEvent.getByToken(pfClusterProducerHCAL_, clusterHcalHandle);
  //const reco::PFClusterCollection* forIsolationHcal = clusterHcalHandle.product();
  clusterHandles.push_back(clusterHcalHandle);

  if (useHF_) {
    iEvent.getByToken(pfClusterProducerHFEM_, clusterHfemHandle);
    clusterHandles.push_back(clusterHfemHandle);
    iEvent.getByToken(pfClusterProducerHFHAD_, clusterHfhadHandle);
    clusterHandles.push_back(clusterHfhadHandle);
  }

  T1IsolationMap recoCandMap(recoCandHandle);
  HcalPFClusterIsolation<T1> isoAlgo(
      drMax_, drVetoBarrel_, drVetoEndcap_, etaStripBarrel_, etaStripEndcap_, energyBarrel_, energyEndcap_, useEt_);

  for (unsigned int iReco = 0; iReco < recoCandHandle->size(); iReco++) {
    T1Ref candRef(recoCandHandle, iReco);

    float sum = isoAlgo.getSum(candRef, clusterHandles);

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

typedef HLTHcalPFClusterIsolationProducer<reco::RecoEcalCandidate> EgammaHLTHcalPFClusterIsolationProducer;
typedef HLTHcalPFClusterIsolationProducer<reco::RecoChargedCandidate> MuonHLTHcalPFClusterIsolationProducer;

DEFINE_FWK_MODULE(EgammaHLTHcalPFClusterIsolationProducer);
DEFINE_FWK_MODULE(MuonHLTHcalPFClusterIsolationProducer);
