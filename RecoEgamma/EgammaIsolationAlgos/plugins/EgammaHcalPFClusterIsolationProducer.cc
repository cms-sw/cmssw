//*****************************************************************************
// File:      EgammaHcalPFClusterIsolationProducer.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matteo Sani
// Institute: UCSD
//*****************************************************************************

#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/ParticleFlowReco/interface/PFCluster.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterFwd.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/TypeID.h"
#include "HLTrigger/HLTcore/interface/defaultModuleLabel.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/HcalPFClusterIsolation.h"

#include <typeinfo>

template <typename T1>
class EgammaHcalPFClusterIsolationProducer : public edm::global::EDProducer<> {
public:
  typedef std::vector<T1> T1Collection;
  typedef edm::Ref<T1Collection> T1Ref;
  explicit EgammaHcalPFClusterIsolationProducer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

private:
  const edm::EDGetTokenT<T1Collection> emObjectProducer_;
  const edm::EDGetTokenT<reco::PFClusterCollection> pfClusterProducerHCAL_;
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
  const double useEt_;
};

template <typename T1>
EgammaHcalPFClusterIsolationProducer<T1>::EgammaHcalPFClusterIsolationProducer(const edm::ParameterSet& config)
    :

      emObjectProducer_(consumes(config.getParameter<edm::InputTag>("candidateProducer"))),
      pfClusterProducerHCAL_(consumes(config.getParameter<edm::InputTag>("pfClusterProducerHCAL"))),
      pfClusterProducerHFEM_(consumes(config.getParameter<edm::InputTag>("pfClusterProducerHFEM"))),
      pfClusterProducerHFHAD_(consumes(config.getParameter<edm::InputTag>("pfClusterProducerHFHAD"))),
      useHF_(config.getParameter<bool>("useHF")),
      drMax_(config.getParameter<double>("drMax")),
      drVetoBarrel_(config.getParameter<double>("drVetoBarrel")),
      drVetoEndcap_(config.getParameter<double>("drVetoEndcap")),
      etaStripBarrel_(config.getParameter<double>("etaStripBarrel")),
      etaStripEndcap_(config.getParameter<double>("etaStripEndcap")),
      energyBarrel_(config.getParameter<double>("energyBarrel")),
      energyEndcap_(config.getParameter<double>("energyEndcap")),
      useEt_(config.getParameter<bool>("useEt")) {
  produces<edm::ValueMap<float>>();
}

template <typename T1>
void EgammaHcalPFClusterIsolationProducer<T1>::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("candidateProducer", edm::InputTag("gedGsfElectrons"));
  desc.add<edm::InputTag>("pfClusterProducerHCAL", edm::InputTag("particleFlowClusterHCAL"));
  desc.ifValue(edm::ParameterDescription<bool>("useHF", false, true),
               true >> (edm::ParameterDescription<edm::InputTag>(
                            "pfClusterProducerHFEM", {"hltParticleFlowClusterHFEM"}, true) and
                        edm::ParameterDescription<edm::InputTag>(
                            "pfClusterProducerHFHAD", {"hltParticleFlowClusterHFHAD"}, true)) or
                   false >> (edm::ParameterDescription<edm::InputTag>("pfClusterProducerHFEM", {""}, true) and
                             edm::ParameterDescription<edm::InputTag>("pfClusterProducerHFHAD", {""}, true)));
  desc.add<double>("drMax", 0.3);
  desc.add<double>("drVetoBarrel", 0.0);
  desc.add<double>("drVetoEndcap", 0.0);
  desc.add<double>("etaStripBarrel", 0.0);
  desc.add<double>("etaStripEndcap", 0.0);
  desc.add<double>("energyBarrel", 0.0);
  desc.add<double>("energyEndcap", 0.0);
  desc.add<bool>("useEt", true);
  descriptions.add(defaultModuleLabel<EgammaHcalPFClusterIsolationProducer<T1>>(), desc);
}

template <typename T1>
void EgammaHcalPFClusterIsolationProducer<T1>::produce(edm::StreamID,
                                                       edm::Event& iEvent,
                                                       const edm::EventSetup&) const {
  auto emObjectHandle = iEvent.getHandle(emObjectProducer_);

  auto isoMap = std::make_unique<edm::ValueMap<float>>();
  edm::ValueMap<float>::Filler filler(*isoMap);
  std::vector<float> retV(emObjectHandle->size(), 0);

  std::vector<edm::Handle<reco::PFClusterCollection>> clusterHandles{iEvent.getHandle(pfClusterProducerHCAL_)};

  if (useHF_) {
    clusterHandles.push_back(iEvent.getHandle(pfClusterProducerHFEM_));
    clusterHandles.push_back(iEvent.getHandle(pfClusterProducerHFHAD_));
  }

  HcalPFClusterIsolation<T1> isoAlgo(
      drMax_, drVetoBarrel_, drVetoEndcap_, etaStripBarrel_, etaStripEndcap_, energyBarrel_, energyEndcap_, useEt_);

  for (unsigned int iReco = 0; iReco < emObjectHandle->size(); iReco++) {
    T1Ref candRef(emObjectHandle, iReco);
    retV[iReco] = isoAlgo.getSum(candRef, clusterHandles);
  }

  filler.insert(emObjectHandle, retV.begin(), retV.end());
  filler.fill();

  iEvent.put(std::move(isoMap));
}

typedef EgammaHcalPFClusterIsolationProducer<reco::GsfElectron> ElectronHcalPFClusterIsolationProducer;
typedef EgammaHcalPFClusterIsolationProducer<reco::Photon> PhotonHcalPFClusterIsolationProducer;

DEFINE_FWK_MODULE(ElectronHcalPFClusterIsolationProducer);
DEFINE_FWK_MODULE(PhotonHcalPFClusterIsolationProducer);
