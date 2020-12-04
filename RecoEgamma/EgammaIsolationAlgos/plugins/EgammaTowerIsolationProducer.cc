//*****************************************************************************
// File:      EgammaTowerIsolationProducer.cc
// ----------------------------------------------------------------------------
// OrigAuth:  Matthias Mozer
// Institute: IIHE-VUB
//=============================================================================
//*****************************************************************************

#include "DataFormats/CaloTowers/interface/CaloTowerCollection.h"
#include "DataFormats/Candidate/interface/CandAssociation.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "RecoEgamma/EgammaIsolationAlgos/interface/EgammaTowerIsolation.h"

class EgammaTowerIsolationProducer : public edm::stream::EDProducer<> {
public:
  explicit EgammaTowerIsolationProducer(const edm::ParameterSet&);
  ~EgammaTowerIsolationProducer() override;

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------

  edm::InputTag emObjectProducer_;
  edm::InputTag towerProducer_;

  double egHcalIsoPtMin_;
  double egHcalIsoConeSizeOut_;
  double egHcalIsoConeSizeIn_;
  signed int egHcalDepth_;

  edm::ParameterSet conf_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EgammaTowerIsolationProducer);

EgammaTowerIsolationProducer::EgammaTowerIsolationProducer(const edm::ParameterSet& config) : conf_(config) {
  // use configuration file to setup input/output collection names
  emObjectProducer_ = conf_.getParameter<edm::InputTag>("emObjectProducer");

  towerProducer_ = conf_.getParameter<edm::InputTag>("towerProducer");

  egHcalIsoPtMin_ = conf_.getParameter<double>("etMin");
  egHcalIsoConeSizeIn_ = conf_.getParameter<double>("intRadius");
  egHcalIsoConeSizeOut_ = conf_.getParameter<double>("extRadius");
  egHcalDepth_ = conf_.getParameter<int>("Depth");

  //register your products
  produces<edm::ValueMap<double>>();
}

EgammaTowerIsolationProducer::~EgammaTowerIsolationProducer() {}

//
// member functions
//

// ------------ method called to produce the data  ------------
void EgammaTowerIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get the  filtered objects
  edm::Handle<edm::View<reco::Candidate>> emObjectHandle;
  iEvent.getByLabel(emObjectProducer_, emObjectHandle);

  // Get the barrel hcal hits
  edm::Handle<CaloTowerCollection> towerHandle;
  iEvent.getByLabel(towerProducer_, towerHandle);
  const CaloTowerCollection* towers = towerHandle.product();

  auto isoMap = std::make_unique<edm::ValueMap<double>>();
  edm::ValueMap<double>::Filler filler(*isoMap);
  std::vector<double> retV(emObjectHandle->size(), 0);

  EgammaTowerIsolation myHadIsolation(
      egHcalIsoConeSizeOut_, egHcalIsoConeSizeIn_, egHcalIsoPtMin_, egHcalDepth_, towers);

  for (size_t i = 0; i < emObjectHandle->size(); ++i) {
    double isoValue = myHadIsolation.getTowerEtSum(&(emObjectHandle->at(i)));
    retV[i] = isoValue;
  }

  filler.insert(emObjectHandle, retV.begin(), retV.end());
  filler.fill();
  iEvent.put(std::move(isoMap));
}
