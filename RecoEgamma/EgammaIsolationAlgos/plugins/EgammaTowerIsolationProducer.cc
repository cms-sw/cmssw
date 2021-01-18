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

  void produce(edm::Event&, const edm::EventSetup&) override;

private:
  // ----------member data ---------------------------

  const edm::EDGetTokenT<edm::View<reco::Candidate>> emObjectProducer_;
  const edm::EDGetTokenT<CaloTowerCollection> towerProducer_;

  double egHcalIsoPtMin_;
  double egHcalIsoConeSizeOut_;
  double egHcalIsoConeSizeIn_;
  signed int egHcalDepth_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EgammaTowerIsolationProducer);

EgammaTowerIsolationProducer::EgammaTowerIsolationProducer(const edm::ParameterSet& config)
    : emObjectProducer_{consumes(config.getParameter<edm::InputTag>("emObjectProducer"))}

      ,
      towerProducer_{consumes(config.getParameter<edm::InputTag>("towerProducer"))} {
  egHcalIsoPtMin_ = config.getParameter<double>("etMin");
  egHcalIsoConeSizeIn_ = config.getParameter<double>("intRadius");
  egHcalIsoConeSizeOut_ = config.getParameter<double>("extRadius");
  egHcalDepth_ = config.getParameter<int>("Depth");

  //register your products
  produces<edm::ValueMap<double>>();
}

// ------------ method called to produce the data  ------------
void EgammaTowerIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get the  filtered objects
  auto emObjectHandle = iEvent.getHandle(emObjectProducer_);

  // Get the barrel hcal hits
  auto const& towers = iEvent.get(towerProducer_);

  auto isoMap = std::make_unique<edm::ValueMap<double>>();
  edm::ValueMap<double>::Filler filler(*isoMap);
  std::vector<double> retV(emObjectHandle->size(), 0);

  EgammaTowerIsolation myHadIsolation(
      egHcalIsoConeSizeOut_, egHcalIsoConeSizeIn_, egHcalIsoPtMin_, egHcalDepth_, &towers);

  for (size_t i = 0; i < emObjectHandle->size(); ++i) {
    double isoValue = myHadIsolation.getTowerEtSum(&(emObjectHandle->at(i)));
    retV[i] = isoValue;
  }

  filler.insert(emObjectHandle, retV.begin(), retV.end());
  filler.fill();
  iEvent.put(std::move(isoMap));
}
