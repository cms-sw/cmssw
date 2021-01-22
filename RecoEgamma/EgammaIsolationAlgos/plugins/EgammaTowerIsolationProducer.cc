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

  const edm::EDPutTokenT<edm::ValueMap<double>> putToken_;

  const double egHcalIsoPtMin_;
  const double egHcalIsoConeSizeOut_;
  const double egHcalIsoConeSizeIn_;
  const signed int egHcalDepth_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(EgammaTowerIsolationProducer);

EgammaTowerIsolationProducer::EgammaTowerIsolationProducer(const edm::ParameterSet& config)
    : emObjectProducer_{consumes(config.getParameter<edm::InputTag>("emObjectProducer"))},
      towerProducer_{consumes(config.getParameter<edm::InputTag>("towerProducer"))},
      putToken_{produces<edm::ValueMap<double>>()},
      egHcalIsoPtMin_{config.getParameter<double>("etMin")},
      egHcalIsoConeSizeOut_{config.getParameter<double>("extRadius")},
      egHcalIsoConeSizeIn_{config.getParameter<double>("intRadius")},
      egHcalDepth_{config.getParameter<int>("Depth")} {}

void EgammaTowerIsolationProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  // Get the  filtered objects
  auto emObjectHandle = iEvent.getHandle(emObjectProducer_);

  // Get the barrel hcal hits
  auto const& towers = iEvent.get(towerProducer_);

  edm::ValueMap<double> isoMap;
  edm::ValueMap<double>::Filler filler(isoMap);
  std::vector<double> retV(emObjectHandle->size(), 0);

  EgammaTowerIsolation myHadIsolation(
      egHcalIsoConeSizeOut_, egHcalIsoConeSizeIn_, egHcalIsoPtMin_, egHcalDepth_, &towers);

  for (size_t i = 0; i < emObjectHandle->size(); ++i) {
    double isoValue = myHadIsolation.getTowerEtSum(&(emObjectHandle->at(i)));
    retV[i] = isoValue;
  }

  filler.insert(emObjectHandle, retV.begin(), retV.end());
  filler.fill();
  iEvent.emplace(putToken_, std::move(isoMap));
}
