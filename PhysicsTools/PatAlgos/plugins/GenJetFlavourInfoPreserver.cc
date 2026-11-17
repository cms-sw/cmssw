/**
  \class    pat::GenJetFlavourInfoPreserver GenJetFlavourInfoPreserver.h "PhysicsTools/JetMCAlgos/interface/GenJetFlavourInfoPreserver.h"
  \brief    Transfers the JetFlavourInfos from the original GenJets to the slimmedGenJets in MiniAOD 
            Template on the target gen-jet type: GenJetFlavourInfoPreserver (reco::GenJet, unchanged) and
            PATGenJetFlavourInfoPreserver (pat::GenJet, keys the flavour infos to a pat::GenJet collection)
            
  \author   Andrej Saibel, andrej.saibel@cern.ch
*/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/RefToPtr.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "DataFormats/JetMatching/interface/JetFlavourInfo.h"
#include "DataFormats/JetMatching/interface/JetFlavourInfoMatching.h"

#include "DataFormats/JetReco/interface/GenJet.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/PatCandidates/interface/GenJet.h"

namespace pat {

  template <typename T>
  class GenJetFlavourInfoPreserverT : public edm::stream::EDProducer<> {
  public:
    explicit GenJetFlavourInfoPreserverT(const edm::ParameterSet& iConfig);
    ~GenJetFlavourInfoPreserverT() override {}

    void produce(edm::Event& iEvent, const edm::EventSetup& iSetup) override;

  private:
    const edm::EDGetTokenT<edm::View<reco::GenJet> > genJetsToken_;
    const edm::EDGetTokenT<edm::View<reco::Jet> > slimmedGenJetsToken_;

    const edm::EDGetTokenT<reco::JetFlavourInfoMatchingCollection> genJetFlavourInfosToken_;
    const edm::EDGetTokenT<edm::Association<std::vector<T> > > slimmedGenJetAssociationToken_;
  };

  typedef GenJetFlavourInfoPreserverT<reco::GenJet> GenJetFlavourInfoPreserver;
  typedef GenJetFlavourInfoPreserverT<pat::GenJet> PATGenJetFlavourInfoPreserver;

}  // namespace pat

template <typename T>
pat::GenJetFlavourInfoPreserverT<T>::GenJetFlavourInfoPreserverT(const edm::ParameterSet& iConfig)
    : genJetsToken_(consumes<edm::View<reco::GenJet> >(iConfig.getParameter<edm::InputTag>("genJets"))),
      slimmedGenJetsToken_(consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>("slimmedGenJets"))),
      genJetFlavourInfosToken_(
          consumes<reco::JetFlavourInfoMatchingCollection>(iConfig.getParameter<edm::InputTag>("genJetFlavourInfos"))),
      slimmedGenJetAssociationToken_(consumes<edm::Association<std::vector<T> > >(
          iConfig.getParameter<edm::InputTag>("slimmedGenJetAssociation"))) {
  produces<reco::JetFlavourInfoMatchingCollection>();
}

template <typename T>
void pat::GenJetFlavourInfoPreserverT<T>::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  using namespace edm;
  using namespace std;

  Handle<View<reco::GenJet> > genJets;
  iEvent.getByToken(genJetsToken_, genJets);

  Handle<View<reco::Jet> > slimmedGenJets;
  iEvent.getByToken(slimmedGenJetsToken_, slimmedGenJets);

  Handle<reco::JetFlavourInfoMatchingCollection> genJetFlavourInfos;
  iEvent.getByToken(genJetFlavourInfosToken_, genJetFlavourInfos);

  Handle<edm::Association<std::vector<T> > > slimmedGenJetAssociation;
  iEvent.getByToken(slimmedGenJetAssociationToken_, slimmedGenJetAssociation);

  auto jetFlavourInfos = std::make_unique<reco::JetFlavourInfoMatchingCollection>(reco::JetRefBaseProd(slimmedGenJets));
  assert(genJets->size() == genJetFlavourInfos->size());

  edm::Ref<std::vector<T> > slimmedGenJetRef;

  for (unsigned int i = 0; i < genJets->size(); ++i) {
    slimmedGenJetRef = (*slimmedGenJetAssociation)[genJets->refAt(i)];
    if (!slimmedGenJetRef)
      continue;
    (*jetFlavourInfos)[reco::JetBaseRef(slimmedGenJetRef)] = (*genJetFlavourInfos)[i].second;
  }

  iEvent.put(std::move(jetFlavourInfos));
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(GenJetFlavourInfoPreserver);
DEFINE_FWK_MODULE(PATGenJetFlavourInfoPreserver);
