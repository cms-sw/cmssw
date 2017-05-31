//
// $Id: GenJetFlavourInfoPreserver.cc,v 1.0 2017/05/18 18:45:45 $
//

/**
  \class    pat::GenJetFlavourInfoPreserver GenJetFlavourInfoPreserver.h "PhysicsTools/JetMCAlgos/interface/GenJetFlavourInfoPreserver.h"
  \brief    Transfers the JetFlavourInfos from the original GenJets to the slimmedGenJets in MiniAOD 
            
  \author   Andrej Saibel
*/


#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "DataFormats/Common/interface/Association.h"
#include "DataFormats/Common/interface/RefToPtr.h"
#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/JetReco/interface/GenJet.h"

#include "DataFormats/JetReco/interface/Jet.h"
#include "DataFormats/JetReco/interface/JetCollection.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourInfo.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourInfoMatching.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/HepMCCandidate/interface/GenParticleFwd.h"


namespace pat {

  class GenJetFlavourInfoPreserver : public edm::stream::EDProducer<> {
  public:
    explicit GenJetFlavourInfoPreserver(const edm::ParameterSet & iConfig);
    virtual ~GenJetFlavourInfoPreserver() { }
    
    virtual void produce(edm::Event & iEvent, const edm::EventSetup & iSetup);
    
  private:
    const edm::EDGetTokenT<edm::View<reco::GenJet> > GenJetsToken_;
    const edm::EDGetTokenT<edm::View<reco::Jet> > slimmedGenJetsToken_;

    const StringCutObjectSelector<reco::GenJet> cut_;
    
    const edm::EDGetTokenT<reco::JetFlavourInfoMatchingCollection> ak4GenJetFlavourInfosToken_;
  };

} // namespace

pat::GenJetFlavourInfoPreserver::GenJetFlavourInfoPreserver(const edm::ParameterSet & iConfig) :
    GenJetsToken_(consumes<edm::View<reco::GenJet> >(iConfig.getParameter<edm::InputTag>("GenJets"))),
    slimmedGenJetsToken_(consumes<edm::View<reco::Jet> >(iConfig.getParameter<edm::InputTag>("slimmedGenJets"))),
    cut_(iConfig.getParameter<std::string>("cut")),
    ak4GenJetFlavourInfosToken_(consumes<reco::JetFlavourInfoMatchingCollection>(iConfig.getParameter<edm::InputTag>("GenJetFlavourInfos")))
{
    produces<reco::JetFlavourInfoMatchingCollection>();
}

void 
pat::GenJetFlavourInfoPreserver::produce(edm::Event & iEvent, const edm::EventSetup & iSetup) {
    using namespace edm;
    using namespace std;

    Handle<View<reco::GenJet> >      GenJets;
    iEvent.getByToken(GenJetsToken_, GenJets);

    Handle<View<reco::Jet> >      slimmedGenJets;
    iEvent.getByToken(slimmedGenJetsToken_, slimmedGenJets);

    Handle<reco::JetFlavourInfoMatchingCollection> ak4GenJetFlavourInfos;
    iEvent.getByToken(ak4GenJetFlavourInfosToken_,ak4GenJetFlavourInfos);

    auto jetFlavourInfos = std::make_unique<reco::JetFlavourInfoMatchingCollection>(reco::JetRefBaseProd(slimmedGenJets));


    uint slimmedId = 0;



    for (View<reco::GenJet>::const_iterator it = GenJets->begin(), ed = GenJets->end(); it != ed; ++it) {
        if (!cut_(*it)) continue;

        for(reco::JetFlavourInfoMatchingCollection::const_iterator JetInfo  = ak4GenJetFlavourInfos->begin(); JetInfo != ak4GenJetFlavourInfos->end();++JetInfo){
                
                if((JetInfo - ak4GenJetFlavourInfos->begin()) < (it - GenJets->begin())) continue;
                else if((JetInfo - ak4GenJetFlavourInfos->begin()) > (it - GenJets->begin())) continue;

                (*jetFlavourInfos)[slimmedGenJets->refAt(slimmedId)] = reco::JetFlavourInfo(JetInfo->second.getbHadrons(), JetInfo->second.getcHadrons(), JetInfo->second.getPartons(), JetInfo->second.getLeptons(), JetInfo->second.getHadronFlavour(), JetInfo->second.getPartonFlavour());

                break;


        }

       slimmedId++; 
    }

    iEvent.put(std::move(jetFlavourInfos));
}

#include "FWCore/Framework/interface/MakerMacros.h"
using namespace pat;
DEFINE_FWK_MODULE(GenJetFlavourInfoPreserver);
