// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "DataFormats/JetReco/interface/GenJet.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourInfo.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourInfoMatching.h"
#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/NanoAOD/interface/FlatTable.h"

#include "CommonTools/Utils/interface/StringCutObjectSelector.h"
#include "CommonTools/Utils/interface/StringObjectFunction.h"

class GenJetFlavourTableProducer : public edm::stream::EDProducer<> {
    public:
        explicit GenJetFlavourTableProducer(const edm::ParameterSet &iConfig) :
            name_(iConfig.getParameter<std::string>("name")),
            src_(consumes<std::vector<reco::GenJet> >(iConfig.getParameter<edm::InputTag>("src"))),
            cut_(iConfig.getParameter<std::string>("cut"), true),
            deltaR_(iConfig.getParameter<double>("deltaR")),
            jetFlavourInfosToken_(consumes<reco::JetFlavourInfoMatchingCollection>(iConfig.getParameter<edm::InputTag>("jetFlavourInfos")))
        {
            produces<nanoaod::FlatTable>();
        }

        ~GenJetFlavourTableProducer() override {};

        static void fillDescriptions(edm::ConfigurationDescriptions & descriptions) {
            edm::ParameterSetDescription desc;
            desc.add<edm::InputTag>("src")->setComment("input genJet collection");
            desc.add<edm::InputTag>("jetFlavourInfos")->setComment("input flavour info collection");
            desc.add<std::string>("name")->setComment("name of the genJet FlatTable we are extending with flavour information");
            desc.add<std::string>("cut")->setComment("cut on input genJet collection");
            desc.add<double>("deltaR")->setComment("deltaR to match genjets");
            descriptions.add("genJetFlavourTable", desc);
        }

    private:
        void produce(edm::Event&, edm::EventSetup const&) override ;

        std::string name_;
        edm::EDGetTokenT<std::vector<reco::GenJet> > src_;
        const StringCutObjectSelector<reco::GenJet> cut_;
        const double deltaR_;
        edm::EDGetTokenT<reco::JetFlavourInfoMatchingCollection> jetFlavourInfosToken_;

};

// ------------ method called to produce the data  ------------
void
GenJetFlavourTableProducer::produce(edm::Event& iEvent, const edm::EventSetup& iSetup)
{
    edm::Handle<reco::GenJetCollection> jets;
    iEvent.getByToken(src_, jets);
    
    edm::Handle<reco::JetFlavourInfoMatchingCollection> jetFlavourInfos;
    iEvent.getByToken(jetFlavourInfosToken_, jetFlavourInfos);

    unsigned int ncand = 0;
    std::vector<int> partonFlavour;
    std::vector<uint8_t> hadronFlavour;
    
    for (const reco::GenJet & jet : *jets) {
      if (!cut_(jet)) continue;
      ++ncand;
      bool matched = false;
      for (const reco::JetFlavourInfoMatching & jetFlavourInfoMatching : *jetFlavourInfos) {
        if (deltaR(jet.p4(), jetFlavourInfoMatching.first->p4()) < deltaR_) {
          partonFlavour.push_back(jetFlavourInfoMatching.second.getPartonFlavour());
          hadronFlavour.push_back(jetFlavourInfoMatching.second.getHadronFlavour());
          matched = true;
          break;
        }
      }
      if (!matched) {
        partonFlavour.push_back(0);
        hadronFlavour.push_back(0);
      }
    }

    auto tab  = std::make_unique<nanoaod::FlatTable>(ncand, name_, false, true);
    tab->addColumn<int>("partonFlavour", partonFlavour, "flavour from parton matching", nanoaod::FlatTable::IntColumn);
    tab->addColumn<uint8_t>("hadronFlavour", hadronFlavour, "flavour from hadron ghost clustering", nanoaod::FlatTable::UInt8Column);

    iEvent.put(std::move(tab));
}

#include "FWCore/Framework/interface/MakerMacros.h"
//define this as a plug-in
DEFINE_FWK_MODULE(GenJetFlavourTableProducer);
