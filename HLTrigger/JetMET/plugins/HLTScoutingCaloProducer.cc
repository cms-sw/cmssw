// -*- C++ -*-
//
// Package:    HLTrigger/JetMET
// Class:      HLTScoutingCaloProducer
// 
/**\class HLTScoutingCaloProducer HLTScoutingCaloProducer.cc HLTrigger/JetMET/plugins/HLTScoutingCaloProducer.cc

Description: Producer for ScoutingCaloJets from reco::CaloJet objects

*/
//
// Original Author:  Dustin James Anderson
//         Created:  Fri, 12 Jun 2015 15:49:20 GMT
//
//

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/METReco/interface/CaloMETCollection.h"
#include "DataFormats/METReco/interface/CaloMET.h"
#include "DataFormats/BTauReco/interface/JetTag.h"

#include "DataFormats/Scouting/interface/ScoutingCaloJet.h"
#include "DataFormats/Scouting/interface/ScoutingVertex.h"

#include "DataFormats/Math/interface/deltaR.h"

class HLTScoutingCaloProducer : public edm::global::EDProducer<> {
    public:
        explicit HLTScoutingCaloProducer(const edm::ParameterSet&);
        ~HLTScoutingCaloProducer();

        static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

    private:
        virtual void produce(edm::StreamID sid, edm::Event & iEvent, edm::EventSetup const & setup) const override final;

        const edm::EDGetTokenT<reco::CaloJetCollection> caloJetCollection_;
        const edm::EDGetTokenT<reco::JetTagCollection> caloJetTagCollection_;
        const edm::EDGetTokenT<reco::VertexCollection> vertexCollection_;
        const edm::EDGetTokenT<reco::CaloMETCollection> metCollection_;
        const edm::EDGetTokenT<double> rho_;

        const double caloJetPtCut;
        const double caloJetEtaCut;

        const bool doMet;
        const bool doJetTags;
};

//
// constructors and destructor
//
HLTScoutingCaloProducer::HLTScoutingCaloProducer(const edm::ParameterSet& iConfig):
    caloJetCollection_(consumes<reco::CaloJetCollection>(iConfig.getParameter<edm::InputTag>("caloJetCollection"))),
    caloJetTagCollection_(consumes<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("caloJetTagCollection"))),
    vertexCollection_(consumes<reco::VertexCollection>(iConfig.getParameter<edm::InputTag>("vertexCollection"))),
    metCollection_(consumes<reco::CaloMETCollection>(iConfig.getParameter<edm::InputTag>("metCollection"))),
    rho_(consumes<double>(iConfig.getParameter<edm::InputTag>("rho"))),
    caloJetPtCut(iConfig.getParameter<double>("caloJetPtCut")),
    caloJetEtaCut(iConfig.getParameter<double>("caloJetEtaCut")),
    doMet(iConfig.getParameter<bool>("doMet")),
    doJetTags(iConfig.getParameter<bool>("doJetTags"))
{
    //register products
    produces<ScoutingCaloJetCollection>();
    produces<ScoutingVertexCollection>();
    produces<double>("rho");
    produces<double>("caloMetPt");
    produces<double>("caloMetPhi");
}

HLTScoutingCaloProducer::~HLTScoutingCaloProducer()
{ }

// ------------ method called to produce the data  ------------
    void
HLTScoutingCaloProducer::produce(edm::StreamID sid, edm::Event & iEvent, edm::EventSetup const & setup) const
{
    using namespace edm;

    //get calo jets
    Handle<reco::CaloJetCollection> caloJetCollection;
    std::auto_ptr<ScoutingCaloJetCollection> outCaloJets(new ScoutingCaloJetCollection());
    if(iEvent.getByToken(caloJetCollection_, caloJetCollection)){
        //get jet tags
        Handle<reco::JetTagCollection> caloJetTagCollection;
        bool haveJetTags = false;
        if(doJetTags && iEvent.getByToken(caloJetTagCollection_, caloJetTagCollection)){
            haveJetTags = true;
        }

        for(auto &jet : *caloJetCollection){
            if(jet.pt() > caloJetPtCut && fabs(jet.eta()) < caloJetEtaCut){
                //find the jet tag corresponding to the jet
                float tagValue = -20;
                float minDR2 = 0.01;
                if(haveJetTags){
                    for(auto &tag : *caloJetTagCollection){
                        float dR2 = reco::deltaR2(jet, *(tag.first));
                        if(dR2 < minDR2){
                            minDR2 = dR2;
                            tagValue = tag.second;
                        }
                    }
                }
                outCaloJets->emplace_back(
                        jet.pt(), jet.eta(), jet.phi(), jet.mass(),
                        jet.jetArea(), jet.maxEInEmTowers(), jet.maxEInHadTowers(),
                        jet.hadEnergyInHB(), jet.hadEnergyInHE(), jet.hadEnergyInHF(),
                        jet.emEnergyInEB(), jet.emEnergyInEE(), jet.emEnergyInHF(),
                        jet.towersArea(), 0.0, tagValue
                        );
            }
        }
    }

    //get vertices
    Handle<reco::VertexCollection> vertexCollection;
    std::auto_ptr<ScoutingVertexCollection> outVertices(new ScoutingVertexCollection());
    if(iEvent.getByToken(vertexCollection_, vertexCollection)){
        //produce vertices (only if present; otherwise return an empty collection)
        for(auto &vtx : *vertexCollection){
            outVertices->emplace_back(
                        vtx.x(), vtx.y(), vtx.z(), vtx.zError()
                        );
        }
    }

    //get rho
    Handle<double>rho;
    std::auto_ptr<double> outRho(new double(-999));
    if(iEvent.getByToken(rho_, rho)){
        outRho.reset(new double(*rho));
    }

    //get MET 
    Handle<reco::CaloMETCollection> metCollection;
    std::auto_ptr<double> outMetPt(new double(-999));
    std::auto_ptr<double> outMetPhi(new double(-999));
    if(doMet && iEvent.getByToken(metCollection_, metCollection)){
        outMetPt.reset(new double(metCollection->front().pt()));
        outMetPhi.reset(new double(metCollection->front().phi()));
    }

    //put output
    iEvent.put(outCaloJets);
    iEvent.put(outVertices);
    iEvent.put(outRho, "rho");
    iEvent.put(outMetPt, "caloMetPt");
    iEvent.put(outMetPhi, "caloMetPhi");
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void
HLTScoutingCaloProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("caloJetCollection",edm::InputTag("hltAK4CaloJets"));
    desc.add<edm::InputTag>("caloJetTagCollection",edm::InputTag("hltCombinedSecondaryVertexBJetTagsCalo"));
    desc.add<edm::InputTag>("vertexCollection", edm::InputTag("hltPixelVertices"));
    desc.add<edm::InputTag>("metCollection", edm::InputTag("hltMet"));
    desc.add<edm::InputTag>("rho", edm::InputTag("hltFixedGridRhoFastjetAllCalo"));
    desc.add<double>("caloJetPtCut", 20.0);
    desc.add<double>("caloJetEtaCut", 3.0);
    desc.add<bool>("doMet", true);
    desc.add<bool>("doJetTags", false);
    descriptions.add("hltScoutingCaloProducer", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(HLTScoutingCaloProducer);
