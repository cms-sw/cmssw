// -*- C++ -*-
//
// Package:    UserCode/L1TriggerDPG
// Class:      L1PhaseIITreeProducer
// 
/**\class L1PhaseIITreeProducer L1PhaseIITreeProducer.cc UserCode/L1TriggerDPG/src/L1PhaseIITreeProducer.cc

Description: Produce L1 Extra tree

Implementation:

*/
//
// Original Author:  Alex Tapper
//         Created:  
// $Id: L1PhaseIITreeProducer.cc,v 1.5 2013/01/06 21:55:55 jbrooke Exp $
//
//


// system include files
#include <memory>

// framework
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

// data formats
#include "DataFormats/L1Trigger/interface/L1EmParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EmParticle.h"
#include "DataFormats/L1Trigger/interface/L1JetParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1JetParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticleFwd.h"
#include "DataFormats/L1Trigger/interface/L1EtMissParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkMuonParticleFwd.h"

#include "DataFormats/L1TrackTrigger/interface/L1TkGlbMuonParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkGlbMuonParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkPrimaryVertex.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEtMissParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEtMissParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkEmParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkElectronParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkElectronParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkJetParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkTauParticleFwd.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkHTMissParticle.h"
#include "DataFormats/L1TrackTrigger/interface/L1TkHTMissParticleFwd.h"

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1TVertex/interface/Vertex.h"

#include "DataFormats/JetReco/interface/PFJet.h"


// ROOT output stuff
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "TTree.h"

#include "L1Trigger/L1TNtuples/interface/L1AnalysisPhaseII.h"

//
// class declaration
//

class L1PhaseIITreeProducer : public edm::EDAnalyzer {
        public:
                explicit L1PhaseIITreeProducer(const edm::ParameterSet&);
                ~L1PhaseIITreeProducer();


        private:
                virtual void beginJob(void) ;
                virtual void analyze(const edm::Event&, const edm::EventSetup&);
                virtual void endJob();

        public:

                L1Analysis::L1AnalysisPhaseII* l1Extra;
                L1Analysis::L1AnalysisPhaseIIDataFormat * l1ExtraData;

        private:

                unsigned maxL1Extra_;

                // output file
                edm::Service<TFileService> fs_;

                // tree
                TTree * tree_;

                std::vector< edm::EDGetTokenT<l1t::TauBxCollection> > tauTokens_;
                edm::EDGetTokenT<l1t::JetBxCollection> jetToken_;
                edm::EDGetTokenT<l1t::EtSumBxCollection> sumToken_;
                edm::EDGetTokenT<l1t::MuonBxCollection> muonToken_;

                std::vector<edm::EDGetTokenT<l1t::EGammaBxCollection> > egToken_;
                std::vector<edm::EDGetTokenT<l1t::L1TkElectronParticleCollection> > tkEGToken_;
                std::vector<edm::EDGetTokenT<l1t::L1TkElectronParticleCollection> > tkEGLooseToken_;
                std::vector< edm::EDGetTokenT<l1t::L1TkEmParticleCollection> > tkEMToken_;

                edm::EDGetTokenT<l1t::L1TkMuonParticleCollection> TkMuonToken_;
                edm::EDGetTokenT<l1t::L1TkGlbMuonParticleCollection> TkGlbMuonToken_;

                edm::EDGetTokenT<l1t::L1TkTauParticleCollection> tkTauToken_;
                edm::EDGetTokenT<l1t::L1TkJetParticleCollection> tkTrackerJetToken_;
                edm::EDGetTokenT<l1t::L1TkEtMissParticleCollection> tkMetToken_;

                std::vector< edm::EDGetTokenT<l1t::L1TkHTMissParticleCollection> > tkMhtToken_;

                edm::EDGetTokenT<l1t::L1TkJetParticleCollection> tkCaloJetToken_;

                edm::EDGetTokenT<std::vector<reco::PFJet>> ak4L1PF_;

                edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> muonKalman_;

                edm::EDGetTokenT<std::vector<reco::PFMET> > l1PFMet_;

                edm::EDGetTokenT<float> z0PuppiToken_;
                edm::EDGetTokenT<l1t::VertexCollection> l1vertextdrToken_;
                edm::EDGetTokenT<l1t::VertexCollection> l1verticesToken_;
                edm::EDGetTokenT<l1t::L1TkPrimaryVertexCollection> l1TkPrimaryVertexToken_;




};

L1PhaseIITreeProducer::L1PhaseIITreeProducer(const edm::ParameterSet& iConfig){
        jetToken_ = consumes<l1t::JetBxCollection>(iConfig.getUntrackedParameter<edm::InputTag>("jetToken"));
        sumToken_ = consumes<l1t::EtSumBxCollection>(iConfig.getUntrackedParameter<edm::InputTag>("sumToken"));
        muonToken_ = consumes<l1t::MuonBxCollection>(iConfig.getUntrackedParameter<edm::InputTag>("muonToken"));

        const auto& egammatokens=iConfig.getParameter<std::vector<edm::InputTag>>("egTokens");
        for (const auto& egtoken: egammatokens) {
                egToken_.push_back(consumes<l1t::EGammaBxCollection>(egtoken));
        }

        const auto& taus = iConfig.getUntrackedParameter<std::vector<edm::InputTag>>("tauTokens");
        for (const auto& tau: taus) {
                tauTokens_.push_back(consumes<l1t::TauBxCollection>(tau));
        }

        const auto& eletokens=iConfig.getParameter<std::vector<edm::InputTag>>("tkEGTokens");
        for (const auto& eletoken: eletokens) {
                tkEGToken_.push_back(consumes<l1t::L1TkElectronParticleCollection>(eletoken));
        }
        const auto& eleloosetokens=iConfig.getParameter<std::vector<edm::InputTag>>("tkEGLooseTokens");
        for (const auto& eleloosetoken: eleloosetokens) {
                tkEGLooseToken_.push_back(consumes<l1t::L1TkElectronParticleCollection>(eleloosetoken));
        }
        const auto& photokens=iConfig.getParameter<std::vector<edm::InputTag>>("tkEMTokens");
        for (const auto& photoken: photokens) {
                tkEMToken_.push_back(consumes<l1t::L1TkEmParticleCollection>(photoken));
        }

        TkMuonToken_ = consumes<l1t::L1TkMuonParticleCollection>(iConfig.getParameter<edm::InputTag>("TkMuonToken"));

        TkGlbMuonToken_ = consumes<l1t::L1TkGlbMuonParticleCollection>(iConfig.getParameter<edm::InputTag>("TkGlbMuonToken"));
        tkTauToken_ = consumes<l1t::L1TkTauParticleCollection>(iConfig.getParameter<edm::InputTag>("tkTauToken"));

        tkTrackerJetToken_ = consumes<l1t::L1TkJetParticleCollection>(iConfig.getParameter<edm::InputTag>("tkTrackerJetToken"));
        tkMetToken_ = consumes<l1t::L1TkEtMissParticleCollection>(iConfig.getParameter<edm::InputTag>("tkMetToken"));
        //tkMhtToken_ = consumes<l1t::L1TkHTMissParticleCollection>(iConfig.getParameter<edm::InputTag>("tkMhtToken"));

        const auto& mhttokens=iConfig.getParameter<std::vector<edm::InputTag>>("tkMhtTokens");
        for (const auto& mhttoken: mhttokens) {
                tkMhtToken_.push_back(consumes<l1t::L1TkHTMissParticleCollection>(mhttoken));
        }


        tkCaloJetToken_ = consumes<l1t::L1TkJetParticleCollection>(iConfig.getParameter<edm::InputTag>("tkCaloJetToken"));

        ak4L1PF_ = consumes<std::vector<reco::PFJet> > (iConfig.getParameter<edm::InputTag>("ak4L1PF"));

        muonKalman_ = consumes<l1t::RegionalMuonCandBxCollection> (iConfig.getParameter<edm::InputTag>("muonKalman"));

        l1PFMet_  = consumes< std::vector<reco::PFMET> > (iConfig.getParameter<edm::InputTag>("l1PFMet"));

        z0PuppiToken_ = consumes< float > (iConfig.getParameter<edm::InputTag>("zoPuppi"));
        l1vertextdrToken_ = consumes< l1t::VertexCollection> (iConfig.getParameter<edm::InputTag>("l1vertextdr"));
        l1verticesToken_  = consumes< l1t::VertexCollection> (iConfig.getParameter<edm::InputTag>("l1vertices"));
        l1TkPrimaryVertexToken_ = consumes< l1t::L1TkPrimaryVertexCollection> (iConfig.getParameter<edm::InputTag>("l1TkPrimaryVertex"));

        maxL1Extra_ = iConfig.getParameter<unsigned int>("maxL1Extra");

        l1Extra     = new L1Analysis::L1AnalysisPhaseII();
        l1ExtraData = l1Extra->getData();

        // set up output
        tree_=fs_->make<TTree>("L1PhaseIITree", "L1PhaseIITree");
        tree_->Branch("L1PhaseII", "L1Analysis::L1AnalysisPhaseIIDataFormat", &l1ExtraData, 32000, 3);

}


L1PhaseIITreeProducer::~L1PhaseIITreeProducer()
{

        // do anything here that needs to be done at desctruction time
        // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called to for each event  ------------
        void
L1PhaseIITreeProducer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{


        l1Extra->Reset();

        edm::Handle<l1t::MuonBxCollection> muon;
        edm::Handle<l1t::L1TkGlbMuonParticleCollection> TkGlbMuon;
        edm::Handle<l1t::L1TkMuonParticleCollection> TkMuon;

        iEvent.getByToken(muonToken_, muon);
        iEvent.getByToken(TkGlbMuonToken_,TkGlbMuon);
        iEvent.getByToken(TkMuonToken_,TkMuon);

        edm::Handle<l1t::RegionalMuonCandBxCollection> muonsKalman;
        iEvent.getByToken(muonKalman_,muonsKalman);

        edm::Handle<l1t::L1TkTauParticleCollection> tkTau;
        iEvent.getByToken(tkTauToken_, tkTau);

        edm::Handle<l1t::JetBxCollection> jet;
        edm::Handle<l1t::EtSumBxCollection> sums;
        iEvent.getByToken(jetToken_,  jet);
        iEvent.getByToken(sumToken_, sums);

        edm::Handle<l1t::L1TkJetParticleCollection> tkTrackerJet;
        edm::Handle<l1t::L1TkJetParticleCollection> tkCaloJet;
        edm::Handle<l1t::L1TkEtMissParticleCollection> tkMets;
        //edm::Handle<l1t::L1TkHTMissParticleCollection> tkMhts;

        iEvent.getByToken(tkTrackerJetToken_, tkTrackerJet);
        iEvent.getByToken(tkCaloJetToken_, tkCaloJet);
        iEvent.getByToken(tkMetToken_, tkMets);
        //iEvent.getByToken(tkMhtToken_, tkMhts);

        edm::Handle<std::vector<reco::PFJet>> ak4L1PFs;
        iEvent.getByToken(ak4L1PF_,ak4L1PFs);
        edm::Handle< std::vector<reco::PFMET> > l1PFMet;
        iEvent.getByToken(l1PFMet_, l1PFMet);

        // now also fill vertices 

        edm::Handle<float> z0Puppi;
        iEvent.getByToken(z0PuppiToken_,z0Puppi);
        float Z0=*z0Puppi;

        edm::Handle<std::vector<l1t::Vertex> > l1vertextdr;
        edm::Handle<std::vector<l1t::Vertex> > l1vertices;
        iEvent.getByToken(l1vertextdrToken_,l1vertextdr);
        iEvent.getByToken(l1verticesToken_,l1vertices);

        edm::Handle<std::vector<l1t::L1TkPrimaryVertex> > l1TkPrimaryVertex;
        iEvent.getByToken(l1TkPrimaryVertexToken_,l1TkPrimaryVertex);

        float vertexTDRZ0=-999; 
        if(l1vertextdr->size()>0) vertexTDRZ0=l1vertextdr->at(0).z0();

        if(l1vertices.isValid() && l1TkPrimaryVertex.isValid() &&  l1vertices->size()>0 && l1TkPrimaryVertex->size()>0){
              l1Extra->SetVertices(Z0,vertexTDRZ0,l1vertices,l1TkPrimaryVertex);
        }
        else {
                edm::LogWarning("MissingProduct") << "One of the L1TVertex collections is not valid " << std::endl;
                std::cout<<"Getting the vertices!"<<std::endl;
                std::cout<<Z0<<"   "<<l1vertextdr->size() <<"  "<< l1vertices->size() <<"   "<<  l1TkPrimaryVertex->size()<<std::endl;
        }

        if (jet.isValid()){ 
                l1Extra->SetJet(jet, maxL1Extra_);
        } else {
                edm::LogWarning("MissingProduct") << "L1Upgrade Jets not found. Branch will not be filled" << std::endl;
        }

        if (sums.isValid()){ 
                l1Extra->SetSum(sums, maxL1Extra_);  
        } else {
                edm::LogWarning("MissingProduct") << "L1Upgrade EtSums not found. Branch will not be filled" << std::endl;
        }

        if (muon.isValid()){ 
                l1Extra->SetMuon(muon, maxL1Extra_);
        } else {
                edm::LogWarning("MissingProduct") << "L1Upgrade Muons not found. Branch will not be filled" << std::endl;
        }

        if (muonsKalman.isValid()){
                l1Extra->SetMuonKF(muonsKalman, maxL1Extra_);
        } else {
                edm::LogWarning("MissingProduct") << "L1Upgrade KBMTF Muons not found. Branch will not be filled" << std::endl;
        }

        for (auto & tautoken: tauTokens_){
                // keeping the format in the Run2 upgrade producer for this for consistency, even if it is a bit weird
                edm::Handle<l1t::TauBxCollection> tau;
                iEvent.getByToken(tautoken,  tau);
                if (tau.isValid()){ 
                        l1Extra->SetTau(tau, maxL1Extra_);
                } else {
                        edm::LogWarning("MissingProduct") << "L1Upgrade Tau not found. Branch will not be filled" << std::endl;
                }
        }

        for (auto & eletoken: tkEGToken_){
                edm::Handle<l1t::L1TkElectronParticleCollection> tkEG;
                iEvent.getByToken(eletoken, tkEG);

                if (tkEG.isValid()){
                        l1Extra->SetTkEG(tkEG, maxL1Extra_);
                } else {
                        edm::LogWarning("MissingProduct") << "L1PhaseII TkEG not found. Branch will not be filled" << std::endl;
                }
        }

        for (auto & eleloosetoken: tkEGLooseToken_){
                edm::Handle<l1t::L1TkElectronParticleCollection> tkEGLoose;
                iEvent.getByToken(eleloosetoken, tkEGLoose);

                if (tkEGLoose.isValid()){
                        l1Extra->SetTkEGLoose(tkEGLoose, maxL1Extra_);
                } else {
                        edm::LogWarning("MissingProduct") << "L1PhaseII tkEGLoose not found. Branch will not be filled" << std::endl;
                }
        }



        for (auto & egtoken: egToken_){
                edm::Handle<l1t::EGammaBxCollection> eg;
                iEvent.getByToken(egtoken,   eg);
                if (eg.isValid()){
                        l1Extra->SetEG(eg, maxL1Extra_);
                } else {
                        edm::LogWarning("MissingProduct") << "L1PhaseII Barrel EG not found. Branch will not be filled" << std::endl;
                }
        }

        for (auto & photoken: tkEMToken_){
                edm::Handle<l1t::L1TkEmParticleCollection> tkEM;
                iEvent.getByToken(photoken, tkEM);

                if (tkEM.isValid()){
                        l1Extra->SetTkEM(tkEM, maxL1Extra_);
                } else {
                        edm::LogWarning("MissingProduct") << "L1PhaseII  TkEM not found. Branch will not be filled" << std::endl;
                }
        }

        if (tkTau.isValid()){
                l1Extra->SetTkTau(tkTau, maxL1Extra_);
        } else {
                edm::LogWarning("MissingProduct") << "L1PhaseII TkTau not found. Branch will not be filled" << std::endl;
        }

        if (tkTrackerJet.isValid()){
                l1Extra->SetTkJet(tkTrackerJet, maxL1Extra_);
        } else {
                edm::LogWarning("MissingProduct") << "L1PhaseII tkTrackerJets not found. Branch will not be filled" << std::endl;
        }

        if (tkCaloJet.isValid()){
                l1Extra->SetTkCaloJet(tkCaloJet, maxL1Extra_);
        } else {
                edm::LogWarning("MissingProduct") << "L1PhaseII TkCaloJets not found. Branch will not be filled" << std::endl;
        }

        if (TkGlbMuon.isValid()){
                l1Extra->SetTkGlbMuon(TkGlbMuon, maxL1Extra_);
        } else {
                edm::LogWarning("MissingProduct") << "L1PhaseII TkGlbMuons not found. Branch will not be filled" << std::endl;
        }
        if (TkMuon.isValid()){
                l1Extra->SetTkMuon(TkMuon, maxL1Extra_);
        } else {
                edm::LogWarning("MissingProduct") << "L1PhaseII TkMuons not found. Branch will not be filled" << std::endl;
        }

        if (tkMets.isValid()){
                l1Extra->SetTkMET(tkMets);
        } else {
                edm::LogWarning("MissingProduct") << "L1PhaseII TkMET not found. Branch will not be filled" << std::endl;
        }

        for (auto & tkmhttoken: tkMhtToken_){
                edm::Handle<l1t::L1TkHTMissParticleCollection> tkMhts;
                iEvent.getByToken(tkmhttoken, tkMhts); 

                if (tkMhts.isValid()){
                        l1Extra->SetTkMHT(tkMhts);
                } else {
                        edm::LogWarning("MissingProduct") << "L1PhaseII TkMHT not found. Branch will not be filled" << std::endl;
                }
        }

        if (ak4L1PFs.isValid()){
                l1Extra->SetPFJet(ak4L1PFs, maxL1Extra_);
        } else {
                edm::LogWarning("MissingProduct") << "L1PhaseII PFJets not found. Branch will not be filled" << std::endl;
        }

        if(l1PFMet.isValid()){
                l1Extra->SetL1METPF(l1PFMet);
        } else{
                edm::LogWarning("MissingProduct") << "L1PFMet missing"<<std::endl;
        }

        tree_->Fill();

}

// ------------ method called once each job just before starting event loop  ------------
        void 
L1PhaseIITreeProducer::beginJob(void)
{
}

// ------------ method called once each job just after ending the event loop  ------------
void 
L1PhaseIITreeProducer::endJob() {
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1PhaseIITreeProducer);
