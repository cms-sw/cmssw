#ifndef SusyDQM_H
#define SusyDQM_H

#include <string>
#include <vector>
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <sstream>
#include <math.h>

//ROOT includes
#include "TLorentzVector.h"

//Framework includes
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include "DataFormats/Common/interface/Handle.h"

//DQM includes
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQM/Physics/interface/Hemisphere.hh"
#include "DQM/Physics/interface/Davismt2.h"

//Reco includes
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/MuonEnergy.h"
#include "DataFormats/MuonReco/interface/MuonIsolation.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "DataFormats/JetReco/interface/PFJetCollection.h"
#include "DataFormats/METReco/interface/PFMETCollection.h"
#include "DataFormats/METReco/interface/PFMET.h"
#include "RecoEgamma/EgammaTools/interface/ConversionTools.h"

//GEN includes
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/JetReco/interface/GenJetCollection.h"
#include "SimDataFormats/JetMatching/interface/JetFlavourMatching.h"
#include "SimDataFormats/JetMatching/interface/JetFlavour.h"

//PAT includes
#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/PatCandidates/interface/Muon.h"
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/PatCandidates/interface/Jet.h"
#include "DataFormats/PatCandidates/interface/MET.h"

//Math includes
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"
#include "DataFormats/Math/interface/LorentzVector.h"

using namespace std;

template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
class SusyDQM : public DQMEDAnalyzer {

    public:
        SusyDQM(const edm::ParameterSet& ps);
        virtual ~SusyDQM();

    protected:
        virtual void analyze(edm::Event const& e, edm::EventSetup const& eSetup);
        virtual bool setupEvent(const edm::Event& evt);

        virtual void selectObjects(edm::Event const& e);
        virtual void fillHadronic(edm::Event const& e);
        virtual void fillLeptonic(edm::Event const& e);
        virtual void fillPhotonic(edm::Event const& e);

    private:
        void bookHistograms(DQMStore::IBooker& booker, edm::Run const&, edm::EventSetup const&) override;
        virtual bool goodElectron(const Ele*);
        virtual bool goodMuon(const Mu*);
        virtual bool goodPhoton(const edm::Event& evt, const Pho*);
        virtual bool goodJet(const Jet*);

        std::vector<math::XYZTLorentzVector> createHemispheresRazor(std::vector<math::XYZTLorentzVector> jets);
        double calcDeltaHT(std::vector<math::XYZTLorentzVector> jets);
        double calcMR(TLorentzVector ja, TLorentzVector jb);
        double calcRsq(double MR, TLorentzVector ja, TLorentzVector jb, edm::Handle<std::vector<Met> > inputMet);
        double calcMT2(float testMass, bool massive, vector<TLorentzVector> jets, TLorentzVector MET, int hemi_seed, int hemi_association);
        const reco::GenParticle* findFirstMotherWithDifferentID(const reco::GenParticle*);

        //Objects from config
        edm::EDGetTokenT<std::vector<Mu> > muons_;
        edm::EDGetTokenT<std::vector<Ele> > electrons_;
        edm::EDGetTokenT<std::vector<Pho> > photons_;
        edm::EDGetTokenT<std::vector<Jet> > jets_;
        edm::EDGetTokenT<std::vector<Met> > met_;
        edm::EDGetTokenT<reco::VertexCollection> vertex_;
        edm::EDGetTokenT<reco::ConversionCollection> conversions_;
        edm::EDGetTokenT<reco::BeamSpot> beamSpot_;
        edm::EDGetTokenT<double> fixedGridRhoFastjetAll_;
        edm::EDGetTokenT<reco::JetTagCollection> jetTagCollection_;
        edm::EDGetTokenT<edm::View<reco::GenParticle> > genParticles_;
        edm::EDGetTokenT<reco::GenJetCollection> genJets_;
        edm::EDGetTokenT<reco::JetFlavourMatchingCollection> jetFlavorMatch_;

        edm::Handle<std::vector<Mu> > muons;
        edm::Handle<std::vector<Ele> > electrons;
        edm::Handle<std::vector<Pho> > photons;
        edm::Handle<std::vector<Jet> > jets;
        edm::Handle<std::vector<Met> > met;
        edm::Handle<reco::VertexCollection> vertex;
        edm::Handle<reco::ConversionCollection> conversions;
        edm::Handle<reco::BeamSpot> beamSpot;
        edm::Handle<double> fixedGridRhoFastjetAll;
        edm::Handle<reco::JetTagCollection> jetTagCollection;
        edm::Handle<edm::View<reco::GenParticle> > genParticles;
        edm::Handle<reco::GenJetCollection> genJets;
        edm::Handle<reco::JetFlavourMatchingCollection> jetFlavorMatch;

        //Lorentz vectors for selected physics objects
        vector<const Mu*> goodMuons;
        vector<const Ele*> goodElectrons;
        vector<const Pho*> goodPhotons;
        vector<const Jet*> goodJets;

        //Cut values from config
        double jetPtCut;
        double jetEtaCut;
        double jetCSVV2Cut;

        double elePtCut;
        double eleEtaCut;
        int eleMaxMissingHits;
        double eleDEtaInCutBarrel;
        double eleDPhiInCutBarrel;
        double eleSigmaIetaIetaCutBarrel;
        double eleHoverECutBarrel;
        double eleD0CutBarrel;
        double eleDZCutBarrel;
        double eleOneOverEMinusOneOverPCutBarrel;
        double eleRelIsoCutBarrel;
        double eleDEtaInCutEndcap;
        double eleDPhiInCutEndcap;
        double eleSigmaIetaIetaCutEndcap;
        double eleHoverECutEndcap;
        double eleD0CutEndcap;
        double eleDZCutEndcap;
        double eleOneOverEMinusOneOverPCutEndcap;
        double eleRelIsoCutEndcap;
 
        double muPtCut;
        double muEtaCut;
        double muRelIsoCut;

        double phoPtCut;
        double phoEtaCut;
        double phoHoverECutBarrel;
        double phoSigmaIetaIetaCutBarrel;
        double phoChHadIsoCutBarrel;
        double phoNeuHadIsoCutBarrel;
        double phoNeuHadIsoSlopeBarrel;
        double phoPhotIsoCutBarrel;
        double phoPhotIsoSlopeBarrel;
        double phoHoverECutEndcap;
        double phoSigmaIetaIetaCutEndcap;
        double phoChHadIsoCutEndcap;
        double phoNeuHadIsoCutEndcap;
        double phoNeuHadIsoSlopeEndcap;
        double phoPhotIsoCutEndcap;
        double phoPhotIsoSlopeEndcap;

        bool useGen;
        
        //DQM Histograms

        //Leptonic -- histograms EXCLUSIVE in lepton number
        MonitorElement* leadingElePt_HT250; 
        MonitorElement* leadingEleEta_HT250; 
        MonitorElement* leadingElePhi_HT250; 
        MonitorElement* leadingMuPt_HT250; 
        MonitorElement* leadingMuEta_HT250; 
        MonitorElement* leadingMuPhi_HT250; 
        MonitorElement* mTLepMET_singleLepton; 
        MonitorElement* HT_singleLepton60; 
        MonitorElement* missingEt_singleLepton60; 
        MonitorElement* nJets_singleLepton60; 

        MonitorElement* muonEfficiencyVsPt_denominator;
        MonitorElement* muonEfficiencyVsPt_numerator;
        MonitorElement* electronEfficiencyVsPt_denominator;
        MonitorElement* electronEfficiencyVsPt_numerator;

        MonitorElement* osDiMuonMass; 
        MonitorElement* osDiElectronMass; 
        MonitorElement* ssDiMuonMass; 
        MonitorElement* ssDiElectronMass; 
        MonitorElement* osOfDiLeptonMass; 
        MonitorElement* ssOfDiLeptonMass; 

        MonitorElement* HT_threeOrMoreLeptons; 
        MonitorElement* missingEt_threeOrMoreLeptons;

        //Photonic -- histograms INCLUSIVE in photon number
        MonitorElement* leadingPhoPt_HT250;
        MonitorElement* leadingPhoEta_HT250;
        MonitorElement* leadingPhoPhi_HT250;
        MonitorElement* HT_photon80;
        MonitorElement* missingEt_photon80;
        MonitorElement* nJets_photon80;
        MonitorElement* deltaPhiPhoMET_photon80;
        MonitorElement* diPhotonMass_HT250;
        MonitorElement* diPhotonMass_MET100;
        MonitorElement* MR_Rsq0p02Diphoton;
        MonitorElement* Rsq_MR100Diphoton;

        //Hadronic -- histograms exclude events with selected leptons
        MonitorElement* leadingJetPt_pT80;
        MonitorElement* leadingJetEta_pT80;
        MonitorElement* leadingJetPhi_pT80;
        MonitorElement* deltaPhiJJ_2Jets80; //inclusive in number of jets
        MonitorElement* missingEt_HT250;
        MonitorElement* missingEt_1BTaggedJet;
        MonitorElement* metPhi_MET150;
        MonitorElement* HT_MET150;
        MonitorElement* MHT;
        MonitorElement* missingEtOverMHT;
        MonitorElement* MHTOverHT;
        MonitorElement* nJets;
        MonitorElement* nBTaggedJetsCSVV2M_HT250;
        MonitorElement* nJets_HT250;
        MonitorElement* nJets_MET150;
        MonitorElement* deltaPhiJetMET_jet80;
        MonitorElement* minDeltaPhiJetMET;
        MonitorElement* deltaPhiJJ_dijet80Exclusive; //exclusive dijet events
        MonitorElement* MR_RSq0p15; //razor variable histogram
        MonitorElement* RSq_MR300; //razor variable histogram
        MonitorElement* alphaT;
        MonitorElement* mT2;

        MonitorElement* fractionOfGoodJetsVsEta_numerator;
        MonitorElement* fractionOfGoodJetsVsPhi_numerator;
        MonitorElement* fractionOfGoodJetsVsEta_denominator;
        MonitorElement* fractionOfGoodJetsVsPhi_denominator;
        MonitorElement* csvV2MediumEfficiencyVsPt_numerator;
        MonitorElement* csvV2MediumEfficiencyVsPt_denominator;
};

//constructor
template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
SusyDQM<Mu, Ele, Pho, Jet, Met>::SusyDQM(const edm::ParameterSet& pset) {
    edm::LogInfo("SusyDQM") << "Beginning SUSY DQM sequence\n";

    muons_ = consumes<std::vector<Mu> >(
            pset.getParameter<edm::InputTag>("muonCollection"));
    electrons_ = consumes<std::vector<Ele> >(
            pset.getParameter<edm::InputTag>("electronCollection"));
    photons_ = consumes<std::vector<Pho> >(
            pset.getParameter<edm::InputTag>("photonCollection"));
    jets_ = consumes<std::vector<Jet> >(
            pset.getParameter<edm::InputTag>("jetCollection"));
    met_ = consumes<std::vector<Met> >(
            pset.getParameter<edm::InputTag>("metCollection"));
    vertex_ = consumes<reco::VertexCollection>(
            pset.getParameter<edm::InputTag>("vertexCollection"));
    conversions_ = consumes<reco::ConversionCollection>(
            pset.getParameter<edm::InputTag>("conversions"));
    beamSpot_ = consumes<reco::BeamSpot>(
            pset.getParameter<edm::InputTag>("beamSpot"));
    fixedGridRhoFastjetAll_ = consumes<double>(
            pset.getParameter<edm::InputTag>("fixedGridRhoFastjetAll"));
    jetTagCollection_ = consumes<reco::JetTagCollection>(pset.getParameter<edm::InputTag>("jetTagCollection"));
    genParticles_ = consumes<edm::View<reco::GenParticle> >(pset.getParameter<edm::InputTag>("genParticles"));
    genJets_ = consumes<reco::GenJetCollection>(pset.getParameter<edm::InputTag>("genJets"));
    jetFlavorMatch_ = consumes<reco::JetFlavourMatchingCollection>(pset.getParameter<edm::InputTag>("jetFlavourAssociation"));

    jetPtCut = pset.getParameter<double>("jetPtCut");
    jetEtaCut = pset.getParameter<double>("jetEtaCut");
    jetCSVV2Cut = pset.getParameter<double>("csvv2Cut");

    muPtCut = pset.getParameter<double>("muPtCut");
    muEtaCut = pset.getParameter<double>("muEtaCut");
    muRelIsoCut = pset.getParameter<double>("muRelIsoCut");

    elePtCut = pset.getParameter<double>("elePtCut");
    eleEtaCut = pset.getParameter<double>("eleEtaCut");
    eleMaxMissingHits = pset.getParameter<int>("eleMaxMissingHits");
    eleDEtaInCutBarrel = pset.getParameter<double>("eleDEtaInCutBarrel");
    eleDPhiInCutBarrel = pset.getParameter<double>("eleDPhiInCutBarrel");
    eleSigmaIetaIetaCutBarrel = pset.getParameter<double>("eleSigmaIetaIetaCutBarrel");
    eleHoverECutBarrel = pset.getParameter<double>("eleHoverECutBarrel");
    eleD0CutBarrel = pset.getParameter<double>("eleD0CutBarrel");
    eleDZCutBarrel = pset.getParameter<double>("eleDZCutBarrel");
    eleOneOverEMinusOneOverPCutBarrel = pset.getParameter<double>("eleOneOverEMinusOneOverPCutBarrel");
    eleRelIsoCutBarrel = pset.getParameter<double>("eleRelIsoCutBarrel");
    eleDEtaInCutEndcap = pset.getParameter<double>("eleDEtaInCutEndcap");
    eleDPhiInCutEndcap = pset.getParameter<double>("eleDPhiInCutEndcap");
    eleSigmaIetaIetaCutEndcap = pset.getParameter<double>("eleSigmaIetaIetaCutEndcap");
    eleHoverECutEndcap = pset.getParameter<double>("eleHoverECutEndcap");
    eleD0CutEndcap = pset.getParameter<double>("eleD0CutEndcap");
    eleDZCutEndcap = pset.getParameter<double>("eleDZCutEndcap");
    eleOneOverEMinusOneOverPCutEndcap = pset.getParameter<double>("eleOneOverEMinusOneOverPCutEndcap");
    eleRelIsoCutEndcap = pset.getParameter<double>("eleRelIsoCutEndcap");

    phoPtCut = pset.getParameter<double>("phoPtCut");
    phoEtaCut = pset.getParameter<double>("phoEtaCut");
    phoHoverECutBarrel = pset.getParameter<double>("phoHoverECutBarrel");
    phoSigmaIetaIetaCutBarrel = pset.getParameter<double>("phoSigmaIetaIetaCutBarrel");
    phoChHadIsoCutBarrel = pset.getParameter<double>("phoChHadIsoCutBarrel");
    phoNeuHadIsoCutBarrel = pset.getParameter<double>("phoNeuHadIsoCutBarrel");
    phoNeuHadIsoSlopeBarrel = pset.getParameter<double>("phoNeuHadIsoSlopeBarrel");
    phoPhotIsoCutBarrel = pset.getParameter<double>("phoPhotIsoCutBarrel");
    phoPhotIsoSlopeBarrel = pset.getParameter<double>("phoPhotIsoSlopeBarrel");
    phoHoverECutEndcap = pset.getParameter<double>("phoHoverECutEndcap");
    phoSigmaIetaIetaCutEndcap = pset.getParameter<double>("phoSigmaIetaIetaCutEndcap");
    phoChHadIsoCutEndcap = pset.getParameter<double>("phoChHadIsoCutEndcap");
    phoNeuHadIsoCutEndcap = pset.getParameter<double>("phoNeuHadIsoCutEndcap");
    phoNeuHadIsoSlopeEndcap = pset.getParameter<double>("phoNeuHadIsoSlopeEndcap");
    phoPhotIsoCutEndcap = pset.getParameter<double>("phoPhotIsoCutEndcap");
    phoPhotIsoSlopeEndcap = pset.getParameter<double>("phoPhotIsoSlopeEndcap");

    useGen = pset.getParameter<bool>("useGen");
}

//destructor
template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
SusyDQM<Mu, Ele, Pho, Jet, Met>::~SusyDQM() {
    edm::LogInfo("SusyDQM") << "Deleting SusyDQM object\n";
}

//book histograms
template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
void SusyDQM<Mu, Ele, Pho, Jet, Met>::bookHistograms(DQMStore::IBooker& booker, edm::Run const& run, edm::EventSetup const& es){
    booker.cd();

    //book lepton histograms
    booker.setCurrentFolder("Physics/Susy/SingleLepton");
    leadingElePt_HT250 = booker.book1D("leadingElePt_HT250", "Leading electron p_{T}, HT > 250 (GeV); p_{T} (GeV)", 50, 20.0 , 2000);
    leadingEleEta_HT250 = booker.book1D("leadingEleEta_HT250", "Leading electron #eta, HT > 250; #eta", 20, -2.5, 2.5);
    leadingElePhi_HT250 = booker.book1D("leadingElePhi_HT250", "Leading electron #phi, HT > 250; #phi", 20, -3.1415, 3.1415);
    leadingMuPt_HT250 = booker.book1D("leadingMuPt_HT250", "Leading muon p_{T}, HT > 250 (GeV); p_{T} (GeV)", 50, 20.0, 2000);
    leadingMuEta_HT250 = booker.book1D("leadingMuEta_HT250", "Leading muon #eta, HT > 250; #eta", 20, -2.4, 2.4);
    leadingMuPhi_HT250 = booker.book1D("leadingMuPhi_HT250", "Leading muon #phi, HT > 250; #phi", 20, -3.1415, 3.1415);
    mTLepMET_singleLepton = booker.book1D("mTLepMET_singleLepton", "m_{T} of lepton + MET in single lepton events (GeV); m_{T} (GeV)", 50, 0., 2000);
    HT_singleLepton60 = booker.book1D("HT_singleLepton60", "HT, single lepton with p_{T} > 60 (GeV); HT (GeV)", 50, 0., 4000);
    missingEt_singleLepton60 = booker.book1D("missingEt_singleLepton60", "MET, single lepton with p_{T} > 60 (GeV); MET (GeV)", 50, 0., 2000);
    nJets_singleLepton60 = booker.book1D("nJets_singleLepton60", "n_{jets} p_{T} > 40 GeV , single lepton with p_{T} > 60; n_{jets}", 20, 0, 20);

    muonEfficiencyVsPt_numerator = booker.book1D("muonEfficiencyVsPt_numerator", "Loose muon ID efficiency vs p_{T};gen muon p_{T}", 20, 10, 1000); 
    muonEfficiencyVsPt_denominator = booker.book1D("muonEfficiencyVsPt_denominator", "Loose muon ID efficiency vs p_{T}; gen muon p_{T}", 20, 10, 1000); 
    electronEfficiencyVsPt_numerator = booker.book1D("electronEfficiencyVsPt_numerator", "Loose electron ID efficiency vs p_{T}; gen electron p_{T}", 20, 10, 1000); 
    electronEfficiencyVsPt_denominator = booker.book1D("electronEfficiencyVsPt_denominator", "Loose electron ID efficiency vs p_{T}; gen electron p_{T}", 20, 10, 1000); 

    booker.setCurrentFolder("Physics/Susy/DiLepton");
    osDiMuonMass = booker.book1D("osDiMuonMass", "OS di-muon mass (GeV); mass (GeV)", 50, 0., 500);
    osDiElectronMass = booker.book1D("osDiElectronMass", "OS di-electron mass (GeV); mass (GeV)", 50, 0., 500);
    ssDiMuonMass = booker.book1D("ssDiMuonMass", "SS di-muon mass (GeV); mass (GeV)", 50, 0., 500);
    ssDiElectronMass = booker.book1D("ssDiElectronMass", "SS di-electron mass (GeV); mass (GeV)", 50, 0., 500);
    osOfDiLeptonMass = booker.book1D("osOfDiLeptonMass", "OS opposite-flavor dilepton mass (GeV); mass (GeV)", 50, 0., 500);
    ssOfDiLeptonMass = booker.book1D("ssOfDiLeptonMass", "SS opposite-flavor dilepton mass (GeV); mass (GeV)", 50, 0., 500);

    booker.setCurrentFolder("Physics/Susy/MultiLepton");
    HT_threeOrMoreLeptons = booker.book1D("HT_threeOrMoreLeptons", "HT in events with 3+ leptons (GeV); HT (GeV)", 50, 0, 4000);
    missingEt_threeOrMoreLeptons = booker.book1D("missingEt_threeOrMoreLeptons", "MET in events with 3+ leptons (GeV); MET (GeV)", 50, 0., 2000);

    //book photon histograms
    booker.setCurrentFolder("Physics/Susy/SinglePhoton");
    leadingPhoPt_HT250 = booker.book1D("leadingPhoPt_HT250", "Leading photon p_{T}, HT > 250 (GeV); p_{T} (GeV)", 50, 0., 2000);
    leadingPhoEta_HT250 = booker.book1D("leadingPhoEta_HT250", "Leading photon #eta, HT > 250; #eta", 20, -2.5, 2.5);
    leadingPhoPhi_HT250 = booker.book1D("leadingPhoPhi_HT250", "Leading photon #phi, HT > 250; #phi", 20, -3.1415, 3.1415);
    deltaPhiPhoMET_photon80 = booker.book1D("deltaPhiPhoMET_photon80", "#Delta #phi_{MET, #gamma}, photon p_{T} > 80 GeV; #Delta #phi", 20, 0., 3.1415);
    HT_photon80 = booker.book1D("HT_photon80", "HT, photon p_{T} > 80 (GeV); HT (GeV)", 50, 0., 4000);
    missingEt_photon80 = booker.book1D("missingEt_photon80", "MET, photon p_{T} > 80 (GeV); MET (GeV)", 50, 0., 2000);
    nJets_photon80 = booker.book1D("nJets_photon80", "n_{jets} p_{T} > 40 GeV, photon p_{T} > 80; n_{jets}", 20, 0, 20);

    booker.setCurrentFolder("Physics/Susy/DiPhoton");
    diPhotonMass_HT250 = booker.book1D("diPhotonMass_HT250", "Diphoton mass, HT > 250 (GeV); mass (GeV)", 50, 0., 500);
    diPhotonMass_MET100 = booker.book1D("diPhotonMass_MET100", "Diphoton mass, MET > 100 (GeV); mass (GeV)", 50, 0., 500);
    MR_Rsq0p02Diphoton = booker.book1D("MR_diphoton", "MR in diphoton events (GeV); MR (GeV)", 50, 0., 800);
    Rsq_MR100Diphoton = booker.book1D("Rsq_diphoton", "R^{2} in diphoton events (GeV); R^{2}", 50, 0., 0.6);

    //book jet histograms
    booker.setCurrentFolder("Physics/Susy/Hadronic");
    leadingJetPt_pT80 = booker.book1D("leadingJetPt_pT80", "Leading jet p_{T} (GeV); p_{T} (GeV)", 50, 80, 2000);
    leadingJetEta_pT80 = booker.book1D("leadingJetEta_pT80", "Leading jet #eta; #eta", 20, -3.0, 3.0);
    leadingJetPhi_pT80 = booker.book1D("leadingJetPhi_pT80", "Leading jet #phi; #phi", 20, -3.1415, 3.1415);
    deltaPhiJJ_2Jets80 = booker.book1D("deltaPhiJJ_2Jets80", "#Delta #phi_{two leading jets}; #Delta #phi", 20, 0., 3.1415);
    missingEt_HT250 = booker.book1D("missingEt_HT250", "MET (HT > 250) (GeV); MET (GeV)", 50, 0., 2000);
    missingEt_1BTaggedJet = booker.book1D("missingEt_1BTaggedJet", "MET (>= 1 b-jet in event); MET (GeV)", 50, 0., 2000);
    metPhi_MET150 = booker.book1D("metPhi_MET150", "MET #phi, MET > 150 GeV; #phi", 20, -3.1415, 3.1415);
    HT_MET150 = booker.book1D("hT_MET150", "HT (MET > 150) (GeV); HT (GeV)", 50, 0., 4000);
    MHT = booker.book1D("mHT", "MHT (GeV); MHT (GeV)", 50, 0., 2000);
    missingEtOverMHT = booker.book1D("missingEtOverMHT", "MET/MHT; MET/MHT", 50, 0., 10.0);
    MHTOverHT = booker.book1D("mHTOverHT", "MHT/HT; MHT/HT", 50, 0., 5.0);
    nJets = booker.book1D("nJets", "n_{jets} p_{T} > 40 GeV; n_{jets}", 20, 0, 20);
    nBTaggedJetsCSVV2M_HT250 = booker.book1D("nBTaggedJetsCSVV2M_HT250", "n_{jets} p_{T} > 40 GeV, CSVV2 medium b-tag, HT > 250 GeV; n_{jets}", 10, 0, 10);
    nJets_HT250 = booker.book1D("nJets_HT250", "n_{jets} p_{T} > 40 GeV (HT > 250); n_{jets}", 20, 0, 20);
    nJets_MET150 = booker.book1D("nJets_MET150", "n_{jets} p_{T} > 40 GeV (MET > 150); n_{jets}", 20, 0, 20);
    deltaPhiJetMET_jet80 = booker.book1D("deltaPhiJetMET_jet80", "#Delta #phi_{MET, leading jet}; #Delta #phi", 20, 0, 3.1415);
    minDeltaPhiJetMET = booker.book1D("minDeltaPhiJetMET", "Min #Delta #phi_{jet, MET}; Min #Delta #phi", 20, 0., 3.14159);
    deltaPhiJJ_dijet80Exclusive = booker.book1D("deltaPhiJJ_dijet80Exclusive", "#Delta #phi_{two leading jets}, dijet events; #Delta #phi", 20, 0., 3.1415);
    MR_RSq0p15 = booker.book1D("MR_RSq0p15", "M_{R}, R^{2} > 0.15 (GeV); M_{R} (GeV)", 50, 0., 4000);
    RSq_MR300 = booker.book1D("RSq_MR300", "R^{2}, M_{R} > 300 GeV; R^{2}", 50, 0., 2.0);
    alphaT = booker.book1D("alphaT", "#alpha_{T}; #alpha_{T}", 50, 0.0, 3.0);
    mT2 = booker.book1D("mT2", "M_{T2} (GeV); M_{T2} (GeV)", 50, 0.0, 2000);

    fractionOfGoodJetsVsEta_numerator = booker.book1D("fractionOfGoodJetsVsEta_numerator", "Fraction of jets passing loose ID; #eta", 20, -3.0, 3.0);
    fractionOfGoodJetsVsEta_denominator = booker.book1D("fractionOfGoodJetsVsEta_denominator", "Fraction of jets passing loose ID; #eta", 20, -3.0, 3.0);
    fractionOfGoodJetsVsPhi_numerator = booker.book1D("fractionOfGoodJetsVsPhi_numerator", "Fraction of jets passing loose ID; #phi", 20, -3.0, 3.0);
    fractionOfGoodJetsVsPhi_denominator = booker.book1D("fractionOfGoodJetsVsPhi_denominator", "Fraction of jets passing loose ID; #phi", 20, -3.0, 3.0);
    csvV2MediumEfficiencyVsPt_numerator = booker.book1D("csvV2MediumEfficiencyVsPt_numerator", "CSVV2M b-tag efficiency vs jet p_{T}; jet p_{T}", 40, 40, 2000);
    csvV2MediumEfficiencyVsPt_denominator = booker.book1D("csvV2MediumEfficiencyVsPt_denominator", "CSVV2M b-tag efficiency vs jet p_{T}; jet p_{T}", 40, 40, 2000);

    booker.cd();
}

//muon ID: loose PF ID + loose isolation requirement
template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
bool SusyDQM<Mu, Ele, Pho, Jet, Met>::goodMuon(const Mu* mu) {
    //baseline kinematic cuts
    if(!muon::isLooseMuon(*mu)) return false;
    if(mu->pt() < muPtCut) return false;
    if(fabs(mu->eta()) > muEtaCut) return false;
    
    //get muon isolation computed in a cone dR < 0.4, applying the deltaBeta correction
    //I = [sumChargedHadronPt+ max(0.,sumNeutralHadronPt+sumPhotonPt-0.5sumPUPt]/pt
    double muRelIsolation = (mu->pfIsolationR04().sumChargedHadronPt + max(0., mu->pfIsolationR04().sumNeutralHadronEt + mu->pfIsolationR04().sumPhotonEt - 0.5*mu->pfIsolationR04().sumPUPt))/mu->pt();
    if(muRelIsolation > muRelIsoCut) return false;

    return true;
}

//electron ID: loose cut-based ID
template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
bool SusyDQM<Mu, Ele, Pho, Jet, Met>::goodElectron(const Ele* ele) {
    //baseline cuts
    if(ele->pt() < elePtCut) return false;
    if(fabs(ele->superCluster()->eta()) > eleEtaCut) return false;
    //conversion veto
    if(beamSpot.isValid() && conversions.isValid()){
        if(ConversionTools::hasMatchedConversion(*ele, conversions, beamSpot->position())) return false;
    }
    else{
        edm::LogError("SusyDQM") << "Unable to check for conversions!  BeamSpot or Conversions object is invalid!\n";
        return false;
    }
    //missing hits
    int missHits = ele->gsfTrack()->hitPattern().numberOfHits(reco::HitPattern::MISSING_INNER_HITS);
    if(missHits > eleMaxMissingHits) return false;

    //loose electron ID cuts
    if(fabs(ele->superCluster()->eta()) < 1.479){ //barrel cuts
        //dEtaIn cut
        if(fabs(ele->deltaEtaSuperClusterTrackAtVtx()) > eleDEtaInCutBarrel) return false;
        //dPhiIn cut
        if(fabs(ele->deltaPhiSuperClusterTrackAtVtx()) > eleDPhiInCutBarrel) return false;
        //SigmaIetaIeta cut
        if(ele->full5x5_sigmaIetaIeta() > eleSigmaIetaIetaCutBarrel) return false;
        //H/E cut
        if(ele->hcalOverEcal() > eleHoverECutBarrel) return false;
        //d0 cut
        if(fabs(-ele->gsfTrack()->dxy(vertex->front().position())) > eleD0CutBarrel) return false;
        //dZ cut
        if(fabs(ele->gsfTrack()->dz(vertex->front().position())) > eleDZCutBarrel) return false;
        //1/E-1/p cut
        double ooEmooP = 0;
        if(ele->ecalEnergy() < 1e-6 || !std::isfinite(ele->ecalEnergy())){ //check if electron energy is 0 or infinite
            ooEmooP = 1e30;
        }
        else{
            ooEmooP = fabs(1.0/ele->ecalEnergy() - ele->eSuperClusterOverP()/ele->ecalEnergy());
        }
        if(ooEmooP > eleOneOverEMinusOneOverPCutBarrel) return false;
        //relative isolation cut (compute isolation with delta beta method)
        double eleRelIsolation = (ele->pfIsolationVariables().sumChargedHadronPt + max(0.,ele->pfIsolationVariables().sumNeutralHadronEt + ele->pfIsolationVariables().sumPhotonEt - 0.5*ele->pfIsolationVariables().sumPUPt)) / ele->pt();
        if(eleRelIsolation > eleRelIsoCutBarrel) return false;
    }
    else{ //endcap cuts
        //dEtaIn cut
        if(fabs(ele->deltaEtaSuperClusterTrackAtVtx()) > eleDEtaInCutEndcap) return false;
        //dPhiIn cut
        if(fabs(ele->deltaPhiSuperClusterTrackAtVtx()) > eleDPhiInCutEndcap) return false;
        //SigmaIetaIeta cut
        if(ele->full5x5_sigmaIetaIeta() > eleSigmaIetaIetaCutEndcap) return false;
        //H/E cut
        if(ele->hcalOverEcal() > eleHoverECutEndcap) return false;
        //d0 cut
        if(fabs(-ele->gsfTrack()->dxy(vertex->front().position())) > eleD0CutEndcap) return false;
        //dZ cut
        if(fabs(ele->gsfTrack()->dz(vertex->front().position())) > eleDZCutEndcap) return false;
        //1/E-1/p cut
        double ooEmooP = 0;
        if(ele->ecalEnergy() < 1e-6 || !std::isfinite(ele->ecalEnergy())){ //check if electron energy is 0 or infinite
            ooEmooP = 1e30;
        }
        else{
            ooEmooP = fabs(1.0/ele->ecalEnergy() - ele->eSuperClusterOverP()/ele->ecalEnergy());
        }
        if(ooEmooP > eleOneOverEMinusOneOverPCutEndcap) return false;
        //relative isolation cut (compute isolation with delta beta method)
        double eleRelIsolation = (ele->pfIsolationVariables().sumChargedHadronPt + max(0.,ele->pfIsolationVariables().sumNeutralHadronEt + ele->pfIsolationVariables().sumPhotonEt - 0.5*ele->pfIsolationVariables().sumPUPt)) / ele->pt();
        if(eleRelIsolation > eleRelIsoCutEndcap) return false;
    }
    return true;
}

//loose cut-based photon ID
//NOTE: the isolation values are not quite correct due to an issue with footprint removal!  The ID will not work quite as advertised, but it is a reasonable temporary solution.
template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
bool SusyDQM<Mu, Ele, Pho, Jet, Met>::goodPhoton(const edm::Event& evt, const Pho* pho) {

    //baseline kinematic cuts
    if(pho->pt() < phoPtCut) return false;
    if(fabs(pho->superCluster()->eta()) > phoEtaCut) return false;

    //electron veto -- use the presence of a pixel seed 
    if(pho->hasPixelSeed()) return false;

    //get photon effective areas for isolation
    double effAreaChHad = 0;
    double effAreaNeuHad = 0;
    double effAreaPhot = 0;
    if(fabs(pho->superCluster()->eta()) < 1.0){
        effAreaChHad = 0.0130;
        effAreaNeuHad = 0.0056;
        effAreaPhot = 0.0896;
    }
    else if(fabs(pho->superCluster()->eta()) < 1.479){
        effAreaChHad = 0.0096;
        effAreaNeuHad = 0.0107;
        effAreaPhot = 0.0762;
    }
    else if(fabs(pho->superCluster()->eta()) < 2.0){
        effAreaChHad = 0.0107;
        effAreaNeuHad = 0.0019;
        effAreaPhot = 0.0383;
    }
    else if(fabs(pho->superCluster()->eta()) < 2.2){
        effAreaChHad = 0.0077;
        effAreaNeuHad = 0.0011;
        effAreaPhot = 0.0534;
    }
    else if(fabs(pho->superCluster()->eta()) < 2.3){
        effAreaChHad = 0.0088;
        effAreaNeuHad = 0.0077;
        effAreaPhot = 0.0846;
    }
    else if(fabs(pho->superCluster()->eta()) < 2.4){
        effAreaChHad = 0.0065;
        effAreaNeuHad = 0.0178;
        effAreaPhot = 0.1032;
    }
    else{
        effAreaChHad = 0.0030;
        effAreaNeuHad = 0.1675;
        effAreaPhot = 0.1598;
    }

    //get photon isolation values, corrected using the rho method
    double phoChHadIso = max(0., pho->chargedHadronIso() - (*fixedGridRhoFastjetAll)*effAreaChHad);
    double phoNeuHadIso = max(0., pho->neutralHadronIso() - (*fixedGridRhoFastjetAll)*effAreaNeuHad);
    double phoPhotIso = max(0., pho->photonIso() - (*fixedGridRhoFastjetAll)*effAreaPhot);

    if(fabs(pho->superCluster()->eta()) < 1.479){ //barrel cuts
        //shower shape cuts
        if(pho->hadTowOverEm() > phoHoverECutBarrel) return false;
        if(pho->full5x5_sigmaIetaIeta() > phoSigmaIetaIetaCutBarrel) return false;

        //isolation cuts
        if(phoChHadIso > phoChHadIsoCutBarrel) return false;
        if(phoNeuHadIso > phoNeuHadIsoCutBarrel + phoNeuHadIsoSlopeBarrel*pho->pt()) return false;
        if(phoPhotIso > phoPhotIsoCutBarrel + phoPhotIsoSlopeBarrel*pho->pt()) return false;
    }
    else{ //endcap cuts
        //shower shape cuts
        if(pho->hadTowOverEm() > phoHoverECutEndcap) return false;
        if(pho->full5x5_sigmaIetaIeta() > phoSigmaIetaIetaCutEndcap) return false;

        //isolation cuts
        if(phoChHadIso > phoChHadIsoCutEndcap) return false;
        if(phoNeuHadIso > phoNeuHadIsoCutEndcap + phoNeuHadIsoSlopeEndcap*pho->pt()) return false;
        if(phoPhotIso > phoPhotIsoCutEndcap + phoPhotIsoSlopeEndcap*pho->pt()) return false;
    }
    return true;
}

//jet ID: pT and eta cuts only
//No PU ID is currently applied
//TODO: add jet energy corrections
template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
bool SusyDQM<Mu, Ele, Pho, Jet, Met>::goodJet(const Jet* jet) {
    if(jet->pt() < jetPtCut) return false;
    if(fabs(jet->eta()) > jetEtaCut) return false;
    //apply loose jet quality cuts
    if(jet->neutralHadronEnergyFraction() < 0.99
            && jet->neutralEmEnergyFraction() < 0.99
            && jet->nConstituents() > 1
            && jet->chargedHadronEnergyFraction() > 0
            && jet->chargedMultiplicity() > 0
            && jet->chargedEmEnergyFraction() < 0.99
      ){ //passes
            fractionOfGoodJetsVsEta_numerator->Fill(jet->eta());
            fractionOfGoodJetsVsEta_denominator->Fill(jet->eta());
            fractionOfGoodJetsVsPhi_numerator->Fill(jet->phi());
            fractionOfGoodJetsVsPhi_denominator->Fill(jet->phi());
            return true;
    }
    else{ //fails
            fractionOfGoodJetsVsEta_denominator->Fill(jet->eta());
            fractionOfGoodJetsVsPhi_denominator->Fill(jet->phi());
            return false;
    }
}

//reset arrays and get physics objects
//returns true if all objects are successfully loaded, false otherwise
template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
bool SusyDQM<Mu, Ele, Pho, Jet, Met>::setupEvent(const edm::Event& evt){
    //Muons
    if(!(evt.getByToken(muons_, muons))){
        edm::LogError("SusyDQM") << "Error: muon collection not found!\n";
        return false;
    }
    //Electrons
    if(!(evt.getByToken(electrons_, electrons))){
        edm::LogError("SusyDQM") << "Error: electron collection not found!\n";
        return false;
    }
    //Photons
    if(!(evt.getByToken(photons_, photons))){
        edm::LogError("SusyDQM") << "Error: photon collection not found!\n";
        return false;
    }
    //Jets
    if(!(evt.getByToken(jets_, jets))){
        edm::LogError("SusyDQM") << "Error: jet collection not found!\n";
        return false;
    }
    //MET
    if(!(evt.getByToken(met_, met))){
        edm::LogError("SusyDQM") << "Error: MET collection not found!\n";
        return false;
    }
    //vertices
    if(!(evt.getByToken(vertex_, vertex))){
        edm::LogError("SusyDQM") << "Error: vertex collection not found!\n";
        return false;
    }
    //conversions
    if(!(evt.getByToken(conversions_, conversions))){
        edm::LogError("SusyDQM") << "Error: conversions collection not found!\n";
        return false;
    }
    //beamSpot
    if(!(evt.getByToken(beamSpot_, beamSpot))){
        edm::LogError("SusyDQM") << "Error: beamspot object not found!\n";
        return false;
    }
    //rho
    if(!(evt.getByToken(fixedGridRhoFastjetAll_, fixedGridRhoFastjetAll))){
        edm::LogError("SusyDQM") << "Error: fixedGridRhoFastjetAll not found!\n";
        return false;
    }
    //jet tags
    if(!(evt.getByToken(jetTagCollection_, jetTagCollection))){
        edm::LogError("SusyDQM") << "Error: jetTagCollection not found!\n";
        return false;
    }
    //gen particles
    if(useGen){
        if(!(evt.getByToken(genParticles_, genParticles))){
            edm::LogError("SusyDQM") << "Error: genParticles not found!\n";
            return false;
        }
        if(!(evt.getByToken(genJets_, genJets))){
            edm::LogError("SusyDQM") << "Error: genJets not found!\n";
            return false;
        }
        if(!(evt.getByToken(jetFlavorMatch_, jetFlavorMatch))){
            edm::LogError("SusyDQM") << "Error: jetFlavorMatch not found!\n";
            return false;
        }
    }

    goodMuons = vector<const Mu*>();
    goodElectrons = vector<const Ele*>();
    goodPhotons = vector<const Pho*>();
    goodJets = vector<const Jet*>();

    return true;
}

//select leptons, photons, jets
template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
void SusyDQM<Mu, Ele, Pho, Jet, Met>::selectObjects(const edm::Event& evt) {
    //select good muons
    for(typename std::vector<Mu>::const_iterator mu = muons->begin(); mu != muons->end(); mu++){
        if(goodMuon(&(*mu))){
            goodMuons.push_back(&(*mu));
        }
    }
    //select good electrons
    for(typename std::vector<Ele>::const_iterator ele = electrons->begin(); ele != electrons->end(); ele++){
        if(goodElectron(&(*ele))){
            goodElectrons.push_back(&(*ele));
        }
    }
    //select good photons
    for(typename std::vector<Pho>::const_iterator pho = photons->begin(); pho != photons->end(); pho++){
        if(goodPhoton(evt, &(*pho))){
            goodPhotons.push_back(&(*pho));
        }
    }
    //select good jets
    for(typename std::vector<Jet>::const_iterator jet = jets->begin(); jet != jets->end(); jet++){
        if(useGen){
        //get flavor of jet 
            int jetPartonFlavor = -1;
            for(auto &jetMatch : (*jetFlavorMatch)){
                float deltaRJetMatchJet = reco::deltaR(jetMatch.first->eta(), jetMatch.first->phi(), jet->eta(), jet->phi());
                if(deltaRJetMatchJet < 0.1){
                    const reco::JetFlavour aFlav = jetMatch.second;
                    jetPartonFlavor = aFlav.getFlavour();
                    break;
                }
            }
            if (abs(jetPartonFlavor) == 5){ //b-jet 
                csvV2MediumEfficiencyVsPt_denominator->Fill(jet->pt());
            }
        }
        if(goodJet(&(*jet))){
            goodJets.push_back(&(*jet));
            if(useGen){ //check if the jet is b-tagged
                for(const auto jettag: *jetTagCollection){
                    const float CSV = jettag.second;
                    if(jettag.first->pt() > jetPtCut && CSV > jetCSVV2Cut){
                        //match in deltaR
                        float deltaRJetBJet = reco::deltaR(jettag.first->eta(), jettag.first->phi(), jet->eta(), jet->phi());
                        if(deltaRJetBJet < 0.1){ //jet is b-tagged
                            csvV2MediumEfficiencyVsPt_numerator->Fill(jet->pt());
                        }
                    }
                }
            }
        }
    }
}

//fill lepton histograms -- exclusive in lepton number
template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
void SusyDQM<Mu, Ele, Pho, Jet, Met>::fillLeptonic(const edm::Event& evt) {

    //Check which GEN muons and electrons were identified
    if(useGen){
        for(uint g = 0; g < genParticles->size(); g++){
            const reco::GenParticle gen = genParticles->at(g);
            if(fabs(gen.pdgId()) == 11 && gen.status() == 1){ //electron
                //find the particle's mother -- require it to be an electroweak boson
                const reco::GenParticle *firstMotherWithDifferentID = findFirstMotherWithDifferentID(&gen);
                if (firstMotherWithDifferentID) {
                    int motherId = firstMotherWithDifferentID->pdgId();
                    if(fabs(motherId) != 23 && fabs(motherId) != 24){ //mother is not an EW boson
                        continue;
                    }
                }
                else{
                    continue;
                }

                electronEfficiencyVsPt_denominator->Fill(gen.pt());
                //search for this electron in the collection of selected electrons
                float minDeltaR = -1.0;
                for(uint i = 0; i < goodElectrons.size(); i++){
                    float thisDeltaR =  reco::deltaR(gen.eta(), gen.phi(), goodElectrons[i]->eta(), goodElectrons[i]->phi());
                    if(minDeltaR < 0 || thisDeltaR < minDeltaR) minDeltaR = thisDeltaR;
                }
                if(minDeltaR > 0 && minDeltaR < 0.1){ //electron was found
                    electronEfficiencyVsPt_numerator->Fill(gen.pt());
                }
            }
            else if(fabs(gen.pdgId()) == 13 && gen.status() == 1){ //muon
                //find the particle's mother -- require it to be an electroweak boson
                const reco::GenParticle *firstMotherWithDifferentID = findFirstMotherWithDifferentID(&gen);
                if (firstMotherWithDifferentID) {
                    int motherId = firstMotherWithDifferentID->pdgId();
                    if(fabs(motherId) != 23 && fabs(motherId) != 24){ //mother is not an EW boson
                        continue;
                    }
                }
                else{
                    continue;
                }

                muonEfficiencyVsPt_denominator->Fill(gen.pt());
                //search for this muon in the collection of selected muons
                float minDeltaR = -1.0;
                for(uint i = 0; i < goodMuons.size(); i++){
                    float thisDeltaR = reco::deltaR(gen.eta(), gen.phi(), goodMuons[i]->eta(), goodMuons[i]->phi());
                    if(minDeltaR < 0 || thisDeltaR < minDeltaR) minDeltaR = thisDeltaR;
                }
                if(minDeltaR > 0 && minDeltaR < 0.1){ //muon was found
                    muonEfficiencyVsPt_numerator->Fill(gen.pt());
                }
            }
        }
    }

    //compute HT and MET
    double HT = 0;
    for(uint j = 0; j < goodJets.size(); j++){
        HT += goodJets[j]->pt();
    }
    double MET = met->front().pt();
    
    if(goodMuons.size() > 0 && HT > 250){ //fill leading muon information
        leadingMuPt_HT250->Fill(goodMuons[0]->pt());
        leadingMuEta_HT250->Fill(goodMuons[0]->eta());
        leadingMuPhi_HT250->Fill(goodMuons[0]->phi());
    }
    if(goodElectrons.size() > 0 && HT > 250){ //fill leading electron information
        leadingElePt_HT250->Fill(goodElectrons[0]->pt());
        leadingEleEta_HT250->Fill(goodElectrons[0]->eta());
        leadingElePhi_HT250->Fill(goodElectrons[0]->phi());
    }

    if(goodMuons.size() + goodElectrons.size() == 0) return; //no leptons
    else if(goodMuons.size() + goodElectrons.size() == 1){ //single lepton
        if(goodMuons.size() > 0){ //single muon
            //compute transverse mass = sqrt(2*MET*lepPt*(1-cos(deltaPhi)))
            double mT = sqrt(2*met->front().pt()*goodMuons[0]->pt()*(1 - cos(reco::deltaPhi(met->front().phi(), goodMuons[0]->phi()))));
            mTLepMET_singleLepton->Fill(mT);

            if(goodMuons[0]->pt() > 60){
                HT_singleLepton60->Fill(HT);
                missingEt_singleLepton60->Fill(MET);
                nJets_singleLepton60->Fill(goodJets.size());
            }
        }
        else if(goodElectrons.size() > 0){
            //compute transverse mass = sqrt(2*MET*lepPt*(1-cos(deltaPhi)))
            double mT = sqrt(2*met->front().pt()*goodElectrons[0]->pt()*(1 - cos(reco::deltaPhi(met->front().phi(), goodElectrons[0]->phi()))));
            mTLepMET_singleLepton->Fill(mT);

            if(goodElectrons[0]->pt() > 60){
                HT_singleLepton60->Fill(HT);
                missingEt_singleLepton60->Fill(MET);
                nJets_singleLepton60->Fill(goodJets.size());
            }
        }
        return;
    }
    else if(goodMuons.size() + goodElectrons.size() == 2){ //dilepton
        //case 1: two muons
        if(goodMuons.size() == 2){
            //compute invariant mass
            math::XYZTLorentzVector p1(goodMuons[0]->px(), goodMuons[0]->py(), goodMuons[0]->pz(), goodMuons[0]->energy());
            math::XYZTLorentzVector p2(goodMuons[1]->px(), goodMuons[1]->py(), goodMuons[1]->pz(), goodMuons[1]->energy());
            double invMass = (p1+p2).M();

            //fill appropriate mass histograms
            if(goodMuons[0]->charge() != goodMuons[1]->charge()){
                osDiMuonMass->Fill(invMass);
            }
            else{
                ssDiMuonMass->Fill(invMass);
            }
        }
        //case 2: two electrons
        else if(goodElectrons.size() == 2){
            //compute invariant mass
            math::XYZTLorentzVector p1(goodElectrons[0]->px(), goodElectrons[0]->py(), goodElectrons[0]->pz(), goodElectrons[0]->energy());
            math::XYZTLorentzVector p2(goodElectrons[1]->px(), goodElectrons[1]->py(), goodElectrons[1]->pz(), goodElectrons[1]->energy());
            double invMass = (p1+p2).M();

            //fill appropriate mass histograms
            if(goodElectrons[0]->charge() != goodElectrons[1]->charge()){
                osDiElectronMass->Fill(invMass);
            }
            else{
                ssDiElectronMass->Fill(invMass);
            }

        }
        //case 3: one of each
        else if(goodMuons.size() == 1 && goodElectrons.size() == 1){
            //compute invariant mass
            math::XYZTLorentzVector p1(goodMuons[0]->px(), goodMuons[0]->py(), goodMuons[0]->pz(), goodMuons[0]->energy());
            math::XYZTLorentzVector p2(goodElectrons[0]->px(), goodElectrons[0]->py(), goodElectrons[0]->pz(), goodElectrons[0]->energy());
            double invMass = (p1+p2).M();

            //fill mass histograms
            if(goodMuons[0]->charge() != goodElectrons[0]->charge()) osOfDiLeptonMass->Fill(invMass);
            else ssOfDiLeptonMass->Fill(invMass);
        }
        return;
    }
    else if(goodMuons.size() + goodElectrons.size() >= 3){
        HT_threeOrMoreLeptons->Fill(HT);
        missingEt_threeOrMoreLeptons->Fill(MET);
    }

}

//fill photon histograms (inclusive in number of photons)
template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
void SusyDQM<Mu, Ele, Pho, Jet, Met>::fillPhotonic(const edm::Event& evt) {
    //compute HT and MET
    double HT = 0;
    for(uint j = 0; j < goodJets.size(); j++) HT += goodJets[j]->pt();
    double MET = met->front().pt();

    if(goodPhotons.size() > 0){ 
        //fill leading photon information
        if(HT > 250){
            leadingPhoPt_HT250->Fill(goodPhotons[0]->pt());
            leadingPhoEta_HT250->Fill(goodPhotons[0]->eta());
            leadingPhoPhi_HT250->Fill(goodPhotons[0]->phi());
        }

        //fill single-photon histograms
        if(goodPhotons[0]->pt() > 80){
            HT_photon80->Fill(HT);
            missingEt_photon80->Fill(MET);
            nJets_photon80->Fill(goodJets.size());
            deltaPhiPhoMET_photon80->Fill(reco::deltaPhi(goodPhotons[0]->phi(), met->front().phi()));
        }
    }
    if(goodPhotons.size() > 1){ //fill diphoton histograms using two leading photons
        math::XYZTLorentzVector p1(goodPhotons[0]->px(), goodPhotons[0]->py(), goodPhotons[0]->pz(), goodPhotons[0]->energy());
        math::XYZTLorentzVector p2(goodPhotons[1]->px(), goodPhotons[1]->py(), goodPhotons[1]->pz(), goodPhotons[1]->energy());
        double invMass = (p1+p2).M();
        
        if(HT > 250) diPhotonMass_HT250->Fill(invMass);
        if(MET > 100) diPhotonMass_MET100->Fill(invMass);
        
        if(goodJets.size() > 0){
            //compute razor variables

            vector<math::XYZTLorentzVector> goodJetsP4;
            //jets as XYZTLorentzVectors
            for(uint j = 0; j < goodJets.size(); j++) goodJetsP4.push_back(math::XYZTLorentzVector(goodJets[j]->px(), goodJets[j]->py(), goodJets[j]->pz(), goodJets[j]->energy()));
            //add the diphoton system to the hemispheres
            goodJetsP4.push_back(math::XYZTLorentzVector(goodPhotons[0]->px()+goodPhotons[1]->px(), goodPhotons[0]->py()+goodPhotons[1]->py(), goodPhotons[0]->pz()+goodPhotons[1]->pz(), goodPhotons[0]->energy()+goodPhotons[1]->energy()));
            //partition the jets and photons into hemispheres for razor calculation
            vector<math::XYZTLorentzVector> hemispheresRazor = createHemispheresRazor(goodJetsP4);
            //switch to TLorentzVectors for compatibility with razor variable computation
            TLorentzVector ja(hemispheresRazor[0].x(),hemispheresRazor[0].y(),hemispheresRazor[0].z(),hemispheresRazor[0].t());
            TLorentzVector jb(hemispheresRazor[1].x(),hemispheresRazor[1].y(),hemispheresRazor[1].z(),hemispheresRazor[1].t());

            double MR = calcMR(ja, jb);
            double Rsq = calcRsq(MR, ja, jb, met);
            if(Rsq > 0.02) MR_Rsq0p02Diphoton->Fill(MR);
            if(MR > 100) Rsq_MR100Diphoton->Fill(Rsq);
        }
    }
}

    //fill hadronic histograms
    template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
        void SusyDQM<Mu, Ele, Pho, Jet, Met>::fillHadronic(const edm::Event& evt) {
            if(goodMuons.size() > 0 || goodElectrons.size() > 0) return; //consider all-hadronic events only

    //compute HT, MHT and MET, and minDeltaPhi
    double MET = met->front().pt();
    double METPhi = met->front().phi();
    double HT = 0;
    double sumJetPx = 0;
    double sumJetPy = 0;
    double minDeltaPhi = -1;
    for(uint j = 0; j < goodJets.size(); j++){
        HT += goodJets[j]->pt();
        sumJetPx += goodJets[j]->px();
        sumJetPy += goodJets[j]->py();
        double thisDeltaPhi = reco::deltaPhi(goodJets[j]->phi(), met->front().phi());
        if(minDeltaPhi < 0 || thisDeltaPhi < minDeltaPhi) minDeltaPhi = thisDeltaPhi;
    }
    double mHT = sqrt(sumJetPx*sumJetPx + sumJetPy*sumJetPy);
    if(minDeltaPhi > 0) minDeltaPhiJetMET->Fill(minDeltaPhi);

    //fill HT, MHT and MET histograms
    if(HT > 250) missingEt_HT250->Fill(MET);
    if(MET > 150) metPhi_MET150->Fill(METPhi);
    if(MET > 150) HT_MET150->Fill(HT);
    MHT->Fill(mHT);
    if(goodJets.size() > 0){
        missingEtOverMHT->Fill(MET/mHT);
        MHTOverHT->Fill(mHT/HT);
    }

    //fill jet multiplicity histograms
    nJets->Fill(goodJets.size());
    if(HT > 250) nJets_HT250->Fill(goodJets.size());
    if(MET > 150) nJets_MET150->Fill(goodJets.size());

    //count b-tagged jets
    uint nBTaggedJets = 0;
    for(const auto jet: *jetTagCollection){
        const float CSV = jet.second;
        if(jet.first->pt() > jetPtCut){
            if(CSV > jetCSVV2Cut){
                nBTaggedJets++;
            }
        }
    }
    if(HT > 250) nBTaggedJetsCSVV2M_HT250->Fill(nBTaggedJets);
    if(nBTaggedJets > 0) missingEt_1BTaggedJet->Fill(MET);

    if(goodJets.size() > 0 && goodJets[0]->pt() > 80){ //fill leading jet information
       leadingJetPt_pT80->Fill(goodJets[0]->pt());
       leadingJetEta_pT80->Fill(goodJets[0]->eta());
       leadingJetPhi_pT80->Fill(goodJets[0]->phi());
       deltaPhiJetMET_jet80->Fill(reco::deltaPhi(goodJets[0]->phi(), met->front().phi()));
    }
    if(goodJets.size() > 1 && goodJets[0]->pt() > 80 && goodJets[1]->pt() > 80){
        deltaPhiJJ_2Jets80->Fill(reco::deltaPhi(goodJets[0]->phi(), goodJets[1]->phi()));
        
        //compute razor variables
        vector<math::XYZTLorentzVector> goodJetsP4;
        //jets as XYZTLorentzVectors
        for(uint j = 0; j < goodJets.size(); j++) goodJetsP4.push_back(math::XYZTLorentzVector(goodJets[j]->px(), goodJets[j]->py(), goodJets[j]->pz(), goodJets[j]->energy()));
        //partition the jets into hemispheres for razor calculation
        vector<math::XYZTLorentzVector> hemispheresRazor = createHemispheresRazor(goodJetsP4);
        //switch to TLorentzVectors for compatibility with razor variable computation
        TLorentzVector ja(hemispheresRazor[0].x(),hemispheresRazor[0].y(),hemispheresRazor[0].z(),hemispheresRazor[0].t());
        TLorentzVector jb(hemispheresRazor[1].x(),hemispheresRazor[1].y(),hemispheresRazor[1].z(),hemispheresRazor[1].t());
  
        double MR = calcMR(ja, jb);
        double Rsq = calcRsq(MR, ja, jb, met);
        if(Rsq > 0.15) MR_RSq0p15->Fill(MR);
        if(MR > 300) RSq_MR300->Fill(Rsq);

        //compute alphaT
        double deltaHT = calcDeltaHT(goodJetsP4);
        double AlphaT = 0.5*(1-deltaHT/HT)/sqrt(1-(mHT/HT)*(mHT/HT));
        alphaT->Fill(AlphaT);

        //compute MT2
        
        //jets as TLorentzVectors
        vector<TLorentzVector> goodJetsForMT2;
        for(uint j = 0; j < goodJets.size(); j++) goodJetsForMT2.push_back(TLorentzVector(goodJets[j]->px(), goodJets[j]->py(), goodJets[j]->pz(), goodJets[j]->energy()));
        //MET as TLorentzVector
        TLorentzVector PFMET(met->front().px(), met->front().py(), 0.0, met->front().pt());

        //Get MT2 using test mass = 0, massless hemispheres
        //Seed hemispheres with the 2 objects having maximum invariant mass (choice #2)
        //Cluster hemispheres minimizing the Lund distance (choice #3)
        double MT2 = calcMT2(0.0, false, goodJetsForMT2, PFMET, 2, 3); 
        mT2->Fill(MT2);
        
        if(goodJets.size() == 2){ //fill exclusive dijet histograms
            deltaPhiJJ_dijet80Exclusive->Fill(reco::deltaPhi(goodJets[0]->phi(), goodJets[1]->phi()));
        }
    }
}

//analyze event
template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
void SusyDQM<Mu, Ele, Pho, Jet, Met>::analyze(const edm::Event& evt, const edm::EventSetup& iSetup) {
    bool eventIsGood = setupEvent(evt);
    if(!eventIsGood) return;

    selectObjects(evt);
    fillLeptonic(evt);
    fillPhotonic(evt);
    fillHadronic(evt);
}

//adapted from JetMET/plugins/HLTRHemisphere.cc
//clusters the input four-vectors into two "mega-jets", finding the combination that minimizes the sum(mass^2) of the two mega-jets.
template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
vector<math::XYZTLorentzVector> SusyDQM<Mu, Ele, Pho, Jet, Met>::createHemispheresRazor(std::vector<math::XYZTLorentzVector> jets) {
    using namespace math;
    using namespace reco;
    vector<math::XYZTLorentzVector> hlist;
    math::XYZTLorentzVector j1R(0.1, 0., 0., 0.1);
    math::XYZTLorentzVector j2R(0.1, 0., 0., 0.1);
    int nJets = jets.size();
    if(nJets<2){ // put empty hemispheres if not enough jets
        hlist.push_back(j1R);
        hlist.push_back(j2R);
        return hlist;
    }
    unsigned int N_comb = pow(2,nJets); // compute the number of combinations of jets possible
    //Make the hemispheres
    double M_minR = 9999999999.0;
    unsigned int j_count;
    for (unsigned int i = 0; i < N_comb; i++) {
        math::XYZTLorentzVector j_temp1, j_temp2;
        unsigned int itemp = i;
        j_count = N_comb/2;
        unsigned int count = 0;
        while (j_count > 0) {
            if (itemp/j_count == 1){
                j_temp1 += jets.at(count);
            } else {
                j_temp2 += jets.at(count);
            }
            itemp -= j_count * (itemp/j_count);
            j_count /= 2;
            count++;
        }
        double M_temp = j_temp1.M2() + j_temp2.M2();
        if (M_temp < M_minR) {
            M_minR = M_temp;
            j1R = j_temp1;
            j2R = j_temp2;
        }
    }
    hlist.push_back(j1R);
    hlist.push_back(j2R);
    return hlist;
}

//computes the smallest value of abs(sumEt(pseudojet 1) - sumEt(pseudojet 2)) among all assignments of the input jets to two "pseudojets"
template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
double SusyDQM<Mu, Ele, Pho, Jet, Met>::calcDeltaHT(std::vector<math::XYZTLorentzVector> jets) {
    using namespace math;
    using namespace reco;
    int nJets = jets.size();
    if(nJets<2){ // return -1 if not enough jets
        return -1.0;
    }
    unsigned int N_comb = pow(2,nJets); // compute the number of combinations of jets possible
    //Make the hemispheres
    double minDeltaHT = -1;
    unsigned int j_count;
    for (unsigned int i = 0; i < N_comb; i++) {
        vector<math::XYZTLorentzVector> j_temp1, j_temp2;
        unsigned int itemp = i;
        j_count = N_comb/2;
        unsigned int count = 0;
        while (j_count > 0) {
            if (itemp/j_count == 1){
                j_temp1.push_back(jets.at(count));
            } else {
                j_temp2.push_back(jets.at(count));
            }
            itemp -= j_count * (itemp/j_count);
            j_count /= 2;
            count++;
        }
        double thisSumEt1 = 0;
        for(uint jIndex = 0; jIndex < j_temp1.size(); jIndex++){
            thisSumEt1 += j_temp1[jIndex].Pt();
        }
        double thisSumEt2 = 0;
        for(uint jIndex = 0; jIndex < j_temp2.size(); jIndex++){
            thisSumEt2 += j_temp2[jIndex].Pt();
        }
        double thisDeltaHT = fabs(thisSumEt1 - thisSumEt2);
        if(minDeltaHT < 0 || thisDeltaHT < minDeltaHT) minDeltaHT = thisDeltaHT;
    }
    return minDeltaHT;
}

//adapted from JetMET/plugins/HLTRFilter.cc
template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
double SusyDQM<Mu, Ele, Pho, Jet, Met>::calcMR(TLorentzVector ja, TLorentzVector jb) {
    if(ja.Pt()<=0.2) return -999;
    ja.SetPtEtaPhiM(ja.Pt(),ja.Eta(),ja.Phi(),0.0);
    jb.SetPtEtaPhiM(jb.Pt(),jb.Eta(),jb.Phi(),0.0);
    if(ja.Pt() > jb.Pt()){
        TLorentzVector temp = ja;
        ja = jb;
        jb = temp;
    }
    double A = ja.P();
    double B = jb.P();
    double az = ja.Pz();
    double bz = jb.Pz();
    TVector3 jaT, jbT;
    jaT.SetXYZ(ja.Px(),ja.Py(),0.0);
    jbT.SetXYZ(jb.Px(),jb.Py(),0.0);
    double ATBT = (jaT+jbT).Mag2();
    double MR = sqrt((A+B)*(A+B)-(az+bz)*(az+bz)-
            (jbT.Dot(jbT)-jaT.Dot(jaT))*(jbT.Dot(jbT)-jaT.Dot(jaT))/(jaT+jbT).Mag2());
    double mybeta = (jbT.Dot(jbT)-jaT.Dot(jaT))/
        sqrt(ATBT*((A+B)*(A+B)-(az+bz)*(az+bz)));
    double mygamma = 1./sqrt(1.-mybeta*mybeta);
    //use gamma times MRstar
    return MR*mygamma;
}

template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
double SusyDQM<Mu, Ele, Pho, Jet, Met>::calcRsq(double MR, TLorentzVector ja, TLorentzVector jb, edm::Handle<std::vector<Met> > inputMet) {
    TVector3 met;
    met.SetPtEtaPhi((inputMet->front()).pt(),0.0,(inputMet->front()).phi());
    double MTR = sqrt(0.5*(met.Mag()*(ja.Pt()+jb.Pt()) - met.Dot(ja.Vect()+jb.Vect())));
    double R = MTR/MR;
    return R*R;
}

template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
double SusyDQM<Mu, Ele, Pho, Jet, Met>::calcMT2(float testMass, bool massive, vector<TLorentzVector> jets, TLorentzVector MET, int hemi_seed, int hemi_association){
    //computes MT2 using a test mass of testMass, with hemispheres made massless if massive is set to false
    //hemispheres are clustered by finding the grouping of input jets that minimizes the Lund distance

    using namespace dqm;

    if(jets.size() < 2) return -9999; //need at least two jets for the calculation
    vector<float> px, py, pz, E;
    for(uint i = 0; i < jets.size(); i++){
        //push 4vector components onto individual lists
        px.push_back(jets[i].Px());
        py.push_back(jets[i].Py());
        pz.push_back(jets[i].Pz());
        E.push_back(jets[i].E());
    }
    
    //form the hemispheres using the provided Hemisphere class
    Hemisphere* hemis = new Hemisphere(px, py, pz, E, hemi_seed, hemi_association);
    vector<int> grouping = hemis->getGrouping();
    TLorentzVector pseudojet1(0.,0.,0.,0.);
    TLorentzVector pseudojet2(0.,0.,0.,0.);
        
    //make the hemisphere vectors
    for(uint i=0; i<px.size(); ++i){
        if(grouping[i]==1){
            pseudojet1.SetPx(pseudojet1.Px() + px[i]);
            pseudojet1.SetPy(pseudojet1.Py() + py[i]);
            pseudojet1.SetPz(pseudojet1.Pz() + pz[i]);
            pseudojet1.SetE( pseudojet1.E()  + E[i]);   
        }else if(grouping[i] == 2){
            pseudojet2.SetPx(pseudojet2.Px() + px[i]);
            pseudojet2.SetPy(pseudojet2.Py() + py[i]);
            pseudojet2.SetPz(pseudojet2.Pz() + pz[i]);
            pseudojet2.SetE( pseudojet2.E()  + E[i]);
        }
    }
    delete hemis;
    
    //now compute MT2 using the Davismt2 class

    //these arrays contain (mass, px, py) for the pseudojets and the MET
    double pa[3];
    double pb[3];
    double pmiss[3];
    
    pmiss[0] = 0;
    pmiss[1] = static_cast<double> (MET.Px());
    pmiss[2] = static_cast<double> (MET.Py());
    
    pa[0] = static_cast<double> (massive ? pseudojet1.M() : 0);
    pa[1] = static_cast<double> (pseudojet1.Px());
    pa[2] = static_cast<double> (pseudojet1.Py());
    
    pb[0] = static_cast<double> (massive ? pseudojet2.M() : 0);
    pb[1] = static_cast<double> (pseudojet2.Px());
    pb[2] = static_cast<double> (pseudojet2.Py());
    
    Davismt2 *mt2 = new Davismt2();
    mt2->set_momenta(pa, pb, pmiss);
    mt2->set_mn(testMass);
    Float_t MT2=mt2->get_mt2();
    delete mt2;
    return MT2;
}

template <typename Mu, typename Ele, typename Pho, typename Jet, typename Met>
const reco::GenParticle* SusyDQM<Mu, Ele, Pho, Jet, Met>::findFirstMotherWithDifferentID(const reco::GenParticle *particle){
    // Is this the first parent with a different ID? If yes, return, otherwise
    // go deeper into recursion
    if (particle->numberOfMothers() > 0 && particle->pdgId() != 0) {
        if (particle->pdgId() == particle->mother(0)->pdgId()) {
            return findFirstMotherWithDifferentID((reco::GenParticle*)particle->mother(0));
        } else {
            return (reco::GenParticle*)particle->mother(0);
        }
    }
    return 0;
}

#endif

typedef SusyDQM<reco::Muon, reco::GsfElectron, reco::Photon, reco::PFJet, reco::PFMET> RecoSusyDQM;
typedef SusyDQM< pat::Muon, pat::Electron, pat::Photon, pat::Jet, pat::MET > PatSusyDQM;

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
