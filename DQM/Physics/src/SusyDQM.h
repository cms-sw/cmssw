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
#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
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
        MonitorElement* muonIsolation;
        MonitorElement* eleIsolation;

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
        MonitorElement* leadingJetMass_pT80;
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

typedef SusyDQM<reco::Muon, reco::GsfElectron, reco::Photon, reco::PFJet, reco::PFMET> RecoSusyDQM;
typedef SusyDQM< pat::Muon, pat::Electron, pat::Photon, pat::Jet, pat::MET > PatSusyDQM;

#endif

// Local Variables:
// show-trailing-whitespace: t
// truncate-lines: t
// End:
