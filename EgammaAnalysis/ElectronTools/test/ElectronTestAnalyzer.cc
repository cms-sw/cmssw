// -*- C++ -*-
//
// Package:    ElectronTestAnalyzer
// Class:      ElectronTestAnalyzer
//
/**\class ElectronTestAnalyzer

 Description: <one line class summary>

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Daniele Benedetti

// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/MuonReco/interface/Muon.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/ElectronSeed.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "SimDataFormats/GeneratorProducts/interface/HepMCProduct.h"
#include "RecoEcal/EgammaCoreTools/interface/EcalClusterLazyTools.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "EgammaAnalysis/ElectronTools/interface/EGammaMvaEleEstimator.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "TrackingTools/IPTools/interface/IPTools.h"
#include "FWCore/Utilities/interface/isFinite.h"
#include "DataFormats/Math/interface/normalizedPhi.h"

#include <cmath>
#include <vector>
#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TLorentzVector.h>
#include "TMVA/Factory.h"
#include "TMVA/Tools.h"
#include "TMVA/Reader.h"

//
// class decleration
//

class ElectronTestAnalyzer : public edm::one::EDAnalyzer<edm::one::SharedResources> {
public:
  explicit ElectronTestAnalyzer(const edm::ParameterSet&);
  ~ElectronTestAnalyzer() override;

private:
  void beginJob() override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;
  virtual void myBindVariables();
  virtual void myVar(const reco::GsfElectron& ele,
                     const reco::Vertex& vertex,
                     const TransientTrackBuilder& transientTrackBuilder,
                     EcalClusterLazyTools const& myEcalCluster,
                     bool printDebug = kFALSE);
  virtual void evaluate_mvas(const edm::Event& iEvent, const edm::EventSetup& iSetup);

  bool trainTrigPresel(const reco::GsfElectron& ele);

  edm::ParameterSet conf_;

  edm::EDGetTokenT<reco::GsfElectronCollection> gsfEleToken_;
  edm::EDGetTokenT<reco::GenParticleCollection> genToken_;
  //edm::EDGetTokenT<edm::HepMCProduct>  mcTruthToken_;
  edm::EDGetTokenT<reco::VertexCollection> vertexToken_;
  //edm::EDGetTokenT<reco::PFCandidateCollection> pfCandToken_;
  edm::EDGetTokenT<double> eventrhoToken_;
  edm::EDGetTokenT<reco::MuonCollection> muonToken_;
  edm::EDGetTokenT<EcalRecHitCollection> reducedEBRecHitCollectionToken_;
  edm::EDGetTokenT<EcalRecHitCollection> reducedEERecHitCollectionToken_;
  edm::ESGetToken<TransientTrackBuilder, TransientTrackRecord> builderToken_;

  const EcalClusterLazyTools::ESGetTokens ecalClusterESGetTokens_;

  EGammaMvaEleEstimator* myMVATrigV0;
  EGammaMvaEleEstimator* myMVATrigNoIPV0;
  EGammaMvaEleEstimator* myMVANonTrigV0;

  TMVA::Reader* myTMVAReader;
  Float_t myMVAVar_fbrem;
  Float_t myMVAVar_kfchi2;
  Float_t myMVAVar_kfhits;
  Float_t myMVAVar_gsfchi2;

  Float_t myMVAVar_deta;
  Float_t myMVAVar_dphi;
  Float_t myMVAVar_detacalo;
  Float_t myMVAVar_dphicalo;

  Float_t myMVAVar_see;
  Float_t myMVAVar_spp;
  Float_t myMVAVar_etawidth;
  Float_t myMVAVar_phiwidth;
  Float_t myMVAVar_e1x5e5x5;
  Float_t myMVAVar_R9;
  Float_t myMVAVar_nbrems;

  Float_t myMVAVar_HoE;
  Float_t myMVAVar_EoP;
  Float_t myMVAVar_IoEmIoP;
  Float_t myMVAVar_eleEoPout;
  Float_t myMVAVar_PreShowerOverRaw;
  Float_t myMVAVar_EoPout;

  Float_t myMVAVar_d0;
  Float_t myMVAVar_ip3d;

  Float_t myMVAVar_eta;
  Float_t myMVAVar_pt;

  double _Rho;
  unsigned int ev;
  // ----------member data ---------------------------

  TH1F *h_mva_nonTrig, *h_mva_trig, *h_mva_trigNonIp;
  TH1F* h_fbrem;
  TH1F* h_kfchi2;
  TH1F* h_kfhits;
  TH1F* h_gsfchi2;
  TH1F* h_deta;
  TH1F* h_dphi;
  TH1F* h_detacalo;
  TH1F* h_dphicalo;
  TH1F* h_see;
  TH1F* h_spp;
  TH1F* h_etawidth;
  TH1F* h_phiwidth;
  TH1F* h_e1x5e5x5;
  TH1F* h_nbrems;
  TH1F* h_R9;
  TH1F* h_HoE;
  TH1F* h_EoP;
  TH1F* h_IoEmIoP;
  TH1F* h_eleEoPout;
  TH1F* h_EoPout;
  TH1F* h_PreShowerOverRaw;
  TH1F* h_pt;
};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
ElectronTestAnalyzer::ElectronTestAnalyzer(const edm::ParameterSet& iConfig)
    : conf_(iConfig),
      gsfEleToken_(consumes<reco::GsfElectronCollection>(edm::InputTag("gsfElectrons"))),
      genToken_(consumes<reco::GenParticleCollection>(edm::InputTag("genParticles"))),
      //mcTruthToken_(consumes<edm::HepMCProduct>(edm::InputTag("generator"))),
      vertexToken_(consumes<reco::VertexCollection>(edm::InputTag("offlinePrimaryVertices"))),
      //pfCandToken_(consumes<reco::PFCandidateCollection>(edm::InputTag("particleFlow"))),
      eventrhoToken_(consumes<double>(edm::InputTag("kt6PFJets", "rho"))),
      muonToken_(consumes<reco::MuonCollection>(edm::InputTag("muons"))),
      reducedEBRecHitCollectionToken_(consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEB"))),
      reducedEERecHitCollectionToken_(consumes<EcalRecHitCollection>(edm::InputTag("reducedEcalRecHitsEE"))),
      builderToken_(esConsumes(edm::ESInputTag("", "TransientTrackBuilder"))),
      ecalClusterESGetTokens_{consumesCollector()} {
  usesResource(TFileService::kSharedResource);

  Bool_t manualCat = true;

  ev = 0;

  // NOTE: it is better if you copy the MVA weight files locally

  std::vector<std::string> catWTrigV0;
  catWTrigV0.push_back("../data/Electrons_BDTG_TrigV0_Cat1.weights.xml");
  catWTrigV0.push_back("../data/Electrons_BDTG_TrigV0_Cat2.weights.xml");
  catWTrigV0.push_back("../data/Electrons_BDTG_TrigV0_Cat3.weights.xml");
  catWTrigV0.push_back("../data/Electrons_BDTG_TrigV0_Cat4.weights.xml");
  catWTrigV0.push_back("../data/Electrons_BDTG_TrigV0_Cat5.weights.xml");
  catWTrigV0.push_back("../data/Electrons_BDTG_TrigV0_Cat6.weights.xml");

  myMVATrigV0 = new EGammaMvaEleEstimator();
  myMVATrigV0->initialize("BDT", EGammaMvaEleEstimator::kTrig, manualCat, catWTrigV0);

  std::vector<std::string> catWTrigNoIPV0;
  catWTrigNoIPV0.push_back("../data/Electrons_BDTG_TrigNoIPV0_2012_Cat1.weights.xml");
  catWTrigNoIPV0.push_back("../data/Electrons_BDTG_TrigNoIPV0_2012_Cat2.weights.xml");
  catWTrigNoIPV0.push_back("../data/Electrons_BDTG_TrigNoIPV0_2012_Cat3.weights.xml");
  catWTrigNoIPV0.push_back("../data/Electrons_BDTG_TrigNoIPV0_2012_Cat4.weights.xml");
  catWTrigNoIPV0.push_back("../data/Electrons_BDTG_TrigNoIPV0_2012_Cat5.weights.xml");
  catWTrigNoIPV0.push_back("../data/Electrons_BDTG_TrigNoIPV0_2012_Cat6.weights.xml");

  myMVATrigNoIPV0 = new EGammaMvaEleEstimator();
  myMVATrigNoIPV0->initialize("BDT", EGammaMvaEleEstimator::kTrigNoIP, manualCat, catWTrigNoIPV0);

  std::vector<std::string> catWNonTrigV0;
  catWNonTrigV0.push_back("../data/Electrons_BDTG_NonTrigV0_Cat1.weights.xml");
  catWNonTrigV0.push_back("../data/Electrons_BDTG_NonTrigV0_Cat2.weights.xml");
  catWNonTrigV0.push_back("../data/Electrons_BDTG_NonTrigV0_Cat3.weights.xml");
  catWNonTrigV0.push_back("../data/Electrons_BDTG_NonTrigV0_Cat4.weights.xml");
  catWNonTrigV0.push_back("../data/Electrons_BDTG_NonTrigV0_Cat5.weights.xml");
  catWNonTrigV0.push_back("../data/Electrons_BDTG_NonTrigV0_Cat6.weights.xml");

  myMVANonTrigV0 = new EGammaMvaEleEstimator();
  myMVANonTrigV0->initialize("BDT", EGammaMvaEleEstimator::kNonTrig, manualCat, catWNonTrigV0);

  edm::Service<TFileService> fs;

  h_mva_nonTrig = fs->make<TH1F>("h_mva_nonTrig", " ", 100, -1.1, 1.1);
  h_mva_trig = fs->make<TH1F>("h_mva_trig", " ", 100, -1.1, 1.1);
  h_mva_trigNonIp = fs->make<TH1F>("h_mva_trigNonIp", " ", 100, -1.1, 1.1);

  h_fbrem = fs->make<TH1F>("h_fbrem", " ", 100, -1., 1.);
  h_kfchi2 = fs->make<TH1F>("h_kfchi2", " ", 100, 0, 15);
  h_kfhits = fs->make<TH1F>("h_kfhits", " ", 25, 0, 25);
  h_gsfchi2 = fs->make<TH1F>("h_gsfchi2", " ", 100, 0., 50);

  h_deta = fs->make<TH1F>("h_deta", " ", 100, 0., 0.06);
  h_dphi = fs->make<TH1F>("h_dphi", " ", 100, 0., 0.3);
  h_detacalo = fs->make<TH1F>("h_detacalo", " ", 100, 0., 0.05);
  h_dphicalo = fs->make<TH1F>("h_dphicalo", " ", 100, 0., 0.2);
  h_see = fs->make<TH1F>("h_see", " ", 100, 0., 0.06);
  h_spp = fs->make<TH1F>("h_spp", " ", 100, 0., 0.09);
  h_etawidth = fs->make<TH1F>("h_etawidth", " ", 100, 0., 0.1);
  h_phiwidth = fs->make<TH1F>("h_phiwidth", " ", 100, 0., 0.2);
  h_e1x5e5x5 = fs->make<TH1F>("h_e1x5e5x5", " ", 100, -0.1, 1.1);
  h_R9 = fs->make<TH1F>("h_R9", " ", 100, 0., 2.);
  h_nbrems = fs->make<TH1F>("h_nbrems", " ", 100, 0., 10);
  h_HoE = fs->make<TH1F>("h_HoE", " ", 100, 0., 0.5);
  h_EoP = fs->make<TH1F>("h_EoP", " ", 100, 0., 5.);
  h_IoEmIoP = fs->make<TH1F>("h_IoEmIoP", " ", 100, -0.15, 0.15);
  h_eleEoPout = fs->make<TH1F>("h_eleEoPout", " ", 100, 0., 5.);
  h_EoPout = fs->make<TH1F>("h_EoPout", " ", 100, 0., 5.);
  h_PreShowerOverRaw = fs->make<TH1F>("h_PreShowerOverRaw", " ", 100, 0., 0.3);
  h_pt = fs->make<TH1F>("h_pt", " ", 100, 0., 50);
}

ElectronTestAnalyzer::~ElectronTestAnalyzer() {
  // do anything here that needs to be done at desctruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

// ------------ method called to for each event  ------------
void ElectronTestAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  ElectronTestAnalyzer::evaluate_mvas(iEvent, iSetup);

  edm::Handle<reco::GsfElectronCollection> theEGammaCollection;
  iEvent.getByToken(gsfEleToken_, theEGammaCollection);
  const reco::GsfElectronCollection theEGamma = *(theEGammaCollection.product());

  edm::Handle<reco::GenParticleCollection> genParticles;
  iEvent.getByToken(genToken_, genParticles);
  //InputTag  mcTruthToken_(string("generator"));
  //edm::Handle<edm::HepMCProduct> pMCTruth;
  //iEvent.getByToken(mcTruthToken_,pMCTruth);
  //const HepMC::GenEvent* genEvent = pMCTruth->GetEvent();

  edm::Handle<reco::VertexCollection> thePrimaryVertexColl;
  iEvent.getByToken(vertexToken_, thePrimaryVertexColl);

  _Rho = 0;
  edm::Handle<double> rhoPtr;
  iEvent.getByToken(eventrhoToken_, rhoPtr);
  _Rho = *rhoPtr;

  reco::Vertex dummy;
  const reco::Vertex* pv = &dummy;
  if (thePrimaryVertexColl->size() != 0) {
    pv = &*thePrimaryVertexColl->begin();
  } else {  // create a dummy PV
    reco::Vertex::Error e;
    e(0, 0) = 0.0015 * 0.0015;
    e(1, 1) = 0.0015 * 0.0015;
    e(2, 2) = 15. * 15.;
    reco::Vertex::Point p(0, 0, 0);
    dummy = reco::Vertex(p, e, 0, 0, 0);
  }
  EcalClusterLazyTools lazyTools(
      iEvent, ecalClusterESGetTokens_.get(iSetup), reducedEBRecHitCollectionToken_, reducedEERecHitCollectionToken_);

  TransientTrackBuilder thebuilder = iSetup.getData(builderToken_);

  bool debug = true;
  bool debugMVAclass = false;
  bool debugMyVar = false;

  ev++;

  // Validation from generator events

  for (size_t i(0); i < genParticles->size(); i++) {
    const reco::GenParticle& genPtcl = (*genParticles)[i];
    float etamc = genPtcl.eta();
    float phimc = genPtcl.phi();
    float ptmc = genPtcl.pt();

    if (abs(genPtcl.pdgId()) == 11 && genPtcl.status() == 1 && ptmc > 10. && std::abs(etamc) < 2.5) {
      for (unsigned int j = 0; j < theEGamma.size(); j++) {
        float etareco = theEGamma[j].eta();
        float phireco = theEGamma[j].phi();
        float deta = etamc - etareco;
        float dphi = normalizedPhi(phimc - phireco);
        float dR = sqrt(deta * deta + dphi * dphi);

        if (dR < 0.1) {
          myVar((theEGamma[j]), *pv, thebuilder, lazyTools, debugMyVar);
          myBindVariables();

          h_fbrem->Fill(myMVAVar_fbrem);
          h_kfchi2->Fill(myMVAVar_kfchi2);
          h_kfhits->Fill(myMVAVar_kfhits);
          h_gsfchi2->Fill(myMVAVar_gsfchi2);

          h_deta->Fill(myMVAVar_deta);
          h_dphi->Fill(myMVAVar_dphi);
          h_detacalo->Fill(myMVAVar_detacalo);
          h_dphicalo->Fill(myMVAVar_dphicalo);

          h_see->Fill(myMVAVar_see);
          h_spp->Fill(myMVAVar_spp);
          h_etawidth->Fill(myMVAVar_etawidth);
          h_phiwidth->Fill(myMVAVar_phiwidth);
          h_e1x5e5x5->Fill(myMVAVar_e1x5e5x5);
          h_R9->Fill(myMVAVar_R9);
          h_nbrems->Fill(myMVAVar_nbrems);

          h_HoE->Fill(myMVAVar_HoE);
          h_EoP->Fill(myMVAVar_EoP);
          h_IoEmIoP->Fill(myMVAVar_IoEmIoP);
          h_eleEoPout->Fill(myMVAVar_eleEoPout);
          h_EoPout->Fill(myMVAVar_EoPout);
          h_PreShowerOverRaw->Fill(myMVAVar_PreShowerOverRaw);
          h_pt->Fill(myMVAVar_pt);

          if (debug)
            std::cout << "************************* New Good Event:: " << ev << " *************************"
                      << std::endl;

          // ********************* Triggering electrons

          bool elePresel = trainTrigPresel(theEGamma[j]);

          double mvaTrigMthd1 = -11.;
          double mvaTrigMthd2 = -11.;

          double mvaTrigNonIp = -11.;

          double mvaNonTrigMthd1 = -11;
          double mvaNonTrigMthd2 = -11;

          mvaNonTrigMthd1 = myMVANonTrigV0->mvaValue((theEGamma[j]), *pv, thebuilder, lazyTools, debugMVAclass);
          mvaNonTrigMthd2 = myMVANonTrigV0->mvaValue(myMVAVar_fbrem,
                                                     myMVAVar_kfchi2,
                                                     myMVAVar_kfhits,
                                                     myMVAVar_gsfchi2,
                                                     myMVAVar_deta,
                                                     myMVAVar_dphi,
                                                     myMVAVar_detacalo,
                                                     // myMVAVar_dphicalo,
                                                     myMVAVar_see,
                                                     myMVAVar_spp,
                                                     myMVAVar_etawidth,
                                                     myMVAVar_phiwidth,
                                                     myMVAVar_e1x5e5x5,
                                                     myMVAVar_R9,
                                                     //myMVAVar_nbrems,
                                                     myMVAVar_HoE,
                                                     myMVAVar_EoP,
                                                     myMVAVar_IoEmIoP,
                                                     myMVAVar_eleEoPout,
                                                     myMVAVar_PreShowerOverRaw,
                                                     // myMVAVar_EoPout,
                                                     myMVAVar_eta,
                                                     myMVAVar_pt,
                                                     debugMyVar);
          h_mva_nonTrig->Fill(mvaNonTrigMthd1);

          if (debug)
            std::cout << "Non-Triggering:: MyMVA Method-1 " << mvaNonTrigMthd1 << " MyMVA Method-2 " << mvaNonTrigMthd2
                      << std::endl;

          if (elePresel) {
            mvaTrigNonIp =
                myMVATrigNoIPV0->mvaValue((theEGamma[j]), *pv, _Rho, /*thebuilder,*/ lazyTools, debugMVAclass);

            mvaTrigMthd1 = myMVATrigV0->mvaValue((theEGamma[j]), *pv, thebuilder, lazyTools, debugMVAclass);
            mvaTrigMthd2 = myMVATrigV0->mvaValue(myMVAVar_fbrem,
                                                 myMVAVar_kfchi2,
                                                 myMVAVar_kfhits,
                                                 myMVAVar_gsfchi2,
                                                 myMVAVar_deta,
                                                 myMVAVar_dphi,
                                                 myMVAVar_detacalo,
                                                 // myMVAVar_dphicalo,
                                                 myMVAVar_see,
                                                 myMVAVar_spp,
                                                 myMVAVar_etawidth,
                                                 myMVAVar_phiwidth,
                                                 myMVAVar_e1x5e5x5,
                                                 myMVAVar_R9,
                                                 //myMVAVar_nbrems,
                                                 myMVAVar_HoE,
                                                 myMVAVar_EoP,
                                                 myMVAVar_IoEmIoP,
                                                 myMVAVar_eleEoPout,
                                                 myMVAVar_PreShowerOverRaw,
                                                 // myMVAVar_EoPout,
                                                 myMVAVar_d0,
                                                 myMVAVar_ip3d,
                                                 myMVAVar_eta,
                                                 myMVAVar_pt,
                                                 debugMyVar);
            h_mva_trig->Fill(mvaTrigMthd1);
            h_mva_trigNonIp->Fill(mvaTrigNonIp);
          }

          if (debug)
            std::cout << "Triggering:: ElePreselection " << elePresel << " MyMVA Method-1 " << mvaTrigMthd1
                      << " MyMVA Method-2 " << mvaTrigMthd2 << std::endl;
        }
      }  // End Loop on RECO electrons
    }    // End if MC electrons selection
  }      //End Loop Generator Particles
}
// ------------ method called once each job just before starting event loop  ------------
void ElectronTestAnalyzer::myVar(const reco::GsfElectron& ele,
                                 const reco::Vertex& vertex,
                                 const TransientTrackBuilder& transientTrackBuilder,
                                 EcalClusterLazyTools const& myEcalCluster,
                                 bool printDebug) {
  bool validKF = false;
  reco::TrackRef myTrackRef = ele.closestCtfTrackRef();
  validKF = (myTrackRef.isAvailable());
  validKF = (myTrackRef.isNonnull());

  myMVAVar_fbrem = ele.fbrem();
  myMVAVar_kfchi2 = (validKF) ? myTrackRef->normalizedChi2() : 0;
  myMVAVar_kfhits = (validKF) ? myTrackRef->hitPattern().trackerLayersWithMeasurement() : -1.;
  //  myMVAVar_kfhits          =  (validKF) ? myTrackRef->numberOfValidHits() : -1. ;   // for analysist save also this
  myMVAVar_gsfchi2 = ele.gsfTrack()->normalizedChi2();  // to be checked

  myMVAVar_deta = ele.deltaEtaSuperClusterTrackAtVtx();
  myMVAVar_dphi = ele.deltaPhiSuperClusterTrackAtVtx();
  myMVAVar_detacalo = ele.deltaEtaSeedClusterTrackAtCalo();
  myMVAVar_dphicalo = ele.deltaPhiSeedClusterTrackAtCalo();

  myMVAVar_see = ele.sigmaIetaIeta();  //EleSigmaIEtaIEta
  const auto& vCov = myEcalCluster.localCovariances(*(ele.superCluster()->seed()));
  if (edm::isFinite(vCov[2]))
    myMVAVar_spp = sqrt(vCov[2]);  //EleSigmaIPhiIPhi
  else
    myMVAVar_spp = 0.;
  myMVAVar_etawidth = ele.superCluster()->etaWidth();
  myMVAVar_phiwidth = ele.superCluster()->phiWidth();
  myMVAVar_e1x5e5x5 = (ele.e5x5()) != 0. ? 1. - (ele.e1x5() / ele.e5x5()) : -1.;
  myMVAVar_R9 = myEcalCluster.e3x3(*(ele.superCluster()->seed())) / ele.superCluster()->rawEnergy();
  myMVAVar_nbrems = std::abs(ele.numberOfBrems());

  myMVAVar_HoE = ele.hadronicOverEm();
  myMVAVar_EoP = ele.eSuperClusterOverP();
  myMVAVar_IoEmIoP =
      (1.0 / ele.ecalEnergy()) - (1.0 / ele.p());  // in the future to be changed with ele.gsfTrack()->p()
  myMVAVar_eleEoPout = ele.eEleClusterOverPout();
  myMVAVar_EoPout = ele.eSeedClusterOverPout();
  myMVAVar_PreShowerOverRaw = ele.superCluster()->preshowerEnergy() / ele.superCluster()->rawEnergy();

  myMVAVar_eta = ele.superCluster()->eta();
  myMVAVar_pt = ele.pt();

  //d0
  if (ele.gsfTrack().isNonnull()) {
    myMVAVar_d0 = (-1.0) * ele.gsfTrack()->dxy(vertex.position());
  } else if (ele.closestCtfTrackRef().isNonnull()) {
    myMVAVar_d0 = (-1.0) * ele.closestCtfTrackRef()->dxy(vertex.position());
  } else {
    myMVAVar_d0 = -9999.0;
  }

  //default values for IP3D
  myMVAVar_ip3d = -999.0;
  // myMVAVar_ip3dSig = 0.0;
  if (ele.gsfTrack().isNonnull()) {
    const double gsfsign = ((-ele.gsfTrack()->dxy(vertex.position())) >= 0) ? 1. : -1.;

    const reco::TransientTrack& tt = transientTrackBuilder.build(ele.gsfTrack());
    const std::pair<bool, Measurement1D>& ip3dpv = IPTools::absoluteImpactParameter3D(tt, vertex);
    if (ip3dpv.first) {
      double ip3d = gsfsign * ip3dpv.second.value();
      //double ip3derr = ip3dpv.second.error();
      myMVAVar_ip3d = ip3d;
      // myMVAVar_ip3dSig = ip3d/ip3derr;
    }
  }

  if (printDebug) {
    std::cout << " My Local Variables " << std::endl;
    std::cout << " fbrem " << myMVAVar_fbrem << " kfchi2 " << myMVAVar_kfchi2 << " mykfhits " << myMVAVar_kfhits
              << " gsfchi2 " << myMVAVar_gsfchi2 << " deta " << myMVAVar_deta << " dphi " << myMVAVar_dphi
              << " detacalo " << myMVAVar_detacalo << " dphicalo " << myMVAVar_dphicalo << " see " << myMVAVar_see
              << " spp " << myMVAVar_spp << " etawidth " << myMVAVar_etawidth << " phiwidth " << myMVAVar_phiwidth
              << " e1x5e5x5 " << myMVAVar_e1x5e5x5 << " R9 " << myMVAVar_R9 << " mynbrems " << myMVAVar_nbrems
              << " HoE " << myMVAVar_HoE << " EoP " << myMVAVar_EoP << " IoEmIoP " << myMVAVar_IoEmIoP << " eleEoPout "
              << myMVAVar_eleEoPout << " EoPout " << myMVAVar_EoPout << " PreShowerOverRaw "
              << myMVAVar_PreShowerOverRaw << " d0 " << myMVAVar_d0 << " ip3d " << myMVAVar_ip3d << " eta "
              << myMVAVar_eta << " pt " << myMVAVar_pt << std::endl;
  }
  return;
}
void ElectronTestAnalyzer::myBindVariables() {
  // this binding is needed for variables that sometime diverge.

  if (myMVAVar_fbrem < -1.)
    myMVAVar_fbrem = -1.;

  myMVAVar_deta = std::abs(myMVAVar_deta);
  if (myMVAVar_deta > 0.06)
    myMVAVar_deta = 0.06;

  myMVAVar_dphi = std::abs(myMVAVar_dphi);
  if (myMVAVar_dphi > 0.6)
    myMVAVar_dphi = 0.6;

  if (myMVAVar_EoPout > 20.)
    myMVAVar_EoPout = 20.;

  if (myMVAVar_EoP > 20.)
    myMVAVar_EoP = 20.;

  if (myMVAVar_eleEoPout > 20.)
    myMVAVar_eleEoPout = 20.;

  myMVAVar_detacalo = std::abs(myMVAVar_detacalo);
  if (myMVAVar_detacalo > 0.2)
    myMVAVar_detacalo = 0.2;

  myMVAVar_dphicalo = std::abs(myMVAVar_dphicalo);
  if (myMVAVar_dphicalo > 0.4)
    myMVAVar_dphicalo = 0.4;

  if (myMVAVar_e1x5e5x5 < -1.)
    myMVAVar_e1x5e5x5 = -1;

  if (myMVAVar_e1x5e5x5 > 2.)
    myMVAVar_e1x5e5x5 = 2.;

  if (myMVAVar_R9 > 5)
    myMVAVar_R9 = 5;

  if (myMVAVar_gsfchi2 > 200.)
    myMVAVar_gsfchi2 = 200;

  if (myMVAVar_kfchi2 > 10.)
    myMVAVar_kfchi2 = 10.;

  // Needed for a bug in CMSSW_420, fixed in more recent CMSSW versions
  if (edm::isNotFinite(myMVAVar_spp))
    myMVAVar_spp = 0.;

  return;
}
bool ElectronTestAnalyzer::trainTrigPresel(const reco::GsfElectron& ele) {
  bool myTrigPresel = false;
  if (std::abs(ele.superCluster()->eta()) < 1.479) {
    if (ele.sigmaIetaIeta() < 0.014 && ele.hadronicOverEm() < 0.15 && ele.dr03TkSumPt() / ele.pt() < 0.2 &&
        ele.dr03EcalRecHitSumEt() / ele.pt() < 0.2 && ele.dr03HcalTowerSumEt() / ele.pt() < 0.2 &&
        ele.gsfTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) == 0)
      myTrigPresel = true;
  } else {
    if (ele.sigmaIetaIeta() < 0.035 && ele.hadronicOverEm() < 0.10 && ele.dr03TkSumPt() / ele.pt() < 0.2 &&
        ele.dr03EcalRecHitSumEt() / ele.pt() < 0.2 && ele.dr03HcalTowerSumEt() / ele.pt() < 0.2 &&
        ele.gsfTrack()->hitPattern().numberOfLostHits(reco::HitPattern::MISSING_INNER_HITS) == 0)
      myTrigPresel = true;
  }

  return myTrigPresel;
}
void ElectronTestAnalyzer::beginJob() { ev = 0; }

// ------------ method called once each job just after ending the event loop  ------------
void ElectronTestAnalyzer::endJob() { std::cout << " endJob:: #events " << ev << std::endl; }

void ElectronTestAnalyzer::evaluate_mvas(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<reco::VertexCollection> hVertex;
  iEvent.getByToken(vertexToken_, hVertex);
  const reco::VertexCollection* pvCol = hVertex.product();

  // FIXME: unused variable giving compilation warnings/errors
  //   Handle<double> hRho;
  //   iEvent.getByToken(eventrhoToken_,hRho);
  //   double Rho = *hRho;
  //
  //   Handle<reco::PFCandidateCollection> hPfCandProduct;
  // 	iEvent.getByToken(pfCandToken_, hPfCandProduct);
  //   const reco::PFCandidateCollection &inPfCands = *(hPfCandProduct.product());

  edm::Handle<reco::GsfElectronCollection> theEGammaCollection;
  iEvent.getByToken(gsfEleToken_, theEGammaCollection);
  const reco::GsfElectronCollection inElectrons = *(theEGammaCollection.product());

  edm::Handle<reco::MuonCollection> hMuonProduct;
  iEvent.getByToken(muonToken_, hMuonProduct);
  const reco::MuonCollection inMuons = *(hMuonProduct.product());

  reco::MuonCollection IdentifiedMuons;
  reco::GsfElectronCollection IdentifiedElectrons;

  for (reco::GsfElectronCollection::const_iterator iE = inElectrons.begin(); iE != inElectrons.end(); ++iE) {
    double electronTrackZ = 0;
    if (iE->gsfTrack().isNonnull()) {
      electronTrackZ = iE->gsfTrack()->dz(pvCol->at(0).position());
    } else if (iE->closestCtfTrackRef().isNonnull()) {
      electronTrackZ = iE->closestCtfTrackRef()->dz(pvCol->at(0).position());
    }
    if (std::abs(electronTrackZ) > 0.2)
      continue;

    if (std::abs(iE->superCluster()->eta()) < 1.479) {
      if (iE->pt() > 20) {
        if (iE->sigmaIetaIeta() > 0.01)
          continue;
        if (std::abs(iE->deltaEtaSuperClusterTrackAtVtx()) > 0.007)
          continue;
        if (std::abs(iE->deltaPhiSuperClusterTrackAtVtx()) > 0.8)
          continue;
        if (iE->hadronicOverEm() > 0.15)
          continue;
      } else {
        if (iE->sigmaIetaIeta() > 0.012)
          continue;
        if (std::abs(iE->deltaEtaSuperClusterTrackAtVtx()) > 0.007)
          continue;
        if (std::abs(iE->deltaPhiSuperClusterTrackAtVtx()) > 0.8)
          continue;
        if (iE->hadronicOverEm() > 0.15)
          continue;
      }
    } else {
      if (iE->pt() > 20) {
        if (iE->sigmaIetaIeta() > 0.03)
          continue;
        if (std::abs(iE->deltaEtaSuperClusterTrackAtVtx()) > 0.010)
          continue;
        if (std::abs(iE->deltaPhiSuperClusterTrackAtVtx()) > 0.8)
          continue;
      } else {
        if (iE->sigmaIetaIeta() > 0.032)
          continue;
        if (std::abs(iE->deltaEtaSuperClusterTrackAtVtx()) > 0.010)
          continue;
        if (std::abs(iE->deltaPhiSuperClusterTrackAtVtx()) > 0.8)
          continue;
      }
    }
    IdentifiedElectrons.push_back(*iE);
  }

  for (reco::MuonCollection::const_iterator iM = inMuons.begin(); iM != inMuons.end(); ++iM) {
    if (!(iM->innerTrack().isNonnull())) {
      continue;
    }

    if (!(iM->isGlobalMuon() || iM->isTrackerMuon()))
      continue;
    if (iM->innerTrack()->numberOfValidHits() < 11)
      continue;

    IdentifiedMuons.push_back(*iM);
  }

  EcalClusterLazyTools lazyTools(
      iEvent, ecalClusterESGetTokens_.get(iSetup), reducedEBRecHitCollectionToken_, reducedEERecHitCollectionToken_);

  TransientTrackBuilder thebuilder = iSetup.getData(builderToken_);

  for (reco::GsfElectronCollection::const_iterator iE = inElectrons.begin(); iE != inElectrons.end(); ++iE) {
    reco::GsfElectron ele = *iE;

    double idmva = myMVATrigV0->mvaValue(ele, pvCol->at(0), thebuilder, lazyTools);

    std::cout << "idmva = " << idmva << std::endl;
  }
}

//define this as a plug-in
DEFINE_FWK_MODULE(ElectronTestAnalyzer);
