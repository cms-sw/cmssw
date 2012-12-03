#include "DQM/Physics/src/ExoticaDQM.h"

#include <memory>

// DQM
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

// Framework
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Run.h"
#include "DataFormats/Provenance/interface/EventID.h"

// Candidate handling
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/OverlapChecker.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"
#include "DataFormats/Candidate/interface/CompositeCandidateFwd.h"
#include "DataFormats/Candidate/interface/CandMatchMap.h"

// Vertex utilities
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

// Other
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Common/interface/RefToBase.h"

// Math
#include "DataFormats/Math/interface/Vector3D.h"
#include "DataFormats/Math/interface/LorentzVector.h"
#include "DataFormats/GeometryVector/interface/GlobalVector.h"
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Math/interface/Point3D.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Math/interface/deltaPhi.h"

// vertexing

// Transient tracks
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "TrackingTools/TransientTrack/interface/TransientTrackBuilder.h"
#include "TrackingTools/Records/interface/TransientTrackRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/Records/interface/MuonGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

//JetCorrection
#include "JetMETCorrections/Objects/interface/JetCorrector.h"

// ROOT
#include "TLorentzVector.h"

// STDLIB
#include <iostream>
#include <iomanip>
#include <stdio.h>
#include <string>
#include <sstream>
#include <math.h>

using namespace edm;
using namespace std;
using namespace reco; 
using namespace trigger;

typedef vector<string> vstring;

struct SortCandByDecreasingPt {
  bool operator()( const Candidate &c1, const Candidate &c2) const {
    return c1.pt() > c2.pt();
  }
};


//
// -- Constructor
//
ExoticaDQM::ExoticaDQM(const edm::ParameterSet& ps){

  edm::LogInfo("ExoticaDQM") <<  " Starting ExoticaDQM " << "\n" ;

  bei_ = Service<DQMStore>().operator->();
  bei_->setCurrentFolder("Physics/Exotica");
  bookHistos(bei_);

  typedef std::vector<edm::InputTag> vtag;

  // Get parameters from configuration file
  // Trigger
  theTriggerResultsCollection = ps.getParameter<InputTag>("triggerResultsCollection");
  //
  theTriggerForMultiJetsList  = ps.getParameter<vstring>("triggerMultiJetsList");
  theTriggerForLongLivedList  = ps.getParameter<vstring>("triggerLongLivedList");
  
  //
  ElectronLabel_      = ps.getParameter<InputTag>("electronCollection");
  PFElectronLabelEI_  = ps.getParameter<InputTag>("pfelectronCollectionEI");
  //
  MuonLabel_          = ps.getParameter<InputTag>("muonCollection"); 
  PFMuonLabelEI_      = ps.getParameter<InputTag>("pfmuonCollectionEI");
  //
  TauLabel_           = ps.getParameter<InputTag>("tauCollection");
  //PFTauLabel_       = ps.getParameter<InputTag>("pftauCollection");
  //
  PhotonLabel_        = ps.getParameter<InputTag>("photonCollection");
  //PFPhotonLabel_    = ps.getParameter<InputTag>("pfphotonCollection");
  //
  CaloJetLabel_       = ps.getParameter<InputTag>("caloJetCollection");
  PFJetLabel_         = ps.getParameter<InputTag>("pfJetCollection");
  PFJetLabelEI_       = ps.getParameter<InputTag>("pfJetCollectionEI");
  
  //
  CaloMETLabel_       = ps.getParameter<InputTag>("caloMETCollection");
  PFMETLabel_         = ps.getParameter<InputTag>("pfMETCollection");
  PFMETLabelEI_       = ps.getParameter<InputTag>("pfMETCollectionEI");
  
  //Cuts - MultiJets 
  jetID                    = new reco::helper::JetIDHelper(ps.getParameter<ParameterSet>("JetIDParams"));
  mj_monojet_ptPFJet_      = ps.getParameter<double>("mj_monojet_ptPFJet");
  mj_monojet_ptPFMuon_     = ps.getParameter<double>("mj_monojet_ptPFMuon");
  mj_monojet_ptPFElectron_ = ps.getParameter<double>("mj_monojet_ptPFElectron");
  CaloJetCorService_       = ps.getParameter<std::string>("CaloJetCorService");
  PFJetCorService_         = ps.getParameter<std::string>("PFJetCorService");

  // just to initialize
  //isValidHltConfig_ = false;
}


//
// -- Destructor
//
ExoticaDQM::~ExoticaDQM(){
  edm::LogInfo("ExoticaDQM") <<  " Deleting ExoticaDQM " << "\n" ;
}


//
// -- Begin Job
//
void ExoticaDQM::beginJob(){
  nLumiSecs_ = 0;
  nEvents_   = 0;
  pi = 3.14159265;
}


//
// -- Begin Run
//
void ExoticaDQM::beginRun(Run const& run, edm::EventSetup const& eSetup) {
  edm::LogInfo ("ExoticaDQM") <<"[ExoticaDQM]: Begining of Run";
  
  // passed as parameter to HLTConfigProvider::init(), not yet used
  bool isConfigChanged = false;
  
  // isValidHltConfig_ used to short-circuit analyze() in case of problems
  //  const std::string hltProcessName( "HLT" );
  const std::string hltProcessName = theTriggerResultsCollection.process();
  isValidHltConfig_ = hltConfigProvider_.init( run, eSetup, hltProcessName, isConfigChanged );

}


//
// -- Begin  Luminosity Block
//
void ExoticaDQM::beginLuminosityBlock(edm::LuminosityBlock const& lumiSeg, 
                                            edm::EventSetup const& context) { 
  //edm::LogInfo ("ExoticaDQM") <<"[ExoticaDQM]: Begin of LS transition";
}


//
//  -- Book histograms
//
void ExoticaDQM::bookHistos(DQMStore* bei){
 
  bei->cd();
  
  //--- Multijets
  bei->setCurrentFolder("Physics/Exotica/MultiJets");
  mj_monojet_pfchef               = bei->book1D("mj_monojet_pfchef", "PFJetID CHEF", 50, 0.0 , 1.0);
  mj_monojet_pfnhef               = bei->book1D("mj_monojet_pfnhef", "PFJetID NHEF", 50, 0.0 , 1.0);
  mj_monojet_pfcemf               = bei->book1D("mj_monojet_pfcemf", "PFJetID CEMF", 50, 0.0 , 1.0);
  mj_monojet_pfnemf               = bei->book1D("mj_monojet_pfnemf", "PFJetID NEMF", 50, 0.0 , 1.0);
  mj_monojet_pfJet1_pt            = bei->book1D("mj_monojet_pfJet1_pt", "Pt of PFJet-1 (GeV)", 40, 0.0 , 1000);
  mj_monojet_pfJet2_pt            = bei->book1D("mj_monojet_pfJet2_pt", "Pt of PFJet-2 (GeV)", 40, 0.0 , 1000);
  mj_monojet_pfJet1_eta           = bei->book1D("mj_monojet_pfJet1_eta", "#eta(PFJet-1)", 50, -5.0, 5.0);
  mj_monojet_pfJet2_eta           = bei->book1D("mj_monojet_pfJet2_eta", "#eta(PFJet-2)", 50, -5.0, 5.0);
  mj_monojet_pfJetMulti           = bei->book1D("mj_monojet_pfJetMulti", "No. of PFJets", 10, 0., 10.);
  mj_monojet_deltaPhiPFJet1PFJet2 = bei->book1D("mj_monojet_deltaPhiPFJet1PFJet2", "#Delta#phi(PFJet1, PFJet2)", 40, 0., 4.);
  mj_monojet_deltaRPFJet1PFJet2   = bei->book1D("mj_monojet_deltaRPFJet1PFJet2", "#DeltaR(PFJet1, PFJet2)", 50, 0., 10.);
  mj_monojet_pfmetnomu            = bei->book1D("mj_monojet_pfmetnomu", "PFMET no Mu", 100, 0.0 , 500.0); 
  mj_caloMet_et                   = bei->book1D("mj_caloMet", "Calo Missing E_{T}; GeV", 50, 0.0 , 500);
  mj_caloMet_phi                  = bei->book1D("mj_caloMet_phi", "Calo Missing E_{T} #phi;#phi(MET)", 35, -3.5, 3.5 );
  mj_pfMet_et                     = bei->book1D("mj_pfMet", "Pf Missing E_{T}; GeV", 50,  0.0 , 500);
  mj_pfMet_phi                    = bei->book1D("mj_pfMet_phi", "Pf Missing E_{T} #phi;#phi(MET)", 35, -3.5, 3.5 );
  
  //
  //bei->setCurrentFolder("Physics/Exotica/MultiJetsTrigger"); 
 
  //--- LongLived
  bei->setCurrentFolder("Physics/Exotica/LongLived");
  ll_gammajet_sMajMajPhot         = bei->book1D("ll_gammajet_sMajMajPhot", "sMajMajPhot", 50, 0.0 , 5.0);
  ll_gammajet_sMinMinPhot         = bei->book1D("ll_gammajet_sMinMinPhot", "sMinMinPhot", 50, 0.0 , 5.0);

  //
  //bei->setCurrentFolder("Physics/Exotica/LongLivedTrigger"); 

  //
  bei->setCurrentFolder("Physics/Exotica/EIComparison");
  ei_pfjet1_pt     = bei->book1D("ei_pfjet1_pt",     "Pt of PFJet-1    (EI) (GeV)", 40, 0.0 , 1000);
  ei_pfmet_pt      = bei->book1D("ei_pfmet_pt",      "Pt of PFMET      (EI) (GeV)", 40, 0.0 , 1000);
  //ei_pfmuon_pt     = bei->book1D("ei_pfmuon_pt",     "Pt of PFMuon     (EI) (GeV)", 40, 0.0 , 1000);
  //ei_pfelectron_pt = bei->book1D("ei_pfelectron_pt", "Pt of PFElectron (EI) (GeV)", 40, 0.0 , 1000);
  
  bei->cd();
}


//
//  -- Analyze 
//
void ExoticaDQM::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){ 

  // Calo objects
  // Electrons
  bool ValidCaloElectron = iEvent.getByLabel(ElectronLabel_, ElectronCollection_);
  if(!ValidCaloElectron) return;
  // Muons
  bool ValidCaloMuon = iEvent.getByLabel(MuonLabel_, MuonCollection_);
  if(!ValidCaloMuon) return;
  // Taus
  bool ValidCaloTau = iEvent.getByLabel(TauLabel_, TauCollection_);
  if(!ValidCaloTau) return;  
  // Photons
  bool ValidCaloPhoton = iEvent.getByLabel(PhotonLabel_, PhotonCollection_);
  if(!ValidCaloPhoton) return; 
  // Jets
  bool ValidCaloJet = iEvent.getByLabel(CaloJetLabel_, caloJetCollection_);
  if(!ValidCaloJet) return;
  calojets = *caloJetCollection_; 
  // MET
  bool ValidCaloMET = iEvent.getByLabel(CaloMETLabel_, caloMETCollection_);
  if(!ValidCaloMET) return;
  

  // PF objects
  // PFJets
  bool ValidPFJet = iEvent.getByLabel(PFJetLabel_, pfJetCollection_);
  if(!ValidPFJet) return;
  pfjets = *pfJetCollection_; 
  // PFMETs
  bool ValidPFMET = iEvent.getByLabel(PFMETLabel_, pfMETCollection_);
  if(!ValidPFMET) return;
  
  //#######################################################
  // Jet Correction
  // Define on-the-fly correction Jet
  for(int i=0; i<2; i++){
    CaloJetPx[i]   = 0.;
    CaloJetPy[i]   = 0.;
    CaloJetPt[i]   = 0.;
    CaloJetEta[i]  = 0.;
    CaloJetPhi[i]  = 0.;
    CaloJetEMF[i]  = 0.;
    CaloJetfHPD[i] = 0.;
    CaloJetn90[i]  = 0.;
    PFJetPx[i]     = 0.;
    PFJetPy[i]     = 0.;
    PFJetPt[i]     = 0.;
    PFJetEta[i]    = 0.;
    PFJetPhi[i]    = 0.;
    PFJetNHEF[i]   = 0.;
    PFJetCHEF[i]   = 0.;
    PFJetNEMF[i]   = 0.;
    PFJetCEMF[i]   = 0.;
  }
  
  //---------- CaloJet Correction (on-the-fly) ----------
  const JetCorrector* calocorrector = JetCorrector::getJetCorrector(CaloJetCorService_,iSetup);
  CaloJetCollection::const_iterator calojet_ = calojets.begin();
  for(; calojet_ != calojets.end(); ++calojet_){
    double scale = calocorrector->correction(*calojet_,iEvent, iSetup);	
    jetID->calculate(iEvent, *calojet_);
    
    if(scale*calojet_->pt()>CaloJetPt[0]){
      CaloJetPt[1]   = CaloJetPt[0]; 
      CaloJetPx[1]   = CaloJetPx[0];
      CaloJetPy[1]   = CaloJetPy[0];
      CaloJetEta[1]  = CaloJetEta[0];
      CaloJetPhi[1]  = CaloJetPhi[0];
      CaloJetEMF[1]  = CaloJetEMF[0];
      CaloJetfHPD[1] = CaloJetfHPD[0];
      CaloJetn90[1]  = CaloJetn90[0];
      //
      CaloJetPt[0]   = scale*calojet_->pt();
      CaloJetPx[0]   = scale*calojet_->px();
      CaloJetPy[0]   = scale*calojet_->py();
      CaloJetEta[0]  = calojet_->eta();
      CaloJetPhi[0]  = calojet_->phi();
      CaloJetEMF[0]  = calojet_->emEnergyFraction();
      CaloJetfHPD[0] = jetID->fHPD();
      CaloJetn90[0]  = jetID->n90Hits();
    }
    else if(scale*calojet_->pt()<CaloJetPt[0] && scale*calojet_->pt()>CaloJetPt[1] ){
      CaloJetPt[1]   = scale*calojet_->pt();
      CaloJetPx[1]   = scale*calojet_->px();
      CaloJetPy[1]   = scale*calojet_->py();
      CaloJetEta[1]  = calojet_->eta();
      CaloJetPhi[1]  = calojet_->phi();
      CaloJetEMF[1]  = calojet_->emEnergyFraction();
      CaloJetfHPD[1] = jetID->fHPD();
      CaloJetn90[1]  = jetID->n90Hits();
    }
    else{}
  }
  
  //
  mj_monojet_countPFJet=0;
  const JetCorrector* pfcorrector = JetCorrector::getJetCorrector(PFJetCorService_,iSetup);
  PFJetCollection::const_iterator pfjet_ = pfjets.begin();
  for(; pfjet_ != pfjets.end(); ++pfjet_){
    double scale = pfcorrector->correction(*pfjet_,iEvent, iSetup);
    if(scale*pfjet_->pt()>PFJetPt[0]){
      PFJetPt[1]   = PFJetPt[0];
      PFJetPx[1]   = PFJetPx[0];
      PFJetPy[1]   = PFJetPy[0];
      PFJetEta[1]  = PFJetEta[0];
      PFJetPhi[1]  = PFJetPhi[0];
      PFJetNHEF[1] = PFJetNHEF[0]; 
      PFJetCHEF[1] = PFJetCHEF[0];
      PFJetNEMF[1] = PFJetNEMF[0]; 
      PFJetCEMF[1] = PFJetCEMF[0];
      //
      PFJetPt[0]   = scale*pfjet_->pt();
      PFJetPx[0]   = scale*pfjet_->px();
      PFJetPy[0]   = scale*pfjet_->py();
      PFJetEta[0]  = pfjet_->eta();
      PFJetPhi[0]  = pfjet_->phi();
      PFJetNHEF[0] = pfjet_->neutralHadronEnergyFraction();
      PFJetCHEF[0] = pfjet_->chargedHadronEnergyFraction();
      PFJetNEMF[0] = pfjet_->neutralEmEnergyFraction();
      PFJetCEMF[0] = pfjet_->chargedEmEnergyFraction();
    }
    else if(scale*pfjet_->pt()<PFJetPt[0] && scale*pfjet_->pt()>PFJetPt[1] ){
      PFJetPt[1]   = scale*pfjet_->pt();
      PFJetPx[1]   = scale*pfjet_->px();
      PFJetPy[1]   = scale*pfjet_->py();
      PFJetEta[1]  = pfjet_->eta();
      PFJetPhi[1]  = pfjet_->phi(); 
      PFJetNHEF[1] = pfjet_->neutralHadronEnergyFraction();
      PFJetCHEF[1] = pfjet_->chargedHadronEnergyFraction();
      PFJetNEMF[1] = pfjet_->neutralEmEnergyFraction();
      PFJetCEMF[1] = pfjet_->chargedEmEnergyFraction();
    }
    else{}
    if(scale*pfjet_->pt()>mj_monojet_ptPFJet_) mj_monojet_countPFJet++;
  }
  //#######################################################
  
  
  // Analyze
  //
  analyzeMultiJets(iEvent);
  //analyzeMultiJetsTrigger(iEvent);
  //
  analyzeLongLived(iEvent);
  //analyzeLongLivedTrigger(iEvent);

  analyzeEventInterpretation(iEvent, iSetup);
}

void ExoticaDQM::analyzeMultiJets(const Event & iEvent){ 

  //--- MonoJet
  //bool checkLepton = false;
  //reco::MuonCollection::const_iterator  muon  = MuonCollection_->begin();
  //for(; muon != MuonCollection_->end(); muon++){
  //if(muon->pt()<mj_monojet_ptPFMuon_) continue; 
  //checkLepton = true; 
  //}
  //reco::GsfElectronCollection::const_iterator electron = ElectronCollection_->begin();
  //for(; electron != ElectronCollection_->end(); electron++){
  //if(electron->pt()<mj_monojet_ptPFElectron_) continue; 
  //checkLepton = true; 
  //}
  //if(checkLepton==false){

  if(PFJetPt[0]>0.){
    mj_monojet_pfJet1_pt->Fill(PFJetPt[0]);
    mj_monojet_pfJet1_eta->Fill(PFJetEta[0]);
    mj_monojet_pfchef->Fill(PFJetCHEF[0]);
    mj_monojet_pfnhef->Fill(PFJetNHEF[0]); 
    mj_monojet_pfcemf->Fill(PFJetCEMF[0]);
    mj_monojet_pfnemf->Fill(PFJetNEMF[0]);
    mj_monojet_pfJetMulti->Fill(mj_monojet_countPFJet); 
  }
  if(PFJetPt[1]>0.){
    mj_monojet_pfJet2_pt->Fill(PFJetPt[1]);
    mj_monojet_pfJet2_eta->Fill(PFJetEta[1]);
    mj_monojet_deltaPhiPFJet1PFJet2->Fill(deltaPhi(PFJetPhi[0],PFJetPhi[1]));
    mj_monojet_deltaRPFJet1PFJet2->Fill(deltaR(PFJetEta[0],PFJetPhi[0],
					       PFJetEta[1],PFJetPhi[1]));
  }
  
  //--- MET
  const CaloMETCollection *calometcol = caloMETCollection_.product();
  const CaloMET met = calometcol->front();
  mj_caloMet_et->Fill(met.et());
  mj_caloMet_phi->Fill(met.phi());
  
  //
  const PFMETCollection *pfmetcol = pfMETCollection_.product();
  const PFMET pfmet = pfmetcol->front();
  mj_pfMet_et->Fill(pfmet.et());
  mj_pfMet_phi->Fill(pfmet.phi());
}

void ExoticaDQM::analyzeMultiJetsTrigger(const Event & iEvent){
}

void ExoticaDQM::analyzeLongLived(const Event & iEvent){ 
  // SMajMajPho, SMinMinPho
  // get ECAL reco hits
  Handle<EBRecHitCollection> ecalhitseb;
  const EBRecHitCollection* rhitseb=0;
  iEvent.getByLabel("reducedEcalRecHitsEB", ecalhitseb);    
  rhitseb = ecalhitseb.product(); // get a ptr to the product
  //
  Handle<EERecHitCollection> ecalhitsee;
  const EERecHitCollection* rhitsee=0;
  iEvent.getByLabel("reducedEcalRecHitsEE", ecalhitsee);
  rhitsee = ecalhitsee.product(); // get a ptr to the product
  //
  int nPhot = 0;
  reco::PhotonCollection::const_iterator photon = PhotonCollection_->begin();
  for(; photon != PhotonCollection_->end(); ++photon){
    if(photon->energy()<3.) continue;
    if(nPhot>=40) continue;
    
    const Ptr<CaloCluster> theSeed = photon->superCluster()->seed(); 
    const EcalRecHitCollection* rechits = ( photon->isEB()) ? rhitseb : rhitsee;
    CaloClusterPtr SCseed = photon->superCluster()->seed();
    
    std::pair<DetId, float> maxRH = EcalClusterTools::getMaximum( *theSeed, &(*rechits) );
    
    if(maxRH.second) {
      Cluster2ndMoments moments = EcalClusterTools::cluster2ndMoments(*SCseed, *rechits);
      //std::vector<float> etaphimoments = EcalClusterTools::localCovariances(*SCseed, &(*rechits), &(*topology));
      ll_gammajet_sMajMajPhot->Fill(moments.sMaj);
      ll_gammajet_sMinMinPhot->Fill(moments.sMin);
    }
    else{
      ll_gammajet_sMajMajPhot->Fill(-100.);
      ll_gammajet_sMinMinPhot->Fill(-100.);
    }
    ++nPhot;
  }

}

void ExoticaDQM::analyzeLongLivedTrigger(const Event & iEvent){
}

void ExoticaDQM::analyzeEventInterpretation(const Event & iEvent, const edm::EventSetup& iSetup){  

  // EI
  // PFElectrons
  bool ValidPFElectronEI = iEvent.getByLabel(PFElectronLabelEI_, pfElectronCollectionEI_);
  if(!ValidPFElectronEI) return;
  pfelectronsEI = *pfElectronCollectionEI_;

  // PFMuons
  bool ValidPFMuonEI = iEvent.getByLabel(PFMuonLabelEI_, pfMuonCollectionEI_);
  if(!ValidPFMuonEI) return;
  pfmuonsEI = *pfMuonCollectionEI_;
  
  // PFJets
  bool ValidPFJetEI = iEvent.getByLabel(PFJetLabelEI_, pfJetCollectionEI_);
  if(!ValidPFJetEI) return;
  pfjetsEI = *pfJetCollectionEI_;
  
  // PFMETs
  bool ValidPFMETEI = iEvent.getByLabel(PFMETLabelEI_, pfMETCollectionEI_);
  if(!ValidPFMETEI) return;

  // Jet Correction
  int countJet = 0;
  PFJetEIPt    = -99.;
  const JetCorrector* pfcorrectorEI = JetCorrector::getJetCorrector(PFJetCorService_,iSetup);
  PFJetCollection::const_iterator pfjet_ = pfjetsEI.begin();
  for(; pfjet_ != pfjetsEI.end(); ++pfjet_){
    double scale = pfcorrectorEI->correction(*pfjet_,iEvent, iSetup);
    if(scale*pfjet_->pt()<PFJetEIPt) continue;
    PFJetEIPt   = scale*pfjet_->pt();
    PFJetEIPx   = scale*pfjet_->px();
    PFJetEIPy   = scale*pfjet_->py();
    PFJetEIEta  = pfjet_->eta();
    PFJetEIPhi  = pfjet_->phi();
    PFJetEINHEF = pfjet_->neutralHadronEnergyFraction();
    PFJetEICHEF = pfjet_->chargedHadronEnergyFraction();
    PFJetEINEMF = pfjet_->neutralEmEnergyFraction();
    PFJetEICEMF = pfjet_->chargedEmEnergyFraction();
    countJet++;
  }
  if(countJet>0){
    ei_pfjet1_pt->Fill(PFJetEIPt);
  }
  
  const PFMETCollection *pfmetcolEI = pfMETCollectionEI_.product();
  const PFMET pfmetEI = pfmetcolEI->front();
  ei_pfmet_pt->Fill(pfmetEI.et());
}

//
// -- End Luminosity Block
//
void ExoticaDQM::endLuminosityBlock(edm::LuminosityBlock const& lumiSeg, edm::EventSetup const& eSetup) {
  //edm::LogInfo ("ExoticaDQM") <<"[ExoticaDQM]: End of LS transition, performing the DQM client operation";
  nLumiSecs_++;
  //edm::LogInfo("ExoticaDQM") << "============================================ " 
  //<< endl << " ===> Iteration # " << nLumiSecs_ << " " << lumiSeg.luminosityBlock() 
  //<< endl  << "============================================ " << endl;
}


//
// -- End Run
//
void ExoticaDQM::endRun(edm::Run const& run, edm::EventSetup const& eSetup){
}


//
// -- End Job
//
void ExoticaDQM::endJob(){
  //edm::LogInfo("ExoticaDQM") <<"[ExoticaDQM]: endjob called!";
}
