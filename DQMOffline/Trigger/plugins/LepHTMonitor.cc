#include "DQMOffline/Trigger/plugins/LepHTMonitor.h"

#include <limits>
#include <algorithm>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/HLTReco/interface/TriggerObject.h"

#include "DataFormats/Math/interface/deltaR.h"

#include "DataFormats/EgammaCandidates/interface/ConversionFwd.h"
#include "DataFormats/EgammaCandidates/interface/Conversion.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "RecoEgamma/EgammaTools/src/ConversionTools.cc" //Would prefer to include header, but fails to find hasMatchedConversion definition without this..
#include "DataFormats/PatCandidates/interface/Muon.h"

#include "CommonTools/TriggerUtils/interface/GenericTriggerEventFlag.h"

namespace{
  
  //Offline electron definition
  bool IsGood(const reco::GsfElectron &el, const reco::Vertex::Point &pv_position,
              const reco::BeamSpot::Point &bs_position,
              const edm::Handle<reco::ConversionCollection> &convs){
    const float dEtaIn = el.deltaEtaSuperClusterTrackAtVtx();
    const float dPhiIn = el.deltaPhiSuperClusterTrackAtVtx();
    const float sigmaietaieta = el.full5x5_sigmaIetaIeta();
    const float hOverE = el.hcalOverEcal();
    float d0 = 0.0, dz=0.0;
    if(el.pt()<10) return false;
    try{
      d0=-(el.gsfTrack()->dxy(pv_position));
      dz = el.gsfTrack()->dz(pv_position);
    }catch(...){
      edm::LogError("LepHTMonitor") << "Could not read electron.gsfTrack().\n";
      return false;
    }
    float ooemoop = 1e30;
    if(el.ecalEnergy()>0.0 && std::isfinite(el.ecalEnergy())){
      ooemoop = fabs(1.0/el.ecalEnergy() - el.eSuperClusterOverP()/el.ecalEnergy());
    }
    const auto &iso = el.pfIsolationVariables();
    const float absiso = iso.sumChargedHadronPt
      + std::max(0.0, iso.sumNeutralHadronEt + iso.sumPhotonEt -0.5 * iso.sumPUPt);
    const float relisowithdb = absiso/el.pt();

    bool pass_conversion = false;
    if(convs.isValid()){
      try{
        pass_conversion = !ConversionTools::hasMatchedConversion(el, convs, bs_position);
      }catch(...){
        edm::LogError("LepHTMonitor") << "Electron conversion matching failed.\n";
        return false;
      }
    }

    float etasc = 0.0;
    try{
      etasc = el.superCluster()->eta();
    }catch(...){
      edm::LogError("LepHTMonitor") << "Could not read electron.superCluster().\n";
      return false;
    }
    if(fabs(etasc)>2.5){
      return false;
    }else if(fabs(etasc)>1.479){
      if(fabs(dEtaIn)>0.00733) return false;
      if(fabs(dPhiIn)>0.114) return false;
      if(sigmaietaieta>0.0283) return false;
      if(hOverE>0.0678) return false;
      if(fabs(d0)>0.0739) return false;
      if(fabs(dz)>0.602) return false;
      if(fabs(ooemoop)>0.0898) return false;
      if(relisowithdb>0.1) return false;
      if(!pass_conversion) return false;
    }else{
      if(fabs(dEtaIn)>0.0103) return false;
      if(fabs(dPhiIn)>0.0336) return false;
      if(sigmaietaieta>0.0101) return false;
      if(hOverE>0.0876) return false;
      if(fabs(d0)>0.0118) return false;
      if(fabs(dz)>0.373) return false;
      if(fabs(ooemoop)>0.0174) return false;
      if(relisowithdb>0.1) return false;
      if(!pass_conversion) return false;
    }
    return true;
  }

  //Offline muon definition
  bool IsGood(const reco::Muon &mu, const reco::Vertex::Point &pv_position){
    if(mu.pt()<10) return false;
    try{
      const auto &iso = mu.pfIsolationR04();
      const float absiso = iso.sumChargedHadronPt
	+ std::max(0.0, iso.sumNeutralHadronEt + iso.sumPhotonEt -0.5 * iso.sumPUPt);
      const float relisowithdb = absiso/mu.pt();
      if(relisowithdb>0.2) return false;
    }catch(...){
      edm::LogWarning("LepHTMonitor") << "Could not read muon isolation.\n";
      return false;
    }
    try{
      bool isMed =  muon::isMediumMuon(mu);
      return isMed;
    }catch(...){
      edm::LogWarning("LepHTMonitor") << "Could not read isMediumMuon().\n";
      return false;
    }
  }
}
LepHTMonitor::LepHTMonitor(const edm::ParameterSet &ps):
  theElectronTag_(ps.getParameter<edm::InputTag>("electronCollection")),
  theElectronCollection_(consumes<reco::GsfElectronCollection>(theElectronTag_)),
  theMuonTag_(ps.getParameter<edm::InputTag>("muonCollection")),
  theMuonCollection_(consumes<reco::MuonCollection>(theMuonTag_)),
  thePfMETTag_(ps.getParameter<edm::InputTag>("pfMetCollection")),
  thePfMETCollection_(consumes<reco::PFMETCollection>(thePfMETTag_)),
  thePfJetTag_(ps.getParameter<edm::InputTag>("pfJetCollection")),
  thePfJetCollection_(consumes<reco::PFJetCollection>(thePfJetTag_)),
  theJetTagTag_(ps.getParameter<edm::InputTag>("jetTagCollection")),
  theJetTagCollection_(consumes<reco::JetTagCollection>(theJetTagTag_)),

  theVertexCollectionTag_(ps.getParameter<edm::InputTag>("vertexCollection")),
  theVertexCollection_(consumes<reco::VertexCollection>(theVertexCollectionTag_)),
  theConversionCollectionTag_(ps.getParameter<edm::InputTag>("conversionCollection")),
  theConversionCollection_(consumes<reco::ConversionCollection>(theConversionCollectionTag_)),
  theBeamSpotTag_(ps.getParameter<edm::InputTag>("beamSpot")),
  theBeamSpot_(consumes<reco::BeamSpot>(theBeamSpotTag_)),

  num_genTriggerEventFlag_(new GenericTriggerEventFlag(ps.getParameter<edm::ParameterSet>("numGenericTriggerEventPSet"),consumesCollector(), *this)),
  den_lep_genTriggerEventFlag_(new GenericTriggerEventFlag(ps.getParameter<edm::ParameterSet>("den_lep_GenericTriggerEventPSet"),consumesCollector(), *this)),
  den_HT_genTriggerEventFlag_(new GenericTriggerEventFlag(ps.getParameter<edm::ParameterSet>("den_HT_GenericTriggerEventPSet"),consumesCollector(), *this)),

  triggerPath_(ps.getParameter<std::string>("triggerPath")),

  jetPtCut_(ps.getUntrackedParameter<double>("jetPtCut")),
  jetEtaCut_(ps.getUntrackedParameter<double>("jetEtaCut")),
  metCut_(ps.getUntrackedParameter<double>("metCut")),
  htCut_(ps.getUntrackedParameter<double>("htCut")),

  nmusCut_(ps.getUntrackedParameter<double>("nmus")),
  nelsCut_(ps.getUntrackedParameter<double>("nels")),
  lep_pt_threshold_(ps.getUntrackedParameter<double>("leptonPtThreshold")),

  h_leptonTurnOn_num_(nullptr),
  h_leptonTurnOn_den_(nullptr),
  h_lepEtaTurnOn_num_(nullptr),
  h_lepEtaTurnOn_den_(nullptr),
  h_pfHTTurnOn_num_(nullptr),
  h_pfHTTurnOn_den_(nullptr){
  edm::LogInfo("LepHTMonitor")
    << "Constructor LepHTMonitor::LepHTMonitor\n";
  }

LepHTMonitor::~LepHTMonitor(){
  edm::LogInfo("LepHTMonitor")
    << "Destructor LepHTMonitor::~LepHTMonitor\n";
  if (num_genTriggerEventFlag_) delete num_genTriggerEventFlag_;
  if (den_lep_genTriggerEventFlag_) delete den_lep_genTriggerEventFlag_;
  if (den_HT_genTriggerEventFlag_) delete den_HT_genTriggerEventFlag_;
}

void LepHTMonitor::dqmBeginRun(const edm::Run &run, const edm::EventSetup &e){
  edm::LogInfo("LepHTMonitor") << "LepHTMonitor::beginRun\n";
}

void LepHTMonitor::bookHistograms(DQMStore::IBooker &ibooker,
                                           const edm::Run &iRun, const edm::EventSetup &iSetup){
  edm::LogInfo("LepHTMonitor") << "LepHTMonitor::bookHistograms\n";
  //book at beginRun
  ibooker.cd();
  ibooker.setCurrentFolder("HLT/LepHT/" + triggerPath_);

  bool is_mu = false;
  bool is_ele = false;
  if(theElectronTag_.label() == "" && theMuonTag_.label() != ""){
    is_mu=true;
  }else if(theElectronTag_.label() != "" && theMuonTag_.label() == ""){
    is_ele=true;
  }
  //Cosmetic axis names
  std::string lepton="lepton", Lepton="Lepton";
  if(is_mu && !is_ele){
    lepton="muon";
    Lepton="Muon";
  }else if(is_ele && !is_mu){
    lepton="electron";
    Lepton="Electron";
  }
  //Initialize trigger flags
  if ( num_genTriggerEventFlag_ && num_genTriggerEventFlag_->on() ) num_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_lep_genTriggerEventFlag_ && den_lep_genTriggerEventFlag_->on() ) den_lep_genTriggerEventFlag_->initRun( iRun, iSetup );
  if ( den_HT_genTriggerEventFlag_ && den_HT_genTriggerEventFlag_->on() ) den_HT_genTriggerEventFlag_->initRun( iRun, iSetup );

  //Define variable binning for lepton pT and HT histograms
  int nptbins=11;
  float ptbins[] = {0,10,20,30,40,50,75,100,125,160,200,250};
 
  int nhtbins=15;
  float htbins[] = {0,50,100,150,200,250,300,350,400,450,500,600,750,1000,1500,2000};

  
 
  //num and den hists to be divided in harvesting step to make turn on curves
  h_leptonTurnOn_num_ = ibooker.book1D("leptonTurnOn_num", ("Numerator;Offline "+lepton+" p_{T} [GeV];").c_str(), nptbins, &ptbins[0]);
  h_leptonTurnOn_den_ = ibooker.book1D("leptonTurnOn_den", ("Denominator;Offline "+lepton+" p_{T} [GeV];").c_str(), nptbins, &ptbins[0]);
  h_pfHTTurnOn_num_ = ibooker.book1D("pfHTTurnOn_num", "Numerator;Offline H_{T} [GeV];",
                                     nhtbins,&htbins[0] );
  h_pfHTTurnOn_den_ = ibooker.book1D("pfHTTurnOn_den",
                                     "Denominator;Offline H_{T} [GeV];",
                                     nhtbins,&htbins[0]  );
  h_lepEtaTurnOn_num_ = ibooker.book1D("lepEtaTurnOn_num",
				       "Numerator;Offline lepton #eta [GeV];",
				       10,-2.5,2.5);
  h_lepEtaTurnOn_den_ = ibooker.book1D("lepEtaTurnOn_den",
				       "Denominator;Offline lepton #eta [GeV];",
				       10,-2.5,2.5);


 
  ibooker.cd();
}

void LepHTMonitor::beginLuminosityBlock(const edm::LuminosityBlock &lumiSeg,
                                                 const edm::EventSetup &context){
  edm::LogInfo("LepHTMonitor") << "LepHTMonitor::beginLuminosityBlock\n";
}

void LepHTMonitor::analyze(const edm::Event &e, const edm::EventSetup &eSetup){
  edm::LogInfo("LepHTMonitor") << "LepHTMonitor::analyze\n";

  //Find whether main and auxilliary triggers fired
  bool hasFired = false;
  bool hasFiredAuxiliary = false;
  bool hasFiredLeptonAuxiliary = false;
  if (den_lep_genTriggerEventFlag_->on() && den_lep_genTriggerEventFlag_->accept( e, eSetup) ) hasFiredLeptonAuxiliary=true;
  if (den_HT_genTriggerEventFlag_->on() && den_HT_genTriggerEventFlag_->accept( e, eSetup) ) hasFiredAuxiliary=true;
  if (num_genTriggerEventFlag_->on() && num_genTriggerEventFlag_->accept( e, eSetup) ) hasFired=true;

  if(!(hasFiredAuxiliary || hasFiredLeptonAuxiliary)) return;

  //Vertex
  edm::Handle<reco::VertexCollection> VertexCollection;
  if(theVertexCollectionTag_.label() != ""){
    e.getByToken(theVertexCollection_, VertexCollection);
    if( !VertexCollection.isValid() ){
      edm::LogWarning("LepHTMonitor")
        << "Invalid VertexCollection: " << theVertexCollectionTag_.label() << '\n';
    }
  }

  //Conversions
  edm::Handle<reco::ConversionCollection> ConversionCollection;
  if(theConversionCollectionTag_.label() != ""){
    e.getByToken(theConversionCollection_, ConversionCollection);
    if( !ConversionCollection.isValid() ){
      edm::LogWarning("LepHTMonitor")
        << "Invalid ConversionCollection: " << theConversionCollectionTag_.label() << '\n';
    }
  }

  //Beam Spot
  edm::Handle<reco::BeamSpot> BeamSpot;
  if(theBeamSpotTag_.label() != ""){
    e.getByToken(theBeamSpot_, BeamSpot);
    if( !BeamSpot.isValid() ){
      edm::LogWarning("LepHTMonitor")
        << "Invalid BeamSpot: " << theBeamSpotTag_.label() << '\n';
    }
  }

  //MET
  edm::Handle<reco::PFMETCollection> pfMETCollection;
  if(thePfMETTag_.label() != ""){
    e.getByToken(thePfMETCollection_, pfMETCollection);
    if( !pfMETCollection.isValid() ){
      edm::LogWarning("LepHTMonitor")
        << "Invalid PFMETCollection: " << thePfMETTag_.label() << '\n';
    }
  }

  //Jets
  edm::Handle<reco::PFJetCollection> pfJetCollection;
  if(thePfJetTag_.label() != ""){
    e.getByToken (thePfJetCollection_,pfJetCollection);
    if( !pfJetCollection.isValid() ){
      edm::LogWarning("LepHTMonitor")
        << "Invalid PFJetCollection: " << thePfJetTag_.label() << '\n';
    }
  }

  //Electron
  edm::Handle<reco::GsfElectronCollection> ElectronCollection;
  if(theElectronTag_.label() != ""){
    e.getByToken (theElectronCollection_, ElectronCollection);
    if( !ElectronCollection.isValid() ){
      edm::LogWarning("LepHTMonitor")
        << "Invalid GsfElectronCollection: " << theElectronTag_.label() << '\n';
    }
  }

  //Muon
  edm::Handle<reco::MuonCollection> MuonCollection;
  if(theMuonTag_.label() != ""){
    e.getByToken (theMuonCollection_, MuonCollection);
    if( !MuonCollection.isValid() ){
      edm::LogWarning("LepHTMonitor")
        << "Invalid MuonCollection: " << theMuonTag_.label() << '\n';
    }
  }

  //Get offline HT
  double pfHT = -1.0;
  if(pfJetCollection.isValid()){
    pfHT=0.0;
    for(const auto &pfjet: *pfJetCollection){
      if(pfjet.pt() < jetPtCut_) continue;
      if(fabs(pfjet.eta()) > jetEtaCut_) continue;
      pfHT += pfjet.pt();
    }
  }

  //Get offline MET
  double pfMET = -1.0;
  if(pfMETCollection.isValid() && pfMETCollection->size()){
    pfMET = pfMETCollection->front().et();
  }


  //Find offline leptons and keep track of pt,eta of leading and trailing leptons
  double lep_max_pt = -1.0;
  double lep_eta=0;
  double trailing_ele_eta=0;
  double trailing_mu_eta=0;
  double min_ele_pt= -1.0;
  double min_mu_pt=-1.0;
  int nels=0;
  int nmus=0;
  if(VertexCollection.isValid() && VertexCollection->size()){//for quality checks
    //Try to find a reco electron
    if(ElectronCollection.isValid()
       && ConversionCollection.isValid()
       && BeamSpot.isValid()){
      for(const auto &electron: *ElectronCollection){
        if(IsGood(electron, VertexCollection->front().position(),
                  BeamSpot->position(), ConversionCollection)){
          if(electron.pt()>lep_max_pt) {lep_max_pt=electron.pt(); lep_eta=electron.eta();} 
	  if(electron.pt()<min_ele_pt || min_ele_pt<0){ min_ele_pt=electron.pt(); trailing_ele_eta=electron.eta();} 
	  nels++;
        }
      }
    }

    //Try to find a reco muon
    if(MuonCollection.isValid()){
      for(const auto &muon: *MuonCollection){
        if(IsGood(muon, VertexCollection->front().position())){
          if(muon.pt()>lep_max_pt) {lep_max_pt=muon.pt(); lep_eta=muon.eta();} 
          if(muon.pt()<min_mu_pt || min_mu_pt<0) {min_mu_pt=muon.pt(); trailing_mu_eta=muon.eta(); } 
	  nmus++;
        }
      }
    }
  }

  
  //Fill single lepton triggers with leading lepton pT
  float lep_pt = lep_max_pt;
  //For dilepton triggers, use trailing rather than leading lepton
  if(nmusCut_>=2) {lep_pt = min_mu_pt; lep_eta = trailing_mu_eta;}
  if(nelsCut_>=2) {lep_pt = min_ele_pt; lep_eta = trailing_ele_eta;}
	   
  const bool nleps_cut = nels>=nelsCut_ && nmus>=nmusCut_;
  const bool lep_plateau = lep_pt>lep_pt_threshold_ || lep_pt_threshold_<0.0;

  //Overflow protection (apparently not taken care of in harvesting step)
  if(lep_pt > 250) lep_pt=249;
  if(pfHT > 2000) pfHT=1999;

  //Fill lepton pT and eta histograms
  if(hasFiredLeptonAuxiliary || !e.isRealData()){
   
    if(nleps_cut  && (pfMET>metCut_ || metCut_<0.0) && (pfHT>htCut_ || htCut_<0.0)){
      if(h_leptonTurnOn_den_) h_leptonTurnOn_den_->Fill(lep_pt);
      if(h_leptonTurnOn_num_ && hasFired) h_leptonTurnOn_num_->Fill(lep_pt);

      if(lep_plateau){ //Fill Eta histograms for leptons above pT threshold
	if(h_lepEtaTurnOn_den_) h_lepEtaTurnOn_den_->Fill(lep_eta);
       	if(h_lepEtaTurnOn_num_ && hasFired) h_lepEtaTurnOn_num_->Fill(lep_eta);
      }
    }
  }
  
  //Fill HT turn-on histograms
  if(hasFiredAuxiliary || !e.isRealData()){
    if(nleps_cut && lep_plateau ){
      if(h_pfHTTurnOn_den_) h_pfHTTurnOn_den_->Fill(pfHT);
      if(h_pfHTTurnOn_num_ && hasFired) h_pfHTTurnOn_num_->Fill(pfHT);
    }
  }
}

void LepHTMonitor::endLuminosityBlock(const edm::LuminosityBlock &lumiSeg,
                                               const edm::EventSetup &eSetup){
  edm::LogInfo("LepHTMonitor")
    << "LepHTMonitor::endLuminosityBlock\n";
}

void LepHTMonitor::endRun(const edm::Run &run, const edm::EventSetup &eSetup){
  edm::LogInfo("LepHTMonitor") << "LepHTMonitor::endRun\n";
}

//define this as a plug-in
DEFINE_FWK_MODULE(LepHTMonitor);
