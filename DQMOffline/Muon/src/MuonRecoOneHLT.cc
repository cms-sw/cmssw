#include "DQMOffline/Muon/src/MuonRecoOneHLT.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/MuonReco/interface/MuonEnergy.h"

#include "FWCore/Common/interface/TriggerNames.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
#include "TMath.h"
using namespace std;
using namespace edm;

// Uncomment to DEBUG
//#define DEBUG

MuonRecoOneHLT::MuonRecoOneHLT(const edm::ParameterSet& pSet, MuonServiceProxy *theService):MuonAnalyzerBase(theService) {
  parameters = pSet;
  
  ParameterSet muonparms = parameters.getParameter<edm::ParameterSet>("muonTrigger");
  _MuonEventFlag         = new GenericTriggerEventFlag( muonparms );

  // Trigger Expresions in case de connection to the DB fails
  muonExpr_             = muonparms.getParameter<std::vector<std::string> >("hltPaths");
}


MuonRecoOneHLT::~MuonRecoOneHLT() {
  delete _MuonEventFlag;
}
void MuonRecoOneHLT::beginJob(DQMStore * dbe) {
#ifdef DEBUG
  cout << "[MuonRecoOneHLT]  beginJob " << endl;
#endif
  dbe->setCurrentFolder("Muons/MuonRecoOneHLT");
  
  muReco = dbe->book1D("muReco", "Muon Reconstructed Tracks", 6, 1, 7);
  muReco->setBinLabel(1,"glb+tk+sta"); 
  muReco->setBinLabel(2,"glb+sta");
  muReco->setBinLabel(3,"tk+sta");
  muReco->setBinLabel(4,"tk");
  muReco->setBinLabel(5,"sta");
  muReco->setBinLabel(6,"calo");

  // monitoring of eta parameter
  etaBin = parameters.getParameter<int>("etaBin");
  etaMin = parameters.getParameter<double>("etaMin");
  etaMax = parameters.getParameter<double>("etaMax");
  
  std::string histname = "GlbMuon_";
  etaGlbTrack.push_back(dbe->book1D(histname+"Glb_eta", "#eta_{GLB}", etaBin, etaMin, etaMax));
  etaGlbTrack.push_back(dbe->book1D(histname+"Tk_eta", "#eta_{TKfromGLB}", etaBin, etaMin, etaMax));
  etaGlbTrack.push_back(dbe->book1D(histname+"Sta_eta", "#eta_{STAfromGLB}", etaBin, etaMin, etaMax));

  etaTrack = dbe->book1D("TkMuon_eta", "#eta_{TK}", etaBin, etaMin, etaMax);
  etaStaTrack = dbe->book1D("StaMuon_eta", "#eta_{STA}", etaBin, etaMin, etaMax);

  // monitoring of phi paramater
  phiBin = parameters.getParameter<int>("phiBin");
  phiMin = parameters.getParameter<double>("phiMin");
  phiMax = parameters.getParameter<double>("phiMax");
  phiGlbTrack.push_back(dbe->book1D(histname+"Glb_phi", "#phi_{GLB}", phiBin, phiMin, phiMax));
  phiGlbTrack[0]->setAxisTitle("rad");
  phiGlbTrack.push_back(dbe->book1D(histname+"Tk_phi", "#phi_{TKfromGLB}", phiBin, phiMin, phiMax));
  phiGlbTrack[1]->setAxisTitle("rad");
  phiGlbTrack.push_back(dbe->book1D(histname+"Sta_phi", "#phi_{STAfromGLB}", phiBin, phiMin, phiMax));
  phiGlbTrack[2]->setAxisTitle("rad");
  phiTrack = dbe->book1D("TkMuon_phi", "#phi_{TK}", phiBin, phiMin, phiMax);
  phiTrack->setAxisTitle("rad"); 
  phiStaTrack = dbe->book1D("StaMuon_phi", "#phi_{STA}", phiBin, phiMin, phiMax);
  phiStaTrack->setAxisTitle("rad"); 

  // monitoring of the chi2 parameter
  chi2Bin = parameters.getParameter<int>("chi2Bin");
  chi2Min = parameters.getParameter<double>("chi2Min");
  chi2Max = parameters.getParameter<double>("chi2Max");
  chi2OvDFGlbTrack.push_back(dbe->book1D(histname+"Glb_chi2OverDf", "#chi_{2}OverDF_{GLB}", chi2Bin, chi2Min, chi2Max));
  chi2OvDFGlbTrack.push_back(dbe->book1D(histname+"Tk_chi2OverDf", "#chi_{2}OverDF_{TKfromGLB}", phiBin, chi2Min, chi2Max));
  chi2OvDFGlbTrack.push_back(dbe->book1D(histname+"Sta_chi2OverDf", "#chi_{2}OverDF_{STAfromGLB}", chi2Bin, chi2Min, chi2Max));
  chi2OvDFTrack = dbe->book1D("TkMuon_chi2OverDf", "#chi_{2}OverDF_{TK}", chi2Bin, chi2Min, chi2Max);
  chi2OvDFStaTrack = dbe->book1D("StaMuon_chi2OverDf", "#chi_{2}OverDF_{STA}", chi2Bin, chi2Min, chi2Max);

  // monitoring of the transverse momentum
  ptBin = parameters.getParameter<int>("ptBin");
  ptMin = parameters.getParameter<double>("ptMin");
  ptMax = parameters.getParameter<double>("ptMax");
  ptGlbTrack.push_back(dbe->book1D(histname+"Glb_pt", "pt_{GLB}", ptBin, ptMin, ptMax));
  ptGlbTrack[0]->setAxisTitle("GeV"); 
  ptGlbTrack.push_back(dbe->book1D(histname+"Tk_pt", "pt_{TKfromGLB}", ptBin, ptMin, ptMax));
  ptGlbTrack[1]->setAxisTitle("GeV"); 
  ptGlbTrack.push_back(dbe->book1D(histname+"Sta_pt", "pt_{STAfromGLB}", ptBin, ptMin, ptMax));
  ptGlbTrack[2]->setAxisTitle("GeV"); 
  ptTrack = dbe->book1D("TkMuon_pt", "pt_{TK}", ptBin, ptMin, ptMax);
  ptTrack->setAxisTitle("GeV"); 
  ptStaTrack = dbe->book1D("StaMuon_pt", "pt_{STA}", ptBin, ptMin, ptMax);
  ptStaTrack->setAxisTitle("GeV"); 


}
void MuonRecoOneHLT::beginRun(const edm::Run& iRun, const edm::EventSetup& iSetup){
#ifdef DEBUG
  cout << "[MuonRecoOneHLT]  beginRun " << endl;
  cout << "[MuonRecoOneHLT]  Is MuonEventFlag On? "<< _MuonEventFlag->on() << endl;
#endif
  if ( _MuonEventFlag->on() ) _MuonEventFlag->initRun( iRun, iSetup );


  if (_MuonEventFlag->on() && _MuonEventFlag->expressionsFromDB(_MuonEventFlag->hltDBKey(), iSetup)[0] != "CONFIG_ERROR")
    muonExpr_      = _MuonEventFlag->expressionsFromDB(_MuonEventFlag->hltDBKey(),           iSetup);
}
void MuonRecoOneHLT::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, 
			     const reco::Muon& recoMu, const edm::TriggerResults& triggerResults) {
#ifdef DEBUG
  cout << "[MuonRecoOneHLT]  analyze "<< endl;
#endif
  
  const edm::TriggerNames& triggerNames = iEvent.triggerNames(triggerResults);
  const unsigned int nTrig(triggerNames.size());
  bool _trig_Muon = false;
  for (unsigned int i=0;i<nTrig;++i){
    if (triggerNames.triggerName(i).find(muonExpr_[0].substr(0,muonExpr_[0].rfind("_v")+2))!=std::string::npos && triggerResults.accept(i))
      _trig_Muon=true;
  }
#ifdef DEBUG
  cout << "[MuonRecoOneHLT]  Trigger Fired ? "<< _trig_Muon << endl;
#endif
  if (!_trig_Muon) return;
  //  if (_MuonEventFlag->on() && !(_MuonEventFlag->accept(iEvent,iSetup))) return;
  
  // Check if Muon is Global
  if(recoMu.isGlobalMuon()) {
    LogTrace(metname)<<"[MuonRecoOneHLT] The mu is global - filling the histos";
    if(recoMu.isTrackerMuon() && recoMu.isStandAloneMuon())          muReco->Fill(1);
    if(!(recoMu.isTrackerMuon()) && recoMu.isStandAloneMuon())       muReco->Fill(2);
    if(!recoMu.isStandAloneMuon())   
      LogTrace(metname)<<"[MuonRecoOneHLT] ERROR: the mu is global but not standalone!";

    // get the track combinig the information from both the Tracker and the Spectrometer
    reco::TrackRef recoCombinedGlbTrack = recoMu.combinedMuon();
    // get the track using only the tracker data
    reco::TrackRef recoTkGlbTrack = recoMu.track();
    // get the track using only the mu spectrometer data
    reco::TrackRef recoStaGlbTrack = recoMu.standAloneMuon();

    etaGlbTrack[0]->Fill(recoCombinedGlbTrack->eta());
    etaGlbTrack[1]->Fill(recoTkGlbTrack->eta());
    etaGlbTrack[2]->Fill(recoStaGlbTrack->eta());
    
    phiGlbTrack[0]->Fill(recoCombinedGlbTrack->phi());
    phiGlbTrack[1]->Fill(recoTkGlbTrack->phi());
    phiGlbTrack[2]->Fill(recoStaGlbTrack->phi());
    
    chi2OvDFGlbTrack[0]->Fill(recoCombinedGlbTrack->normalizedChi2());
    chi2OvDFGlbTrack[1]->Fill(recoTkGlbTrack->normalizedChi2());
    chi2OvDFGlbTrack[2]->Fill(recoStaGlbTrack->normalizedChi2());

    ptGlbTrack[0]->Fill(recoCombinedGlbTrack->pt());
    ptGlbTrack[1]->Fill(recoTkGlbTrack->pt());
    ptGlbTrack[2]->Fill(recoStaGlbTrack->pt());
  }
  // Check if Muon is Tracker but NOT Global
  if(recoMu.isTrackerMuon() && !(recoMu.isGlobalMuon())) {
    LogTrace(metname)<<"[MuonRecoOneHLT] The mu is tracker only - filling the histos";
    if(recoMu.isStandAloneMuon())          muReco->Fill(3);
    if(!(recoMu.isStandAloneMuon()))        muReco->Fill(4);
    
    // get the track using only the tracker data
    reco::TrackRef recoTrack = recoMu.track();

    etaTrack->Fill(recoTrack->eta());
    phiTrack->Fill(recoTrack->phi());
    chi2OvDFTrack->Fill(recoTrack->normalizedChi2());
    ptTrack->Fill(recoTrack->pt());
  }
  // Check if Muon is STA but NOT Global
  if(recoMu.isStandAloneMuon() && !(recoMu.isGlobalMuon())) {
    LogTrace(metname)<<"[MuonRecoOneHLT] The mu is STA only - filling the histos";
    if(!(recoMu.isTrackerMuon()))         muReco->Fill(5);
     
    // get the track using only the mu spectrometer data
    reco::TrackRef recoStaTrack = recoMu.standAloneMuon();

    etaStaTrack->Fill(recoStaTrack->eta());
    phiStaTrack->Fill(recoStaTrack->phi());
    chi2OvDFStaTrack->Fill(recoStaTrack->normalizedChi2());
    ptStaTrack->Fill(recoStaTrack->pt());
  }
  // Check if Muon is Only CaloMuon
  if(recoMu.isCaloMuon() && !(recoMu.isGlobalMuon()) && !(recoMu.isTrackerMuon()) && !(recoMu.isStandAloneMuon()))
    muReco->Fill(6);
}
