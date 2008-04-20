
/*
 *  See header file for a description of this class.
 *
 *  $Date: 2008/04/02 13:59:53 $
 *  $Revision: 1.2 $
 *  \author G. Mila - INFN Torino
 */

#include "DQMOffline/Muon/src/MuonRecoAnalyzer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/MuonReco/interface/MuonEnergy.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
using namespace std;
using namespace edm;



MuonRecoAnalyzer::MuonRecoAnalyzer(const edm::ParameterSet& pSet, MuonServiceProxy *theService):MuonAnalyzerBase(theService) {

 cout<<"[MuonRecoAnalyzer] Constructor called!"<<endl;
  parameters = pSet;

}


MuonRecoAnalyzer::~MuonRecoAnalyzer() { }


void MuonRecoAnalyzer::beginJob(edm::EventSetup const& iSetup,DQMStore * dbe) {

  metname = "muRecoAnalyzer";

  LogTrace(metname)<<"[MuonRecoAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("Muons/MuonRecoAnalyzer");

  muReco = dbe->book1D("muReco", "muReco", 3, 1, 4);
  muReco->setBinLabel(1,"global",1);
  muReco->setBinLabel(2,"tracker",1);
  muReco->setBinLabel(3,"sta",1);
  
  // monitoring of eta parameter
  etaBin = parameters.getParameter<int>("etaBin");
  etaMin = parameters.getParameter<double>("etaMin");
  etaMax = parameters.getParameter<double>("etaMax");
  std::string histname = "GlbMuon_";
  etaGlbTrack.push_back(dbe->book1D(histname+"Glb_eta", histname+"Glb_eta", etaBin, etaMin, etaMax));
  etaGlbTrack.push_back(dbe->book1D(histname+"Tk_eta", histname+"Tk_eta", etaBin, etaMin, etaMax));
  etaGlbTrack.push_back(dbe->book1D(histname+"Sta_eta", histname+"Sta_eta", etaBin, etaMin, etaMax));
  etaResolution.push_back(dbe->book1D("Res_TkGlb_eta", "Res_TkGlb_eta", etaBin*50, etaMin/10, etaMax/10));
  etaResolution.push_back(dbe->book1D("Res_StaGlb_eta", "Res_StaGlb_eta", etaBin*50, etaMin/10, etaMax/10));
  etaResolution.push_back(dbe->book1D("Res_TkSta_eta", "Res_StaGlb_eta", etaBin*50, etaMin/10, etaMax/10));
  etaTrack = dbe->book1D("TkMuon_eta", "TkMuon_eta", etaBin, etaMin, etaMax);
  etaStaTrack = dbe->book1D("StaMuon_eta", "StaMuon_eta", etaBin, etaMin, etaMax);

  // monitoring of theta parameter
  thetaBin = parameters.getParameter<int>("thetaBin");
  thetaMin = parameters.getParameter<double>("thetaMin");
  thetaMax = parameters.getParameter<double>("thetaMax");
  thetaGlbTrack.push_back(dbe->book1D(histname+"Glb_theta", histname+"Glb_theta", thetaBin, thetaMin, thetaMax));
  thetaGlbTrack.push_back(dbe->book1D(histname+"Tk_theta", histname+"Tk_theta", thetaBin, thetaMin, thetaMax));
  thetaGlbTrack.push_back(dbe->book1D(histname+"Sta_theta", histname+"Sta_theta", thetaBin, thetaMin, thetaMax));
  thetaResolution.push_back(dbe->book1D("Res_TkGlb_theta", "Res_TkGlb_theta", thetaBin*50, -(thetaMax/10), thetaMax/10));
  thetaResolution.push_back(dbe->book1D("Res_StaGlb_theta", "Res_StaGlb_theta", thetaBin*50,-(thetaMax/10), thetaMax/10));
  thetaResolution.push_back(dbe->book1D("Res_TkSta_theta", "Res_TkGlb_theta", thetaBin*50, -(thetaMax/10), thetaMax/10));
  thetaTrack = dbe->book1D("TkMuon_theta", "TkMuon_theta", thetaBin, thetaMin, thetaMax);
  thetaStaTrack = dbe->book1D("StaMuon_theta", "StaMuon_theta", thetaBin, thetaMin, thetaMax);

  // monitoring of phi paramater
  phiBin = parameters.getParameter<int>("phiBin");
  phiMin = parameters.getParameter<double>("phiMin");
  phiMax = parameters.getParameter<double>("phiMax");
  phiGlbTrack.push_back(dbe->book1D(histname+"Glb_phi", histname+"Glb_phi", phiBin, phiMin, phiMax));
  phiGlbTrack.push_back(dbe->book1D(histname+"Tk_phi", histname+"Tk_phi", phiBin, phiMin, phiMax));
  phiGlbTrack.push_back(dbe->book1D(histname+"Sta_phi", histname+"Sta_phi", phiBin, phiMin, phiMax));
  phiResolution.push_back(dbe->book1D("Res_TkGlb_phi", "Res_TkGlb_phi", phiBin*50, phiMin/10, phiMax/10));
  phiResolution.push_back(dbe->book1D("Res_StaGlb_phi", "Res_StaGlb_phi", phiBin*50, phiMin/10, phiMax/10));
  phiResolution.push_back(dbe->book1D("Res_TkSta_phi", "Res_TkGlb_phi", phiBin*50, phiMin/10, phiMax/10));
  phiTrack = dbe->book1D("TkMuon_phi", "TkMuon_phi", phiBin, phiMin, phiMax);
  phiStaTrack = dbe->book1D("StaMuon_phi", "StaMuon_phi", phiBin, phiMin, phiMax);

  // monitoring of the momentum
  pBin = parameters.getParameter<int>("pBin");
  pMin = parameters.getParameter<double>("pMin");
  pMax = parameters.getParameter<double>("pMax");
  pGlbTrack.push_back(dbe->book1D(histname+"Glb_p", histname+"Glb_p", pBin, pMin, pMax));
  pGlbTrack.push_back(dbe->book1D(histname+"Tk_p", histname+"Tk_p", pBin, pMin, pMax));
  pGlbTrack.push_back(dbe->book1D(histname+"Sta_p", histname+"Sta_p", pBin, pMin, pMax));
  pTrack = dbe->book1D("TkMuon_p", "TkMuon_p", pBin, pMin, pMax);
  pStaTrack = dbe->book1D("StaMuon_p", "StaMuon_p", pBin, pMin, pMax);

  // monitoring of the transverse momentum
  ptBin = parameters.getParameter<int>("ptBin");
  ptMin = parameters.getParameter<double>("ptMin");
  ptMax = parameters.getParameter<double>("ptMax");
  ptGlbTrack.push_back(dbe->book1D(histname+"Glb_pt", histname+"Glb_pt", ptBin, ptMin, ptMax));
  ptGlbTrack.push_back(dbe->book1D(histname+"Tk_pt", histname+"Tk_pt", ptBin, ptMin, ptMax));
  ptGlbTrack.push_back(dbe->book1D(histname+"Sta_pt", histname+"Sta_pt", ptBin, ptMin, ptMax));
  ptTrack = dbe->book1D("TkMuon_pt", "TkMuon_pt", ptBin, ptMin, ptMax);
  ptStaTrack = dbe->book1D("StaMuon_pt", "StaMuon_pt", ptBin, ptMin, pMax);

  // monitoring of the muon charge
  qGlbTrack.push_back(dbe->book1D(histname+"Glb_q", histname+"Glb_q", 5, -2.5, 2.5));
  qGlbTrack.push_back(dbe->book1D(histname+"Tk_q", histname+"Tk_q", 5, -2.5, 2.5));
  qGlbTrack.push_back(dbe->book1D(histname+"Sta_q", histname+"Sta_q", 5, -2.5, 2.5));
  qTrack = dbe->book1D("TkMuon_q", "TkMuon_q", 5, -2.5, 2.5);
  qStaTrack = dbe->book1D("StaMuon_q", "StaMuon_q", 5, -2.5, 2.5);

  // monitoring of the momentum resolution
  pResBin = parameters.getParameter<int>("pResBin");
  pResMin = parameters.getParameter<double>("pResMin");
  pResMax = parameters.getParameter<double>("pResMax");
  qOverpResolution.push_back(dbe->book1D("Res_TkGlb_qOverp", "Res_TkGlb_qOverp", pResBin*50, pResMin/2, pResMax/2));
  qOverpResolution.push_back(dbe->book1D("Res_StaGlb_qOverp", "Res_StaGlb_qOverp", pResBin*50, pResMin/2, pResMax/2));
  qOverpResolution.push_back(dbe->book1D("Res_TkSta_qOverp", "Res_TkSta_qOverp", pResBin*50, pResMin/2, pResMax/2));
  oneOverpResolution.push_back(dbe->book1D("Res_TkGlb_oneOverp", "Res_TkGlb_oneOverp", pResBin*50, pResMin/2, pResMax/2));
  oneOverpResolution.push_back(dbe->book1D("Res_StaGlb_oneOverp", "Res_StaGlb_oneOverp", pResBin*50, pResMin/2, pResMax/2));
  oneOverpResolution.push_back(dbe->book1D("Res_TkSta_oneOverp", "Res_TkSta_oneOverp", pResBin*50, pResMin/2, pResMax/2));
  qOverptResolution.push_back(dbe->book1D("Res_TkGlb_qOverpt", "Res_TkGlb_qOverpt", pResBin*50, pResMin/2, pResMax/2));
  qOverptResolution.push_back(dbe->book1D("Res_StaGlb_qOverpt", "Res_StaGlb_qOverpt", pResBin*50, pResMin/2, pResMax/2));
  qOverptResolution.push_back(dbe->book1D("Res_TkSta_qOverpt", "Res_TkSta_qOverpt", pResBin*50, pResMin/2, pResMax/2));
  oneOverptResolution.push_back(dbe->book1D("Res_TkGlb_oneOverpt", "Res_TkGlb_oneOverpt", pResBin*50, pResMin/2, pResMax/2));
  oneOverptResolution.push_back(dbe->book1D("Res_StaGlb_oneOverpt", "Res_StaGlb_oneOverpt", pResBin*50, pResMin/2, pResMax/2));
  oneOverptResolution.push_back(dbe->book1D("Res_TkSta_oneOverpt", "Res_TkSta_oneOverpt", pResBin*50, pResMin/2, pResMax/2));

}


void MuonRecoAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::Muon& recoMu) {

  LogTrace(metname)<<"[MuonRecoAnalyzer] Analyze the mu";

  if(recoMu.isGlobalMuon()) {

    LogTrace(metname)<<"[MuonRecoAnalyzer] The mu is global - filling the histos";
    muReco->Fill(1);

    // get the track combinig the information from both the Tracker and the Spectrometer
    reco::TrackRef recoCombinedGlbTrack = recoMu.combinedMuon();
    // get the track using only the tracker data
    reco::TrackRef recoGlbTrack = recoMu.track();
    // get the track using only the mu spectrometer data
    reco::TrackRef recoStaGlbTrack = recoMu.standAloneMuon();
  
    etaGlbTrack[0]->Fill(recoCombinedGlbTrack->eta());
    etaGlbTrack[1]->Fill(recoGlbTrack->eta());
    etaGlbTrack[2]->Fill(recoStaGlbTrack->eta());
    etaResolution[0]->Fill(recoGlbTrack->eta()-recoCombinedGlbTrack->eta());
    etaResolution[1]->Fill(recoStaGlbTrack->eta()-recoCombinedGlbTrack->eta());
    etaResolution[2]->Fill(recoGlbTrack->eta()-recoStaGlbTrack->eta());

    thetaGlbTrack[0]->Fill(recoCombinedGlbTrack->theta());
    thetaGlbTrack[1]->Fill(recoGlbTrack->theta());
    thetaGlbTrack[2]->Fill(recoStaGlbTrack->theta());
    thetaResolution[0]->Fill(recoGlbTrack->theta()-recoCombinedGlbTrack->theta());
    thetaResolution[1]->Fill(recoStaGlbTrack->theta()-recoCombinedGlbTrack->theta());
    thetaResolution[2]->Fill(recoGlbTrack->theta()-recoStaGlbTrack->theta());
    
    phiGlbTrack[0]->Fill(recoCombinedGlbTrack->phi());
    phiGlbTrack[1]->Fill(recoGlbTrack->phi());
    phiGlbTrack[2]->Fill(recoStaGlbTrack->phi());
    phiResolution[0]->Fill(recoGlbTrack->phi()-recoCombinedGlbTrack->phi());
    phiResolution[1]->Fill(recoStaGlbTrack->phi()-recoCombinedGlbTrack->phi());
    phiResolution[2]->Fill(recoGlbTrack->phi()-recoStaGlbTrack->phi());
  
    pGlbTrack[0]->Fill(recoCombinedGlbTrack->p());
    pGlbTrack[1]->Fill(recoGlbTrack->p());
    pGlbTrack[2]->Fill(recoStaGlbTrack->p());

    ptGlbTrack[0]->Fill(recoCombinedGlbTrack->pt());
    ptGlbTrack[1]->Fill(recoGlbTrack->pt());
    ptGlbTrack[2]->Fill(recoStaGlbTrack->pt());

    qGlbTrack[0]->Fill(recoCombinedGlbTrack->charge());
    qGlbTrack[1]->Fill(recoGlbTrack->charge());
    qGlbTrack[2]->Fill(recoStaGlbTrack->charge());
    
    qOverpResolution[0]->Fill((recoGlbTrack->charge()/recoGlbTrack->p())-(recoCombinedGlbTrack->charge()/recoCombinedGlbTrack->p()));
    qOverpResolution[1]->Fill((recoStaGlbTrack->charge()/recoStaGlbTrack->p())-(recoCombinedGlbTrack->charge()/recoCombinedGlbTrack->p()));
    qOverpResolution[2]->Fill((recoGlbTrack->charge()/recoGlbTrack->p())-(recoStaGlbTrack->charge()/recoStaGlbTrack->p()));
    oneOverpResolution[0]->Fill((1/recoGlbTrack->p())-(1/recoCombinedGlbTrack->p()));
    oneOverpResolution[1]->Fill((1/recoStaGlbTrack->p())-(1/recoCombinedGlbTrack->p()));
    oneOverpResolution[2]->Fill((1/recoGlbTrack->p())-(1/recoStaGlbTrack->p()));
    qOverptResolution[0]->Fill((recoGlbTrack->charge()/recoGlbTrack->pt())-(recoCombinedGlbTrack->charge()/recoCombinedGlbTrack->pt()));
    qOverptResolution[1]->Fill((recoStaGlbTrack->charge()/recoStaGlbTrack->pt())-(recoCombinedGlbTrack->charge()/recoCombinedGlbTrack->pt()));
    qOverptResolution[2]->Fill((recoGlbTrack->charge()/recoGlbTrack->pt())-(recoStaGlbTrack->charge()/recoStaGlbTrack->pt()));
    oneOverptResolution[0]->Fill((1/recoGlbTrack->pt())-(1/recoCombinedGlbTrack->pt()));
    oneOverptResolution[1]->Fill((1/recoStaGlbTrack->pt())-(1/recoCombinedGlbTrack->pt()));
    oneOverptResolution[2]->Fill((1/recoGlbTrack->pt())-(1/recoStaGlbTrack->pt()));

  }

  if(recoMu.isTrackerMuon()) {
    LogTrace(metname)<<"[MuonRecoAnalyzer] The mu is tracker only - filling the histos";
    muReco->Fill(2);

    // get the track using only the tracker data
    reco::TrackRef recoTrack = recoMu.track();

    etaTrack->Fill(recoTrack->eta());
    thetaTrack->Fill(recoTrack->theta());
    phiTrack->Fill(recoTrack->phi());
    pTrack->Fill(recoTrack->p());
    ptTrack->Fill(recoTrack->pt());
    qTrack->Fill(recoTrack->charge());
    
  }

  if(recoMu.isStandAloneMuon()) {
    LogTrace(metname)<<"[MuonRecoAnalyzer] The mu is STA only - filling the histos";
    muReco->Fill(3);
    
    // get the track using only the mu spectrometer data
    reco::TrackRef recoStaTrack = recoMu.standAloneMuon();

    etaStaTrack->Fill(recoStaTrack->eta());
    thetaStaTrack->Fill(recoStaTrack->theta());
    phiStaTrack->Fill(recoStaTrack->phi());
    pTrack->Fill(recoStaTrack->p());
    ptTrack->Fill(recoStaTrack->pt());
    qTrack->Fill(recoStaTrack->charge());

  }
    

}
