/*
 *  See header file for a description of this class.
 *
 *  $Date: 2011/05/22 18:17:56 $
 *  $Revision: 1.2 $
 *  \author S. Goy Lopez, CIEMAT 
 */

#include "DQMOffline/Muon/src/MuonKinVsEtaAnalyzer.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h" 
#include "DataFormats/MuonReco/interface/MuonEnergy.h"

#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <string>
using namespace std;
using namespace edm;



MuonKinVsEtaAnalyzer::MuonKinVsEtaAnalyzer(const edm::ParameterSet& pSet, MuonServiceProxy *theService):MuonAnalyzerBase(theService){  parameters = pSet;
}


MuonKinVsEtaAnalyzer::~MuonKinVsEtaAnalyzer() { }


void MuonKinVsEtaAnalyzer::beginJob(DQMStore * dbe) {

  metname = "muonKinVsEta";

  LogTrace(metname)<<"[MuonKinVsEtaAnalyzer] Parameters initialization";
  dbe->setCurrentFolder("Muons/MuonKinVsEtaAnalyzer");

  etaBin = parameters.getParameter<int>("etaBin");
  etaMin = parameters.getParameter<double>("etaMin");
  etaMax = parameters.getParameter<double>("etaMax");

  phiBin = parameters.getParameter<int>("phiBin");
  phiMin = parameters.getParameter<double>("phiMin");
  phiMax = parameters.getParameter<double>("phiMax");

  pBin = parameters.getParameter<int>("pBin");
  pMin = parameters.getParameter<double>("pMin");
  pMax = parameters.getParameter<double>("pMax");

  ptBin = parameters.getParameter<int>("ptBin");
  ptMin = parameters.getParameter<double>("ptMin");
  ptMax = parameters.getParameter<double>("ptMax");

  etaBMin = parameters.getParameter<double>("etaBMin");
  etaBMax = parameters.getParameter<double>("etaBMax");
  etaECMin = parameters.getParameter<double>("etaECMin");
  etaECMax = parameters.getParameter<double>("etaECMax");
  etaOvlpMin = parameters.getParameter<double>("etaOvlpMin");
  etaOvlpMax = parameters.getParameter<double>("etaOvlpMax");

  std::string EtaName;
  for(unsigned int iEtaRegion=0;iEtaRegion<4;iEtaRegion++){
    if (iEtaRegion==0) EtaName = "Barrel_";   
    if (iEtaRegion==1) EtaName = "EndCap_";   
    if (iEtaRegion==2) EtaName = "Overlap_";
    if (iEtaRegion==3) EtaName = "_";

    // monitoring of eta parameter
    etaGlbTrack.push_back(dbe->book1D("GlbMuon_eta_"+EtaName, "#eta_{GLB} "+EtaName, etaBin, etaMin, etaMax));
    etaTrack.push_back(dbe->book1D("TkMuon_eta_"+EtaName, "#eta_{TK} "+EtaName, etaBin, etaMin, etaMax));
    etaStaTrack.push_back(dbe->book1D("StaMuon_eta_"+EtaName, "#eta_{STA} "+EtaName, etaBin, etaMin, etaMax));
    etaTightTrack.push_back(dbe->book1D("TightMuon_eta_"+EtaName, "#eta_{Tight} "+EtaName, etaBin, etaMin, etaMax));
    
    // monitoring of phi paramater
    phiGlbTrack.push_back(dbe->book1D("GlbMuon_phi_"+EtaName, "#phi_{GLB} "+EtaName+ "(rad)", phiBin, phiMin, phiMax));
    phiTrack.push_back(dbe->book1D("TkMuon_phi_"+EtaName, "#phi_{TK}" +EtaName +"(rad)", phiBin, phiMin, phiMax));
    phiStaTrack.push_back(dbe->book1D("StaMuon_phi_"+EtaName, "#phi_{STA}"+EtaName+" (rad)", phiBin, phiMin, phiMax));
    phiTightTrack.push_back(dbe->book1D("TightMuon_phi_"+EtaName, "#phi_{Tight}_"+EtaName, phiBin, phiMin, phiMax));

    // monitoring of the momentum
    pGlbTrack.push_back(dbe->book1D("GlbMuon_p_"+EtaName, "p_{GLB} "+EtaName, pBin, pMin, pMax));
    pTrack.push_back(dbe->book1D("TkMuon_p"+EtaName, "p_{TK} "+EtaName, pBin, pMin, pMax));
    pStaTrack.push_back(dbe->book1D("StaMuon_p"+EtaName, "p_{STA} "+EtaName, pBin, pMin, pMax));
    pTightTrack.push_back(dbe->book1D("TightMuon_p_"+EtaName, "p_{Tight} "+EtaName, pBin, pMin, pMax));

    // monitoring of the transverse momentum
    ptGlbTrack.push_back(dbe->book1D("GlbMuon_pt_" +EtaName, "pt_{GLB} "+EtaName, ptBin, ptMin, ptMax));
    ptTrack.push_back(dbe->book1D("TkMuon_pt_"+EtaName, "pt_{TK} "+EtaName, ptBin, ptMin, ptMax));
    ptStaTrack.push_back(dbe->book1D("StaMuon_pt_"+EtaName, "pt_{STA} "+EtaName, ptBin, ptMin, pMax));
    ptTightTrack.push_back(dbe->book1D("TightMuon_pt_"+EtaName, "pt_{Tight} "+EtaName, ptBin, ptMin, ptMax));
  }
}




void MuonKinVsEtaAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup, const reco::Muon& recoMu) {
  
  reco::BeamSpot beamSpot;
  Handle<reco::BeamSpot> beamSpotHandle;
  iEvent.getByLabel("offlineBeamSpot", beamSpotHandle);
  beamSpot = *beamSpotHandle;
  
  LogTrace(metname)<<"[MuonKinVsEtaAnalyzer] Analyze the mu in different eta regions";

  for(unsigned int iEtaRegion=0;iEtaRegion<4;iEtaRegion++){
    if (iEtaRegion==0) {EtaCutMin= etaBMin; EtaCutMax=etaBMax;}
    if (iEtaRegion==1) {EtaCutMin= etaECMin; EtaCutMax=etaECMax;}
    if (iEtaRegion==2) {EtaCutMin= etaOvlpMin; EtaCutMax=etaOvlpMax;} 
    if (iEtaRegion==3) {EtaCutMin= etaBMin; EtaCutMax=etaECMax;}
    
    if(recoMu.isGlobalMuon()) {
      LogTrace(metname)<<"[MuonKinVsEtaAnalyzer] The mu is global - filling the histos";
      reco::TrackRef recoCombinedGlbTrack = recoMu.combinedMuon();
      // get the track combinig the information from both the glb fit"
      if(fabs(recoCombinedGlbTrack->eta())>EtaCutMin && fabs(recoCombinedGlbTrack->eta())<EtaCutMax){
	etaGlbTrack[iEtaRegion]->Fill(recoCombinedGlbTrack->eta());
	phiGlbTrack[iEtaRegion]->Fill(recoCombinedGlbTrack->phi());
	pGlbTrack[iEtaRegion]->Fill(recoCombinedGlbTrack->p());
	ptGlbTrack[iEtaRegion]->Fill(recoCombinedGlbTrack->pt());
      }
    }

    if(recoMu.isTrackerMuon()) {
      LogTrace(metname)<<"[MuonKinVsEtaAnalyzer] The mu is tracker - filling the histos";
      // get the track using only the tracker data
      reco::TrackRef recoTrack = recoMu.track();
      if(fabs(recoTrack->eta())>EtaCutMin && fabs(recoTrack->eta())<EtaCutMax){
	etaTrack[iEtaRegion]->Fill(recoTrack->eta());
	phiTrack[iEtaRegion]->Fill(recoTrack->phi());
	pTrack[iEtaRegion]->Fill(recoTrack->p());
	ptTrack[iEtaRegion]->Fill(recoTrack->pt());
      }
    }

    if(recoMu.isStandAloneMuon()) {
      LogTrace(metname)<<"[MuonKinVsEtaAnalyzer] The mu is standalone - filling the histos";
      // get the track using only the mu spectrometer data
      reco::TrackRef recoStaTrack = recoMu.standAloneMuon();
      if(fabs(recoStaTrack->eta())>EtaCutMin && fabs(recoStaTrack->eta())<EtaCutMax){
	etaStaTrack[iEtaRegion]->Fill(recoStaTrack->eta());
	phiStaTrack[iEtaRegion]->Fill(recoStaTrack->phi());
	pStaTrack[iEtaRegion]->Fill(recoStaTrack->p());
	ptStaTrack[iEtaRegion]->Fill(recoStaTrack->pt());
      }
    }
    if (recoMu.isGlobalMuon() && recoMu.isTrackerMuon() && recoMu.combinedMuon()->normalizedChi2()<10. 
	&& recoMu.combinedMuon()->hitPattern().numberOfValidMuonHits()>0 && fabs(recoMu.combinedMuon()->dxy(beamSpot.position()))<0.2 
	&& recoMu.combinedMuon()->hitPattern().numberOfValidPixelHits()>0 && recoMu.numberOfMatches() > 1) {
      LogTrace(metname)<<"[MuonKinVsEtaAnalyzer] The mu is Tight - filling the histos";
      reco::TrackRef recoTightTrack = recoMu.combinedMuon();
      if(fabs(recoTightTrack->eta())>EtaCutMin && fabs(recoTightTrack->eta())<EtaCutMax){
	etaTightTrack[iEtaRegion]->Fill(recoTightTrack->eta());
	phiTightTrack[iEtaRegion]->Fill(recoTightTrack->phi());
	pTightTrack[iEtaRegion]->Fill(recoTightTrack->p());
	ptTightTrack[iEtaRegion]->Fill(recoTightTrack->pt());
      }
    }
  }
}
