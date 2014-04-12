#include <iostream>
#include <sstream>
#include <istream>
#include <fstream>
#include <iomanip>
#include <string>
#include <cmath>
#include <functional>
#include <stdlib.h>
#include <string.h>

#include "HLTrigger/HLTanalyzers/interface/HLTMuon.h"

HLTMuon::HLTMuon() {
  evtCounter=0;

  //set parameter defaults 
  _Monte=false;
  _Debug=false;
}

/*  Setup the analysis to put the branch-variables into the tree. */
void HLTMuon::setup(const edm::ParameterSet& pSet, TTree* HltTree) {

  edm::ParameterSet myEmParams = pSet.getParameter<edm::ParameterSet>("RunParameters") ;
  std::vector<std::string> parameterNames = myEmParams.getParameterNames() ;

  for ( std::vector<std::string>::iterator iParam = parameterNames.begin();
	iParam != parameterNames.end(); iParam++ ){
    if  ( (*iParam) == "Monte" ) _Monte =  myEmParams.getParameter<bool>( *iParam );
    else if ( (*iParam) == "Debug" ) _Debug =  myEmParams.getParameter<bool>( *iParam );
  }

  const int kMaxMuon = 10000;
  muonpt = new float[kMaxMuon];
  muonphi = new float[kMaxMuon];
  muoneta = new float[kMaxMuon];
  muonet = new float[kMaxMuon];
  muone = new float[kMaxMuon];
  muonchi2NDF = new float[kMaxMuon];
  muoncharge = new float[kMaxMuon];
  muonTrkIsoR03 = new float[kMaxMuon];
  muonECalIsoR03 = new float[kMaxMuon];
  muonHCalIsoR03 = new float[kMaxMuon];
  muonD0 = new float[kMaxMuon];
  muontype = new int[kMaxMuon];
  muonNValidTrkHits = new int[kMaxMuon];
  muonNValidMuonHits = new int[kMaxMuon];
  const int kMaxMuonL2 = 500;
  muonl2pt = new float[kMaxMuonL2];
  muonl2phi = new float[kMaxMuonL2];
  muonl2eta = new float[kMaxMuonL2];
  muonl2dr = new float[kMaxMuonL2];
  muonl2drsign = new float[kMaxMuonL2];
  muonl2dz = new float[kMaxMuonL2];
  muonl2vtxz = new float[kMaxMuonL2];
  muonl2chg = new int[kMaxMuonL2];
  muonl2pterr = new float[kMaxMuonL2];
  muonl2iso = new int[kMaxMuonL2];
  muonl2nhits = new int[kMaxMuonL2];
  muonl2nchambers = new int[kMaxMuonL2]; 
  muonl2nstat = new int[kMaxMuonL2]; 
  muonl2ndtcscstat = new int[kMaxMuonL2]; 
  muonl21idx = new int[kMaxMuonL2];
  const int kMaxMuonL3 = 500;
  muonl3pt = new float[kMaxMuonL3];
  muonl3phi = new float[kMaxMuonL3];
  muonl3eta = new float[kMaxMuonL3];
  muonl3dr = new float[kMaxMuonL3];
  muonl3dz = new float[kMaxMuonL3];
  muonl3vtxz = new float[kMaxMuonL3];
  muonl3chg = new int[kMaxMuonL3];
  muonl3pterr = new float[kMaxMuonL3];
  muonl3iso = new int[kMaxMuonL3];
  muonl3trk10iso = new int[kMaxMuonL3];
  muonl3nhits = new int[kMaxMuonL3];
  muonl3normchi2 = new float[kMaxMuonL3];
  muonl3npixelhits = new int[kMaxMuonL3];
  muonl3ntrackerhits = new int[kMaxMuonL3];
  muonl3nmuonhits = new int[kMaxMuonL3];
  muonl32idx = new int[kMaxMuonL3];
  muonl3globalpt = new float[kMaxMuonL3]; 
  muonl3globaleta = new float[kMaxMuonL3];
  muonl3globalphi = new float[kMaxMuonL3];
  muonl3globaldr = new float[kMaxMuonL3];
  muonl3globaldrsign = new float[kMaxMuonL3];
  muonl3globaldz = new float[kMaxMuonL3];
  muonl3globalvtxz = new float[kMaxMuonL3];
  muonl3globalchg = new int[kMaxMuonL3];
  muonl3global2idx = new int[kMaxMuonL3];
  const int kMaxTrackerMuon = 500;
  trackermuonpt = new float[kMaxTrackerMuon];
  trackermuonphi = new float[kMaxTrackerMuon];
  trackermuoneta = new float[kMaxTrackerMuon];
  trackermuonchg = new int[kMaxTrackerMuon];
  trackermuonnhits = new int[kMaxTrackerMuon];
  const int kMaxOniaPixel = 500;
  oniaPixelpt = new float[kMaxOniaPixel];
  oniaPixelphi = new float[kMaxOniaPixel];
  oniaPixeleta = new float[kMaxOniaPixel];
  oniaPixeldr = new float[kMaxOniaPixel];
  oniaPixeldz = new float[kMaxOniaPixel];
  oniaPixelchg = new int[kMaxOniaPixel];
  oniaPixelHits = new int[kMaxOniaPixel];
  oniaPixelNormChi2 = new float[kMaxOniaPixel];
  const int kMaxTrackPixel = 500;
  oniaTrackpt = new float[kMaxTrackPixel];
  oniaTrackphi = new float[kMaxTrackPixel];
  oniaTracketa = new float[kMaxTrackPixel];
  oniaTrackdr = new float[kMaxTrackPixel];
  oniaTrackdz = new float[kMaxTrackPixel];
  oniaTrackchg = new int[kMaxTrackPixel];
  oniaTrackHits = new int[kMaxTrackPixel];
  oniaTrackNormChi2 = new float[kMaxTrackPixel];
  const int kMaxMuonL2NoVtx = 500; 
  muonl2novtxpt = new float[kMaxMuonL2NoVtx]; 
  muonl2novtxphi = new float[kMaxMuonL2NoVtx]; 
  muonl2novtxeta = new float[kMaxMuonL2NoVtx]; 
  muonl2novtxdr = new float[kMaxMuonL2NoVtx]; 
  muonl2novtxdrsign = new float[kMaxMuonL2NoVtx]; 
  muonl2novtxdz = new float[kMaxMuonL2NoVtx]; 
  muonl2novtxchg = new int[kMaxMuonL2NoVtx]; 
  muonl2novtxpterr = new float[kMaxMuonL2NoVtx]; 
  muonl2novtxnhits = new int[kMaxMuonL2NoVtx];
  muonl2novtxnchambers = new int[kMaxMuonL2NoVtx];
  muonl2novtxnstat = new int[kMaxMuonL2NoVtx];
  muonl2novtxndtcscstat = new int[kMaxMuonL2NoVtx];
  muonl2novtx1idx = new int[kMaxMuonL2NoVtx]; 
  const int kMaxDiMu = 500;
  dimudca = new float[kMaxDiMu];
  dimu1st = new int[kMaxDiMu];
  dimu2nd = new int[kMaxDiMu];
  const int kMaxDiMuVtx = 500;
  dimuvtx1st = new int[kMaxDiMuVtx];
  dimuvtx2nd = new int[kMaxDiMuVtx];
  dimuvtxchi2 = new float[kMaxDiMuVtx];
  dimuvtxr = new float[kMaxDiMuVtx];
  dimuvtxrsig = new float[kMaxDiMuVtx];
  dimuvtxroversig = new float[kMaxDiMuVtx];
  dimuvtxcosalpha = new float[kMaxDiMuVtx];
  dimuvtxmu2dipmax = new float[kMaxDiMuVtx];
  dimuvtxmu2dipmin = new float[kMaxDiMuVtx];
  dimuvtxmu2dipsigmax = new float[kMaxDiMuVtx];
  dimuvtxmu2dipsigmin = new float[kMaxDiMuVtx];

  //pf
  const int kMaxPfmuon = 10000;
  pfmuonpt = new float[kMaxPfmuon];
  pfmuonphi = new float[kMaxPfmuon];
  pfmuoneta = new float[kMaxPfmuon];
  pfmuonet = new float[kMaxPfmuon];
  pfmuone = new float[kMaxPfmuon];
  pfmuoncharge = new float[kMaxPfmuon];

  // Muon-specific branches of the tree 
  HltTree->Branch("NrecoMuon",&nmuon,"NrecoMuon/I");
  HltTree->Branch("recoMuonPt",muonpt,"recoMuonPt[NrecoMuon]/F");
  HltTree->Branch("recoMuonPhi",muonphi,"recoMuonPhi[NrecoMuon]/F");
  HltTree->Branch("recoMuonEta",muoneta,"recoMuonEta[NrecoMuon]/F");
  HltTree->Branch("recoMuonEt",muonet,"recoMuonEt[NrecoMuon]/F");
  HltTree->Branch("recoMuonE",muone,"recoMuonE[NrecoMuon]/F");
  HltTree->Branch("recoMuonChi2NDF",        muonchi2NDF,       "recoMuonChi2NDF[NrecoMuon]/F");
  HltTree->Branch("recoMuonCharge",         muoncharge  ,      "recoMuonCharge[NrecoMuon]/F");
  HltTree->Branch("recoMuonTrkIsoR03",      muonTrkIsoR03 ,    "recoMuonTrkIsoR03[NrecoMuon]/F");
  HltTree->Branch("recoMuonECalIsoR03",     muonECalIsoR03 ,   "recoMuonECalIsoR03[NrecoMuon]/F");
  HltTree->Branch("recoMuonHCalIsoR03",     muonHCalIsoR03 ,   "recoMuonHCalIsoR03[NrecoMuon]/F");
  HltTree->Branch("recoMuonD0",             muonD0 , 	       "recoMuonD0[NrecoMuon]/F");
  HltTree->Branch("recoMuonType",           muontype       ,   "recoMuonType[NrecoMuon]/I");
  HltTree->Branch("recoMuonNValidTrkHits",  muonNValidTrkHits, "recoMuonNValidTrkHits[NrecoMuon]/I");
  HltTree->Branch("recoMuonNValidMuonHits", muonNValidMuonHits,"recoMuonNValidMuonHits[NrecoMuon]/I");

  HltTree->Branch("NohMuL2",&nmu2cand,"NohMuL2/I");
  HltTree->Branch("ohMuL2Pt",muonl2pt,"ohMuL2Pt[NohMuL2]/F");
  HltTree->Branch("ohMuL2Phi",muonl2phi,"ohMuL2Phi[NohMuL2]/F");
  HltTree->Branch("ohMuL2Eta",muonl2eta,"ohMuL2Eta[NohMuL2]/F");
  HltTree->Branch("ohMuL2Chg",muonl2chg,"ohMuL2Chg[NohMuL2]/I");
  HltTree->Branch("ohMuL2PtErr",muonl2pterr,"ohMuL2PtErr[NohMuL2]/F");
  HltTree->Branch("ohMuL2Iso",muonl2iso,"ohMuL2Iso[NohMuL2]/I");
  HltTree->Branch("ohMuL2Dr",muonl2dr,"ohMuL2Dr[NohMuL2]/F");
  HltTree->Branch("ohMuL2DrSign",muonl2drsign,"ohMuL2DrSign[NohMuL2]/F");
  HltTree->Branch("ohMuL2Dz",muonl2dz,"ohMuL2Dz[NohMuL2]/F");
  HltTree->Branch("ohMuL2VtxZ",muonl2vtxz,"ohMuL2VtxZ[NohMuL2]/F");
  HltTree->Branch("ohMuL2Nhits",muonl2nhits,"ohMuL2Nhits[NohMuL2]/I");
  HltTree->Branch("ohMuL2Nchambers",muonl2nchambers,"ohMuL2Nchambers[NohMuL2]/I");   
  HltTree->Branch("ohMuL2Nstat",muonl2nstat,"ohMuL2Nstat[NohMuL2]/I");   
  HltTree->Branch("ohMuL2NDtCscStat",muonl2ndtcscstat,"ohMuL2NDtCscStat[NohMuL2]/I");   
  HltTree->Branch("ohMuL2L1idx",muonl21idx,"ohMuL2L1idx[NohMuL2]/I");   
  HltTree->Branch("NohMuL3",&nmu3cand,"NohMuL3/I");
  HltTree->Branch("ohMuL3Pt",muonl3pt,"ohMuL3Pt[NohMuL3]/F");
  HltTree->Branch("ohMuL3Phi",muonl3phi,"ohMuL3Phi[NohMuL3]/F");
  HltTree->Branch("ohMuL3Eta",muonl3eta,"ohMuL3Eta[NohMuL3]/F");
  HltTree->Branch("ohMuL3Chg",muonl3chg,"ohMuL3Chg[NohMuL3]/I");
  HltTree->Branch("ohMuL3PtErr",muonl3pterr,"ohMuL3PtErr[NohMuL3]/F");
  HltTree->Branch("ohMuL3Iso",muonl3iso,"ohMuL3Iso[NohMuL3]/I");
  HltTree->Branch("ohMuL3Trk10Iso",muonl3trk10iso,"ohMuL3Trk10Iso[NohMuL3]/I");
  HltTree->Branch("ohMuL3Dr",muonl3dr,"ohMuL3Dr[NohMuL3]/F");
  HltTree->Branch("ohMuL3Dz",muonl3dz,"ohMuL3Dz[NohMuL3]/F");
  HltTree->Branch("ohMuL3VtxZ",muonl3vtxz,"ohMuL3VtxZ[NohMuL3]/F");
  HltTree->Branch("ohMuL3Nhits",muonl3nhits,"ohMuL3Nhits[NohMuL3]/I");    
  HltTree->Branch("ohMuL3NormChi2", muonl3normchi2, "ohMuL3NormChi2[NohMuL3]/F");
  HltTree->Branch("ohMuL3Npixelhits", muonl3npixelhits, "ohMuL3Npixelhits[NohMuL3]/I"); 
  HltTree->Branch("ohMuL3Ntrackerhits", muonl3ntrackerhits, "ohMuL3Ntrackerhits[NohMuL3]/I"); 
  HltTree->Branch("ohMuL3Nmuonhits", muonl3nmuonhits, "ohMuL3Nmuonhits[NohMuL3]/I"); 
  HltTree->Branch("ohMuL3L2idx",muonl32idx,"ohMuL3L2idx[NohMuL3]/I");
  HltTree->Branch("ohMuL3globalPt",muonl3globalpt,"ohMuL3globalPt[NohMuL3]/F");
  HltTree->Branch("ohMuL3globalEta",muonl3globaleta,"ohMuL3globalEta[NohMuL3]/F");
  HltTree->Branch("ohMuL3globalPhi",muonl3globalphi,"ohMuL3globalPhi[NohMuL3]/F");
  HltTree->Branch("ohMuL3globalDr",muonl3globaldr,"ohMuL3globalDr[NohMuL3]/F");
  HltTree->Branch("ohMuL3globalDrSign",muonl3globaldrsign,"ohMuL3globalDrSign[NohMuL3]/F");
  HltTree->Branch("ohMuL3globalDz",muonl3globaldz,"ohMuL3globalDz[NohMuL3]/F");
  HltTree->Branch("ohMuL3globalVtxZ",muonl3globalvtxz,"ohMuL3globalVtxZ[NohMuL3]/F");
  HltTree->Branch("ohMuL3globalL2idx",muonl3global2idx,"ohMuL3globalL2idx[NohMuL3]/I");

  HltTree->Branch("NohOniaPixel",&nOniaPixelCand,"NohOniaPixel/I");
  HltTree->Branch("ohOniaPixelPt",oniaPixelpt,"ohOniaPixelPt[NohOniaPixel]/F");
  HltTree->Branch("ohOniaPixelPhi",oniaPixelphi,"ohOniaPixelPhi[NohOniaPixel]/F");
  HltTree->Branch("ohOniaPixelEta",oniaPixeleta,"ohOniaPixelEta[NohOniaPixel]/F");
  HltTree->Branch("ohOniaPixelChg",oniaPixelchg,"ohOniaPixelChg[NohOniaPixel]/I");
  HltTree->Branch("ohOniaPixelDr",oniaPixeldr,"ohOniaPixelDr[NohOniaPixel]/F");
  HltTree->Branch("ohOniaPixelDz",oniaPixeldz,"ohOniaPixelDz[NohOniaPixel]/F");
  HltTree->Branch("ohOniaPixelHits",oniaPixelHits,"ohOniaPixelHits[NohOniaPixel]/I");
  HltTree->Branch("ohOniaPixelNormChi2",oniaPixelNormChi2,"ohOniaPixelNormChi2[NohOniaPixel]/F");
  HltTree->Branch("NohOniaTrack",&nOniaTrackCand,"NohOniaTrack/I");
  HltTree->Branch("ohOniaTrackPt",oniaTrackpt,"ohOniaTrackPt[NohOniaTrack]/F");
  HltTree->Branch("ohOniaTrackPhi",oniaTrackphi,"ohOniaTrackPhi[NohOniaTrack]/F");
  HltTree->Branch("ohOniaTrackEta",oniaTracketa,"ohOniaTrackEta[NohOniaTrack]/F");
  HltTree->Branch("ohOniaTrackChg",oniaTrackchg,"ohOniaTrackChg[NohOniaTrack]/I");
  HltTree->Branch("ohOniaTrackDr",oniaTrackdr,"ohOniaTrackDr[NohOniaTrack]/F");
  HltTree->Branch("ohOniaTrackDz",oniaTrackdz,"ohOniaTrackDz[NohOniaTrack]/F");
  HltTree->Branch("ohOniaTrackHits",oniaTrackHits,"ohOniaTrackHits[NohOniaTrack]/I");
  HltTree->Branch("ohOniaTrackNormChi2",oniaTrackNormChi2,"ohOniaTrackNormChi2[NohOniaTrack]/F");
  HltTree->Branch("NohMuL2NoVtx",&nmu2cand,"NohMuL2NoVtx/I"); 
  HltTree->Branch("ohMuL2NoVtxPt",muonl2novtxpt,"ohMuL2NoVtxPt[NohMuL2NoVtx]/F"); 
  HltTree->Branch("ohMuL2NoVtxPhi",muonl2novtxphi,"ohMuL2NoVtxPhi[NohMuL2NoVtx]/F"); 
  HltTree->Branch("ohMuL2NoVtxEta",muonl2novtxeta,"ohMuL2NoVtxEta[NohMuL2NoVtx]/F"); 
  HltTree->Branch("ohMuL2NoVtxChg",muonl2novtxchg,"ohMuL2NoVtxChg[NohMuL2NoVtx]/I"); 
  HltTree->Branch("ohMuL2NoVtxPtErr",muonl2novtxpterr,"ohMuL2NoVtxPtErr[NohMuL2NoVtx]/F"); 
  HltTree->Branch("ohMuL2NoVtxDr",muonl2novtxdr,"ohMuL2NoVtxDr[NohMuL2NoVtx]/F"); 
  HltTree->Branch("ohMuL2NoVtxDrSign",muonl2novtxdrsign,"ohMuL2NoVtxDrSign[NohMuL2NoVtx]/F"); 
  HltTree->Branch("ohMuL2NoVtxDz",muonl2novtxdz,"ohMuL2NoVtxDz[NohMuL2NoVtx]/F"); 
  HltTree->Branch("ohMuL2NoVtxNhits",muonl2novtxnhits,"ohMuL2NoVtxNhits[NohMuL2NoVtx]/I");
  HltTree->Branch("ohMuL2NoVtxNchambers",muonl2novtxnchambers,"ohMuL2NoVtxNchambers[NohMuL2NoVtx]/I");  
  HltTree->Branch("ohMuL2NoVtxNstat",muonl2novtxnstat,"ohMuL2NoVtxNstat[NohMuL2NoVtx]/I");  
  HltTree->Branch("ohMuL2NoVtxNDtCscStat",muonl2novtxndtcscstat,"ohMuL2NoVtxNDtCscStat[NohMuL2NoVtx]/I");  
  HltTree->Branch("ohMuL2NoVtxL1idx",muonl2novtx1idx,"ohMuL2NoVtxL1idx[NohMuL2NoVtx]/I");   
  HltTree->Branch("NohDiMu",&nDiMu,"NohDiMu/I");    
  HltTree->Branch("ohDiMuDCA",dimudca,"ohDiMuDCA[NohDiMu]/F");    
  HltTree->Branch("ohDiMu1st",dimu1st,"ohDiMu1st[NohDiMu]/I");    
  HltTree->Branch("ohDiMu2nd",dimu2nd,"ohDiMu2nd[NohDiMu]/I");    
  HltTree->Branch("NohDiMuVtx",&nDiMuVtx,"NohDiMuVtx/I");    
  HltTree->Branch("ohDiMuVtx1st",dimuvtx1st,"ohDiMuVtx1st[NohDiMuVtx]/I");    
  HltTree->Branch("ohDiMuVtx2nd",dimuvtx2nd,"ohDiMuVtx2nd[NohDiMuVtx]/I");    
  HltTree->Branch("ohDiMuVtxChi2",dimuvtxchi2,"ohDiMuVtxChi2[NohDiMuVtx]/F");    
  HltTree->Branch("ohDiMuVtxR",dimuvtxr,"ohDiMuVtxR[NohDiMuVtx]/F");    
  HltTree->Branch("ohDiMuVtxRSig",dimuvtxrsig,"ohDiMuVtxRSig[NohDiMuVtx]/F");    
  HltTree->Branch("ohDiMuVtxROverSig",dimuvtxroversig,"ohDiMuVtxROverSig[NohDiMuVtx]/F");    
  HltTree->Branch("ohDiMuVtxCosAlpha",dimuvtxcosalpha,"ohDiMuVtxCosAlpha[NohDiMuVtx]/F");    
  HltTree->Branch("ohDiMuVtxMu2DIpMax",dimuvtxmu2dipmax,"ohDiMuVtxMu2DIpMax[NohDiMuVtx]/F");    
  HltTree->Branch("ohDiMuVtxMu2DIpMin",dimuvtxmu2dipmin,"ohDiMuVtxMu2DIpMin[NohDiMuVtx]/F");    
  HltTree->Branch("ohDiMuVtxMu2DIpSigMax",dimuvtxmu2dipsigmax,"ohDiMuVtxMu2DIpSigMax[NohDiMuVtx]/F");    
  HltTree->Branch("ohDiMuVtxMu2DIpSigMin",dimuvtxmu2dipsigmin,"ohDiMuVtxMu2DIpSigMin[NohDiMuVtx]/F");    
  HltTree->Branch("NohTrackerMuon",&ntrackermuoncand,"NohTrackerMuon/I");
  HltTree->Branch("ohTrackerMuonPt",trackermuonpt,"ohTrackerMuonPt[NohTrackerMuon]/F");
  HltTree->Branch("ohTrackerMuonPhi",trackermuonphi,"ohTrackerMuonPhi[NohTrackerMuon]/F");
  HltTree->Branch("ohTrackerMuonEta",trackermuoneta,"ohTrackerMuonEta[NohTrackerMuon]/F");
  HltTree->Branch("ohTrackerMuonChg",trackermuonchg,"ohTrackerMuonChg[NohTrackerMuon]/I");
  HltTree->Branch("ohTrackerMuonNhits",trackermuonnhits,"ohTrackerMuonNhits[NohTrackerMuon]/I");

  HltTree->Branch("NpfMuon",&npfmuon,"NpfMuon/I");
  HltTree->Branch("pfMuonPt",pfmuonpt,"pfMuonPt[NpfMuon]/F");
  HltTree->Branch("pfMuonPhi",pfmuonphi,"pfMuonPhi[NpfMuon]/F");
  HltTree->Branch("pfMuonEta",pfmuoneta,"pfMuonEta[NpfMuon]/F");
  HltTree->Branch("pfMuonEt",pfmuonet,"pfMuonEt[NpfMuon]/F");
  HltTree->Branch("pfMuonE",pfmuone,"pfMuonE[NpfMuon]/F");
  HltTree->Branch("pfMuonCharge",         pfmuoncharge  ,      "pfMuonCharge[NpfMuon]/F");


}

/* **Analyze the event** */
void HLTMuon::analyze(const edm::Handle<reco::MuonCollection>                 & Muon,
		      const edm::Handle<reco::PFCandidateCollection>          & pfMuon,
		      const edm::Handle<l1extra::L1MuonParticleCollection>    & MuCands1, 
		      const edm::Handle<reco::RecoChargedCandidateCollection> & MuCands2,
		      const edm::Handle<edm::ValueMap<bool> >                 & isoMap2,
		      const edm::Handle<reco::RecoChargedCandidateCollection> & MuCands3,
		      const edm::Handle<edm::ValueMap<bool> >                 & isoMap3,
		      const edm::Handle<edm::ValueMap<bool> >                 & isoTrk10Map3,
		      const edm::Handle<reco::RecoChargedCandidateCollection> & oniaPixelCands,
		      const edm::Handle<reco::RecoChargedCandidateCollection> & oniaTrackCands,
		      const edm::Handle<reco::VertexCollection> & DiMuVtxCands3,
		      const edm::Handle<reco::RecoChargedCandidateCollection> & MuNoVtxCands2, 
		      const edm::Handle<reco::MuonCollection>                 & trkmucands,
		      const edm::ESHandle<MagneticField> & theMagField,
		      const edm::Handle<reco::BeamSpot> & recoBeamSpotHandle,
		      TTree* HltTree) {

  reco::BeamSpot::Point BSPosition(0,0,0);
  BSPosition = recoBeamSpotHandle->position();
  const GlobalPoint theBeamSpot = GlobalPoint(recoBeamSpotHandle->position().x(),
					      recoBeamSpotHandle->position().y(),
					      recoBeamSpotHandle->position().z());
  reco::BeamSpot vtxBS = *recoBeamSpotHandle;

  //std::cout << " Beginning HLTMuon " << std::endl;

  if (Muon.isValid()) {
    reco::MuonCollection mymuons;
    mymuons = * Muon;
    std::sort(mymuons.begin(),mymuons.end(),PtGreater());
    nmuon = mymuons.size();
    typedef reco::MuonCollection::const_iterator muiter;
    int imu=0;
    for (muiter i=mymuons.begin(); i!=mymuons.end(); i++) 
      {
	muonpt[imu]         = i->pt();
	muonphi[imu]        = i->phi();
	muoneta[imu]        = i->eta();
	muonet[imu]         = i->et();
	muone[imu]          = i->energy(); 
	muontype[imu]       = i->type();
	muoncharge[imu]     = i->charge(); 
	muonTrkIsoR03[imu]  = i->isolationR03().sumPt;
	muonECalIsoR03[imu] = i->isolationR03().emEt;
	muonHCalIsoR03[imu] = i->isolationR03().hadEt;


	if (i->globalTrack().isNonnull())
	  {
	    muonchi2NDF[imu] = i->globalTrack()->normalizedChi2();
	    muonD0[imu] = i->globalTrack()->dxy(BSPosition);
	  }
	else 
	  {
	    muonchi2NDF[imu] = -99.;
	    muonD0[imu] = -99.;}

	if (i->innerTrack().isNonnull()) muonNValidTrkHits[imu] = i->innerTrack()->numberOfValidHits();
	else muonNValidTrkHits[imu] = -99;

	if (i->isGlobalMuon()!=0) muonNValidMuonHits[imu] = i->globalTrack()->hitPattern().numberOfValidMuonHits();
	else muonNValidMuonHits[imu] = -99;

	imu++;
      }
  }
  else {nmuon = 0;}

  l1extra::L1MuonParticleCollection myMucands1; 
  myMucands1 = * MuCands1; 
  //  reco::RecoChargedCandidateCollection myMucands1;
  std::sort(myMucands1.begin(),myMucands1.end(),PtGreater()); 

  /////////////////////////////// Open-HLT muons ///////////////////////////////

  // Dealing with L2 muons
  reco::RecoChargedCandidateCollection myMucands2;
  if (MuCands2.isValid()) {
    //     reco::RecoChargedCandidateCollection myMucands2;
    myMucands2 = * MuCands2;
    std::sort(myMucands2.begin(),myMucands2.end(),PtGreater());
    nmu2cand = myMucands2.size();
    typedef reco::RecoChargedCandidateCollection::const_iterator cand;
    int imu2c=0;
    for (cand i=myMucands2.begin(); i!=myMucands2.end(); i++) {
      reco::TrackRef tk = i->get<reco::TrackRef>();

      muonl2pt[imu2c] = tk->pt();
      // eta (we require |eta|<2.5 in all filters
      muonl2eta[imu2c] = tk->eta();
      muonl2phi[imu2c] = tk->phi();

      // Dr (transverse distance to (0,0,0))
      // For baseline triggers, we do no cut at L2 (|dr|<9999 cm)
      // However, we use |dr|<200 microns at L3, which it probably too tough for LHC startup
      muonl2dr[imu2c] = fabs(tk->dxy(BSPosition));
      muonl2drsign[imu2c] = ( tk->dxyError() > 0. ? muonl2dr[imu2c] / tk->dxyError() : 999. );

      // Dz (longitudinal distance to z=0 when at minimum transverse distance)
      // For baseline triggers, we do no cut (|dz|<9999 cm), neither at L2 nor at L3
      muonl2dz[imu2c] = tk->dz(BSPosition);
      muonl2vtxz[imu2c] = tk->dz();
      muonl2nhits[imu2c] = tk->numberOfValidHits();
      muonl2nchambers[imu2c] = validChambers(tk);
      muonl2nstat[imu2c] = tk->hitPattern().muonStationsWithAnyHits();
      muonl2ndtcscstat[imu2c] = tk->hitPattern().dtStationsWithAnyHits() + tk->hitPattern().cscStationsWithAnyHits();

      // At present we do not cut on this, but on a 90% CL value "ptLx" defined here below
      // We should change this in the future and cut directly on "pt", to avoid unnecessary complications and risks
      // Baseline cuts (HLT exercise):
      //                Relaxed Single muon:  ptLx>16 GeV
      //                Isolated Single muon: ptLx>11 GeV
      //                Relaxed Double muon: ptLx>3 GeV
      double l2_err0 = tk->error(0); // error on q/p
      double l2_abspar0 = fabs(tk->parameter(0)); // |q/p|
      //       double ptLx = tk->pt();
      // convert 50% efficiency threshold to 90% efficiency threshold
      // For L2 muons: nsigma_Pt_ = 3.9
      //       double nsigma_Pt_ = 3.9;
      // For L3 muons: nsigma_Pt_ = 2.2
      // these are the old TDR values for nsigma_Pt_
      // We know that these values are slightly smaller for CMSSW
      // But as quoted above, we want to get rid of this gymnastics in the future
      //       if (abspar0>0) ptLx += nsigma_Pt_*err0/abspar0*tk->pt();

      // Charge
      // We use the charge in some dimuon paths
      muonl2pterr[imu2c] = l2_err0/l2_abspar0;
      muonl2chg[imu2c] = tk->charge();

      if (isoMap2.isValid()){
	// Isolation flag (this is a bool value: true => isolated)
	edm::ValueMap<bool> ::value_type muon1IsIsolated = (*isoMap2)[tk];
	muonl2iso[imu2c] = muon1IsIsolated;
      }
      else {muonl2iso[imu2c] = -999;}

      l1extra::L1MuonParticleRef l1; 
      int il2 = 0; 
      //find the corresponding L1 
      l1 = tk->seedRef().castTo<edm::Ref< L2MuonTrajectorySeedCollection> >()->l1Particle();
      il2++; 
      int imu1idx = 0; 
      if (MuCands1.isValid()) { 
	typedef l1extra::L1MuonParticleCollection::const_iterator candl1; 
	for (candl1 j=myMucands1.begin(); j!=myMucands1.end(); j++) { 
	  if((j->pt() == l1->pt()) &&
	     (j->eta() == l1->eta()) &&
	     (j->phi() == l1->phi()) &&
	     (j->gmtMuonCand().quality() == l1->gmtMuonCand().quality()))
	    {break;}
	  //	  std::cout << << std::endl;
	  //          if ( tkl1 == l1 ) {break;} 
	  imu1idx++; 
	} 
      } 
      else {imu1idx = -999;} 
      muonl21idx[imu2c] = imu1idx; // Index of the L1 muon having matched with the L2 muon with index imu2c 

      imu2c++;
    }
  }
  else {nmu2cand = 0;}

  // Dealing with L3 muons
  reco::RecoChargedCandidateCollection myMucands3;
  if (MuCands3.isValid()) {
    int k = 0; 
    myMucands3 = * MuCands3;
    std::sort(myMucands3.begin(),myMucands3.end(),PtGreater());
    nmu3cand = myMucands3.size();
    typedef reco::RecoChargedCandidateCollection::const_iterator cand;
    int imu3c=0;
    int idimuc=0;
    for (cand i=myMucands3.begin(); i!=myMucands3.end(); i++) {
      reco::TrackRef tk = i->get<reco::TrackRef>();

      reco::RecoChargedCandidateRef candref = reco::RecoChargedCandidateRef(MuCands3,k);

      reco::TrackRef staTrack;
      typedef reco::MuonTrackLinksCollection::const_iterator l3muon;
      int il3 = 0;
      //find the corresponding L2 track
      staTrack = tk->seedRef().castTo<edm::Ref< L3MuonTrajectorySeedCollection> >()->l2Track();
      il3++;
      int imu2idx = 0;
      if (MuCands2.isValid()) {
	typedef reco::RecoChargedCandidateCollection::const_iterator candl2;
	for (candl2 i=myMucands2.begin(); i!=myMucands2.end(); i++) {
	  reco::TrackRef tkl2 = i->get<reco::TrackRef>();
	  if ( tkl2 == staTrack ) {break;}
	  imu2idx++;
	}
      }
      else {imu2idx = -999;}
      muonl3global2idx[imu3c] = imu2idx; // Index of the L2 muon having matched with the L3 muon with index imu3c
      muonl32idx[imu3c] = imu2idx;

      muonl3globalpt[imu3c] = tk->pt();
      muonl3pt[imu3c] = candref->pt();
      // eta (we require |eta|<2.5 in all filters
      muonl3globaleta[imu3c] = tk->eta();
      muonl3globalphi[imu3c] = tk->phi();
      muonl3eta[imu3c] = candref->eta();
      muonl3phi[imu3c] = candref->phi();

      //       // Dr (transverse distance to (0,0,0))
      //       // For baseline triggers, we do no cut at L2 (|dr|<9999 cm)
      //       // However, we use |dr|<300 microns at L3, which it probably too tough for LHC startup
      muonl3dr[imu3c] = fabs( (- (candref->vx()-BSPosition.x()) * candref->py() + (candref->vy()-BSPosition.y()) * candref->px() ) / candref->pt() );
      muonl3globaldr[imu3c] = fabs(tk->dxy(BSPosition));
      muonl3globaldrsign[imu3c] = ( tk->dxyError() > 0. ? muonl3globaldr[imu3c] / tk->dxyError() : -999. );

      //       // Dz (longitudinal distance to z=0 when at minimum transverse distance)
      //       // For baseline triggers, we do no cut (|dz|<9999 cm), neither at L2 nor at L3
      muonl3dz[imu3c] = (candref->vz()-BSPosition.z()) - ((candref->vx()-BSPosition.x())*candref->px()+(candref->vy()-BSPosition.y())*candref->py())/candref->pt() * candref->pz()/candref->pt();
      muonl3globaldz[imu3c] = tk->dz(BSPosition);

      muonl3vtxz[imu3c] = candref->vz() - ( candref->vx()*candref->px() + candref->vy()*candref->py() )/candref->pt() * candref->pz()/candref->pt();
      muonl3globalvtxz[imu3c] = tk->dz();
      //muonl3vtxz[imu3c] = candref->vz();
      //muonl3globalvtxz[imu3c] = tk->vz();

      muonl3nhits[imu3c] = tk->numberOfValidHits();  

      //       // At present we do not cut on this, but on a 90% CL value "ptLx" defined here below
      //       // We should change this in the future and cut directly on "pt", to avoid unnecessary complications and risks
      //       // Baseline cuts (HLT exercise):
      //       //                Relaxed Single muon:  ptLx>16 GeV
      //       //                Isolated Single muon: ptLx>11 GeV
      //       //                Relaxed Double muon: ptLx>3 GeV
      double l3_err0 = tk->error(0); // error on q/p
      double l3_abspar0 = fabs(tk->parameter(0)); // |q/p|
      // //       double ptLx = tk->pt();
      //       // convert 50% efficiency threshold to 90% efficiency threshold
      //       // For L2 muons: nsigma_Pt_ = 3.9
      //       // For L3 muons: nsigma_Pt_ = 2.2
      // //       double nsigma_Pt_ = 2.2;
      //       // these are the old TDR values for nsigma_Pt_
      //       // We know that these values are slightly smaller for CMSSW
      //       // But as quoted above, we want to get rid of this gymnastics in the future
      // //       if (abspar0>0) ptLx += nsigma_Pt_*err0/abspar0*tk->pt();

      // Charge
      // We use the charge in some dimuon paths
      muonl3pterr[imu3c] = l3_err0/l3_abspar0;
      muonl3globalchg[imu3c] = tk->charge();
      muonl3chg[imu3c] = candref->charge();

      muonl3normchi2[imu3c] = tk->normalizedChi2();
      muonl3npixelhits[imu3c] = tk->hitPattern().numberOfValidPixelHits();
      muonl3ntrackerhits[imu3c] = tk->hitPattern().numberOfValidTrackerHits();
      muonl3nmuonhits[imu3c] = tk->hitPattern().numberOfValidMuonHits();

      if (isoMap3.isValid()){
	// Isolation flag (this is a bool value: true => isolated)
	edm::ValueMap<bool> ::value_type muon1IsIsolated = (*isoMap3)[tk];
	muonl3iso[imu3c] = muon1IsIsolated;
      }
      else {muonl3iso[imu3c] = -999;}

      if (isoTrk10Map3.isValid()){
	// Isolation flag (this is a bool value: true => isolated) 
	edm::ValueMap<bool> ::value_type muon1IsTrk10Isolated = (*isoTrk10Map3)[tk];
	muonl3trk10iso[imu3c] = muon1IsTrk10Isolated;
      }
      else {muonl3trk10iso[imu3c] =-999;}

      //Check DCA for muon combinations
      int imu3c2nd = imu3c + 1;// This will be the index in the hltTree for the 2nd muon of the dimuon combination

      for (cand j=i; j!=myMucands3.end(); j++) if (i!=j) {//Loop over all L3 muons from the one we are already treating
	reco::TrackRef tk2nd = j->get<reco::TrackRef>();
	reco::TransientTrack transMu1(*tk, &(*theMagField) );
	reco::TransientTrack transMu2(*tk2nd, &(*theMagField) );
	TrajectoryStateClosestToPoint mu1TS = transMu1.impactPointTSCP();
	TrajectoryStateClosestToPoint mu2TS = transMu2.impactPointTSCP();
	if (mu1TS.isValid() && mu2TS.isValid()) {
	  ClosestApproachInRPhi cApp;
	  cApp.calculate(mu1TS.theState(), mu2TS.theState());
	  if (cApp.status()) {
	    dimudca[idimuc] = cApp.distance();//Save the DCA
	    dimu1st[idimuc] = imu3c;//Save which is the index in the hltTree for the 1st muon
	    dimu2nd[idimuc] = imu3c2nd;//Save which is the index in the hltTree for the 2nd muon
	    idimuc++;
	  }
	}
	imu3c2nd++;
      }

      imu3c++;
      k++; 
    }
    nDiMu = idimuc;
  }

  else {nmu3cand = 0;  nDiMu = 0;}
  // Dealing with dimu vertices
  reco::VertexCollection myDimuvtxcands3;
  if (DiMuVtxCands3.isValid()) {
    myDimuvtxcands3 = * DiMuVtxCands3;
    nDiMuVtx = myDimuvtxcands3.size();
    typedef reco::VertexCollection::const_iterator cand;
    int idimu3c=0;
    for (cand ivtx = myDimuvtxcands3.begin(); ivtx != myDimuvtxcands3.end(); ++ivtx) {
      dimuvtxchi2[idimu3c] = ivtx->normalizedChi2();
      reco::Vertex::trackRef_iterator trackIt = ivtx->tracks_begin();
      reco::TrackRef vertextkRef1 = (*trackIt).castTo<reco::TrackRef>();
      ++trackIt;
      reco::TrackRef vertextkRef2 = (*trackIt).castTo<reco::TrackRef>();
      dimuvtx2nd[idimu3c] = -1; dimuvtx1st[idimu3c] = -1;
      for (int j=0 ; j<nmu3cand ; j++){
	if(fabs(muonl3pt[j] - vertextkRef1->pt()) < 0.0001 && fabs(muonl3eta[j] - vertextkRef1->eta()) < 0.0001 && fabs(muonl3phi[j] - vertextkRef1->phi()) < 0.0001) dimuvtx1st[idimu3c] = j; 
	if(fabs(muonl3pt[j] - vertextkRef2->pt()) < 0.0001 && fabs(muonl3eta[j] - vertextkRef2->eta()) < 0.0001 && fabs(muonl3phi[j] - vertextkRef2->phi()) < 0.0001) dimuvtx2nd[idimu3c] = j; 
      }
      math::XYZVector pperp(vertextkRef1->px() + vertextkRef2->px(), 
			    vertextkRef1->py() + vertextkRef2->py(), 
			    0.);
      reco::Vertex::Point vpoint = ivtx->position();
      GlobalPoint vtxPos (vpoint.x(), vpoint.y(), vpoint.z());
      reco::Vertex::Error verr = ivtx->error();
      GlobalError vtxErr (verr.At(0,0),verr.At(1,0),verr.At(1,1),verr.At(2,0),verr.At(2,1),verr.At(2,2));
      GlobalPoint vtxDisFromBS(-1*((vtxBS.x0() - vtxPos.x()) + (vtxPos.z() - vtxBS.z0())*vtxBS.dxdz()),
			       -1*((vtxBS.y0() - vtxPos.y()) + (vtxPos.z() - vtxBS.z0())*vtxBS.dydz()), 0.0);
      dimuvtxr[idimu3c] = vtxDisFromBS.perp();
      dimuvtxrsig[idimu3c] = sqrt(vtxErr.rerr(vtxDisFromBS));
      dimuvtxroversig[idimu3c] = dimuvtxr[idimu3c]/dimuvtxrsig[idimu3c];
      reco::Vertex::Point vperp(vtxDisFromBS.x(),vtxDisFromBS.y(),0.);
      dimuvtxcosalpha[idimu3c] = vperp.Dot(pperp)/(vperp.R()*pperp.R());
      float mu1ip = -1.0;
      float mu2ip = -1.0;
      float mu1ipsig = -1.0;
      float mu2ipsig = -1.0;
      reco::TransientTrack transMu1(*vertextkRef1, &(*theMagField) );
      TrajectoryStateClosestToPoint trajMu1BS = transMu1.trajectoryStateClosestToPoint(theBeamSpot);
      if(trajMu1BS.isValid()){
	mu1ip = fabs(trajMu1BS.perigeeParameters().transverseImpactParameter());
	if(trajMu1BS.hasError()) mu1ipsig = mu1ip/trajMu1BS.perigeeError().transverseImpactParameterError();
      }
      reco::TransientTrack transMu2(*vertextkRef2, &(*theMagField) );
      TrajectoryStateClosestToPoint trajMu2BS = transMu2.trajectoryStateClosestToPoint(theBeamSpot);
      if(trajMu2BS.isValid()){
	mu2ip = fabs(trajMu2BS.perigeeParameters().transverseImpactParameter());
	if(trajMu2BS.hasError()) mu2ipsig = mu2ip/trajMu2BS.perigeeError().transverseImpactParameterError();
      }
      dimuvtxmu2dipmax[idimu3c] = fmax(mu1ip,mu2ip);
      dimuvtxmu2dipmin[idimu3c] = fmin(mu1ip,mu2ip);
      dimuvtxmu2dipsigmax[idimu3c] = fmax(mu1ipsig,mu2ipsig);
      dimuvtxmu2dipsigmin[idimu3c] = fmin(mu1ipsig,mu2ipsig);
    }


  }
  else {nDiMuVtx = 0;}


  // Dealing with L2 no-Vertex muons
  reco::RecoChargedCandidateCollection muNoVtxMucands2;
  if (MuNoVtxCands2.isValid()) {
    muNoVtxMucands2 = * MuNoVtxCands2;
    std::sort(muNoVtxMucands2.begin(),muNoVtxMucands2.end(),PtGreater());
    nmu2cand = muNoVtxMucands2.size();
    typedef reco::RecoChargedCandidateCollection::const_iterator cand;
    int imu2c=0;
    for (cand i=muNoVtxMucands2.begin(); i!=muNoVtxMucands2.end(); i++) {
      reco::TrackRef tk = i->get<reco::TrackRef>();

      muonl2novtxpt[imu2c] = tk->pt();
      muonl2novtxeta[imu2c] = tk->eta();
      muonl2novtxphi[imu2c] = tk->phi();
      muonl2novtxdr[imu2c] = fabs(tk->dxy(BSPosition));
      muonl2novtxdrsign[imu2c] = ( tk->dxyError() > 0. ? muonl2novtxdr[imu2c] / tk->dxyError() : 999. );
      muonl2novtxdz[imu2c] = tk->dz(BSPosition);
      muonl2novtxnhits[imu2c] = tk->numberOfValidHits();
      muonl2novtxnchambers[imu2c] = validChambers(tk);
      muonl2novtxnstat[imu2c] = tk->hitPattern().muonStationsWithAnyHits();
      muonl2novtxndtcscstat[imu2c] = tk->hitPattern().dtStationsWithAnyHits() + tk->hitPattern().cscStationsWithAnyHits();

      double l2_err0 = tk->error(0); // error on q/p
      double l2_abspar0 = fabs(tk->parameter(0)); // |q/p|

      muonl2novtxpterr[imu2c] = l2_err0/l2_abspar0;
      muonl2novtxchg[imu2c] = tk->charge();

      l1extra::L1MuonParticleRef l1; 
      int il2 = 0; 
      //find the corresponding L1 
      l1 = tk->seedRef().castTo<edm::Ref< L2MuonTrajectorySeedCollection> >()->l1Particle();
      il2++; 
      int imu1idx = 0; 
      if (MuCands1.isValid()) { 
	typedef l1extra::L1MuonParticleCollection::const_iterator candl1; 
	for (candl1 j=myMucands1.begin(); j!=myMucands1.end(); j++) { 
	  if((j->pt() == l1->pt()) &&
	     (j->eta() == l1->eta()) &&
	     (j->phi() == l1->phi()) &&
	     (j->gmtMuonCand().quality() == l1->gmtMuonCand().quality()))
	    {break;}
	  imu1idx++; 
	} 
      } 
      else {imu1idx = -999;} 
      muonl2novtx1idx[imu2c] = imu1idx; // Index of the L1 muon having matched with the L2 muon with index imu2c 

      imu2c++;
    }
  }
  else {nmu2cand = 0;}



  // Dealing with Onia Pixel tracks
  reco::RecoChargedCandidateCollection myOniaPixelCands;
  if (oniaPixelCands.isValid()) {
    myOniaPixelCands = * oniaPixelCands;
    std::sort(myOniaPixelCands.begin(),myOniaPixelCands.end(),PtGreater());
    nOniaPixelCand = myOniaPixelCands.size();
    typedef reco::RecoChargedCandidateCollection::const_iterator cand;
    int ic=0;
    for (cand i=myOniaPixelCands.begin(); i!=myOniaPixelCands.end(); i++) {
      reco::TrackRef tk = i->get<reco::TrackRef>();

      oniaPixelpt[ic] = tk->pt();
      oniaPixeleta[ic] = tk->eta();
      oniaPixelphi[ic] = tk->phi();
      oniaPixeldr[ic] = tk->dxy(BSPosition);
      oniaPixeldz[ic] = tk->dz(BSPosition);
      oniaPixelchg[ic] = tk->charge();
      oniaPixelHits[ic] = tk->numberOfValidHits();
      oniaPixelNormChi2[ic] = tk->normalizedChi2();

      ic++;
    }
  }
  else {nOniaPixelCand = 0;}

  // Dealing with Onia Tracks
  reco::RecoChargedCandidateCollection myOniaTrackCands;
  if (oniaTrackCands.isValid()) {
    myOniaTrackCands = * oniaTrackCands;
    std::sort(myOniaTrackCands.begin(),myOniaTrackCands.end(),PtGreater());
    nOniaTrackCand = myOniaTrackCands.size();
    typedef reco::RecoChargedCandidateCollection::const_iterator cand;
    int ic=0;
    for (cand i=myOniaTrackCands.begin(); i!=myOniaTrackCands.end(); i++) {
      reco::TrackRef tk = i->get<reco::TrackRef>();

      oniaTrackpt[ic] = tk->pt();
      oniaTracketa[ic] = tk->eta();
      oniaTrackphi[ic] = tk->phi();
      oniaTrackdr[ic] = tk->dxy(BSPosition);
      oniaTrackdz[ic] = tk->dz(BSPosition);
      oniaTrackchg[ic] = tk->charge();
      oniaTrackHits[ic] = tk->numberOfValidHits();
      oniaTrackNormChi2[ic] = tk->normalizedChi2();

      ic++;
    }
  }
  else {nOniaTrackCand = 0;}


  // Dealing with trackerMuons
  if(trkmucands.isValid()) {
    int itrackermuc=0;
    for ( unsigned int i=0; i<trkmucands->size(); ++i ){
      const reco::Muon& muon(trkmucands->at(i));
      if (muon.isTrackerMuon()) {
	trackermuonpt[itrackermuc] = muon.pt();
	trackermuoneta[itrackermuc] = muon.eta();
	trackermuonphi[itrackermuc] = muon.phi();
	trackermuonchg[itrackermuc] = muon.charge();
	if ( !muon.innerTrack().isNull() ){
	  trackermuonnhits[itrackermuc] = muon.innerTrack()->numberOfValidHits();
	}
	itrackermuc++;
      }
    }
    ntrackermuoncand=itrackermuc;
  }
  else {ntrackermuoncand = 0;}

  //////////////////////////////////////////////////////////////////////////////

  if (pfMuon.isValid()) {
    reco::PFCandidateCollection mypfmuons;
    mypfmuons = * pfMuon;
    std::sort(mypfmuons.begin(),mypfmuons.end(),PtGreater());
    npfmuon = mypfmuons.size();
    typedef reco::PFCandidateCollection::const_iterator muiter;
    int ipfmu=0;
    for (muiter i=mypfmuons.begin(); i!=mypfmuons.end(); i++) 
      {
	pfmuonpt[ipfmu]         = i->pt();
	pfmuonphi[ipfmu]        = i->phi();
	pfmuoneta[ipfmu]        = i->eta();
	pfmuonet[ipfmu]         = i->et();
	pfmuone[ipfmu]          = i->energy(); 
	pfmuoncharge[ipfmu]     = i->charge(); 
	
	ipfmu++;
      }
  }
  else {npfmuon = 0;}

}

int HLTMuon::validChambers(const reco::TrackRef & track)
{
  // count hits in chambers using std::maps
  std::map<uint32_t,int> DTchambers;
  std::map<uint32_t,int> CSCchambers;

  for (trackingRecHit_iterator hit = track->recHitsBegin();  hit != track->recHitsEnd();  ++hit) {
    if( !((*hit)->isValid()) ) continue;

    DetId id = (*hit)->geographicalId();

    if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::DT) {
      // get the DT chamber index, not the layer index, by using DTChamberId
      uint32_t index = DTChamberId(id).rawId();

      if (DTchambers.find(index) == DTchambers.end()) {
        DTchambers[index] = 0;
      }
      DTchambers[index]++;
    }

    else if (id.det() == DetId::Muon  &&  id.subdetId() == MuonSubdetId::CSC) {
      // get the CSC chamber index, not the layer index, by explicitly setting the layer id to 0
      CSCDetId id2(id);
      uint32_t index = CSCDetId(id2.endcap(), id2.station(), id2.ring(), id2.chamber(), 0);

      if (CSCchambers.find(index) == CSCchambers.end()) {
        CSCchambers[index] = 0;
      }
      CSCchambers[index]++;
    }
  }

  // count chambers that satisfy minimal numbers of hits per chamber
  int validChambers = 0;

  int minDThits = 1;
  int minCSChits = 1;

  for (std::map<uint32_t,int>::const_iterator iter = DTchambers.begin();  iter != DTchambers.end();  ++iter) {
    if (iter->second >= minDThits) {
      validChambers++;
    }
  }
  for (std::map<uint32_t,int>::const_iterator iter = CSCchambers.begin();  iter != CSCchambers.end();  ++iter) {
    if (iter->second >= minCSChits) {
      validChambers++;
    }
  }
  return validChambers;
}
