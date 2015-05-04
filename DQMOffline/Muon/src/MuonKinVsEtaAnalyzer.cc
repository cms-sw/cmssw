/*
 *  See header file for a description of this class.
 *
 *  \author S. Goy Lopez, CIEMAT 
 *  \author S. Folgueras, U. Oviedo
 */
#include "DQMOffline/Muon/interface/MuonKinVsEtaAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include <string>
#include <TMath.h>
using namespace std;
using namespace edm;

//#define DEBUG

MuonKinVsEtaAnalyzer::MuonKinVsEtaAnalyzer(const edm::ParameterSet& pSet) {
  LogTrace(metname)<<"[MuonKinVsEtaAnalyzer] Parameters initialization";
  
  parameters = pSet;

  // the services
  theService = new MuonServiceProxy(parameters.getParameter<ParameterSet>("ServiceParameters"));
  theDbe = edm::Service<DQMStore>().operator->();

  theMuonCollectionLabel_  = consumes<reco::MuonCollection>(parameters.getParameter<InputTag>("MuonCollection"));
  theVertexLabel_          = consumes<reco::VertexCollection>(parameters.getParameter<edm::InputTag>("VertexLabel"));
  theBeamSpotLabel_        = mayConsume<reco::BeamSpot>      (parameters.getParameter<edm::InputTag>("BeamSpotLabel"));

  // Parameters
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

  chiBin = parameters.getParameter<int>("chiBin");
  chiMin = parameters.getParameter<double>("chiMin");
  chiMax = parameters.getParameter<double>("chiMax");
  chiprobMin = parameters.getParameter<double>("chiprobMin");
  chiprobMax = parameters.getParameter<double>("chiprobMax");

  etaBMin = parameters.getParameter<double>("etaBMin");
  etaBMax = parameters.getParameter<double>("etaBMax");
  etaECMin = parameters.getParameter<double>("etaECMin");
  etaECMax = parameters.getParameter<double>("etaECMax");
  etaOvlpMin = parameters.getParameter<double>("etaOvlpMin");
  etaOvlpMax = parameters.getParameter<double>("etaOvlpMax");

}
MuonKinVsEtaAnalyzer::~MuonKinVsEtaAnalyzer() { 
  delete theService;
}

void MuonKinVsEtaAnalyzer::bookHistograms(DQMStore::IBooker & ibooker,
					  edm::Run const & /*iRun*/,
					  edm::EventSetup const& /*iSetup*/){
  ibooker.cd();
  ibooker.setCurrentFolder("Muons/MuonKinVsEtaAnalyzer");


  std::string EtaName;
  for(unsigned int iEtaRegion=0;iEtaRegion<4;iEtaRegion++){
    if (iEtaRegion==0) EtaName = "Barrel";   
    if (iEtaRegion==1) EtaName = "EndCap";   
    if (iEtaRegion==2) EtaName = "Overlap";
    if (iEtaRegion==3) EtaName = "";

    // monitoring of eta parameter
    etaGlbTrack.push_back(ibooker.book1D("GlbMuon_eta_"+EtaName, "#eta_{GLB} "+EtaName, etaBin, etaMin, etaMax));
    etaTrack.push_back(ibooker.book1D("TkMuon_eta_"+EtaName, "#eta_{TK} "+EtaName, etaBin, etaMin, etaMax));
    etaStaTrack.push_back(ibooker.book1D("StaMuon_eta_"+EtaName, "#eta_{STA} "+EtaName, etaBin, etaMin, etaMax));
    etaTightTrack.push_back(ibooker.book1D("TightMuon_eta_"+EtaName, "#eta_{Tight} "+EtaName, etaBin, etaMin, etaMax));
    etaLooseTrack.push_back(ibooker.book1D("LooseMuon_eta_"+EtaName, "#eta_{Loose} "+EtaName, etaBin, etaMin, etaMax));
    etaMediumTrack.push_back(ibooker.book1D("MediumMuon_eta_"+EtaName, "#eta_{Medium} "+EtaName, etaBin, etaMin, etaMax));
    etaSoftTrack.push_back(ibooker.book1D("SoftMuon_eta_"+EtaName, "#eta_{Soft} "+EtaName, etaBin, etaMin, etaMax));
    etaHighPtTrack.push_back(ibooker.book1D("HighPtMuon_eta_"+EtaName, "#eta_{HighPt} "+EtaName, etaBin, etaMin, etaMax));
    
    // monitoring of phi paramater
    phiGlbTrack.push_back(ibooker.book1D("GlbMuon_phi_"+EtaName, "#phi_{GLB} "+EtaName+ "(rad)", phiBin, phiMin, phiMax));
    phiTrack.push_back(ibooker.book1D("TkMuon_phi_"+EtaName, "#phi_{TK}" +EtaName +"(rad)", phiBin, phiMin, phiMax));
    phiStaTrack.push_back(ibooker.book1D("StaMuon_phi_"+EtaName, "#phi_{STA}"+EtaName+" (rad)", phiBin, phiMin, phiMax));
    phiTightTrack.push_back(ibooker.book1D("TightMuon_phi_"+EtaName, "#phi_{Tight}_"+EtaName, phiBin, phiMin, phiMax));
    phiLooseTrack.push_back(ibooker.book1D("LooseMuon_phi_"+EtaName, "#phi_{Loose}_"+EtaName, phiBin, phiMin, phiMax));
    phiMediumTrack.push_back(ibooker.book1D("MediumMuon_phi_"+EtaName, "#phi_{Medium}_"+EtaName, phiBin, phiMin, phiMax));
    phiSoftTrack.push_back(ibooker.book1D("SoftMuon_phi_"+EtaName, "#phi_{Soft}_"+EtaName, phiBin, phiMin, phiMax));
    phiHighPtTrack.push_back(ibooker.book1D("HighPtMuon_phi_"+EtaName, "#phi_{HighPt}_"+EtaName, phiBin, phiMin, phiMax));
    
    // monitoring of the momentum
    pGlbTrack.push_back(ibooker.book1D("GlbMuon_p_"+EtaName, "p_{GLB} "+EtaName, pBin, pMin, pMax));
    pTrack.push_back(ibooker.book1D("TkMuon_p"+EtaName, "p_{TK} "+EtaName, pBin, pMin, pMax));
    pStaTrack.push_back(ibooker.book1D("StaMuon_p"+EtaName, "p_{STA} "+EtaName, pBin, pMin, pMax));
    pTightTrack.push_back(ibooker.book1D("TightMuon_p_"+EtaName, "p_{Tight} "+EtaName, pBin, pMin, pMax));
    pLooseTrack.push_back(ibooker.book1D("LooseMuon_p_"+EtaName, "p_{Loose} "+EtaName, pBin, pMin, pMax));
    pMediumTrack.push_back(ibooker.book1D("MediumMuon_p_"+EtaName, "p_{Medium} "+EtaName, pBin, pMin, pMax));
    pSoftTrack.push_back(ibooker.book1D("SoftMuon_p_"+EtaName, "p_{Soft} "+EtaName, pBin, pMin, pMax));
    pHighPtTrack.push_back(ibooker.book1D("HighPtMuon_p_"+EtaName, "p_{HighPt} "+EtaName, pBin, pMin, pMax));

    // monitoring of the transverse momentum
    ptGlbTrack.push_back(ibooker.book1D("GlbMuon_pt_" +EtaName, "pt_{GLB} "+EtaName, ptBin, ptMin, ptMax));
    ptTrack.push_back(ibooker.book1D("TkMuon_pt_"+EtaName, "pt_{TK} "+EtaName, ptBin, ptMin, ptMax));
    ptStaTrack.push_back(ibooker.book1D("StaMuon_pt_"+EtaName, "pt_{STA} "+EtaName, ptBin, ptMin, pMax));
    ptTightTrack.push_back(ibooker.book1D("TightMuon_pt_"+EtaName, "pt_{Tight} "+EtaName, ptBin, ptMin, ptMax));
    ptLooseTrack.push_back(ibooker.book1D("LooseMuon_pt_"+EtaName, "pt_{Loose} "+EtaName, ptBin, ptMin, ptMax));
    ptMediumTrack.push_back(ibooker.book1D("MediumMuon_pt_"+EtaName, "pt_{Medium} "+EtaName, ptBin, ptMin, ptMax));
    ptSoftTrack.push_back(ibooker.book1D("SoftMuon_pt_"+EtaName, "pt_{Soft} "+EtaName, ptBin, ptMin, ptMax));
    ptHighPtTrack.push_back(ibooker.book1D("HighPtMuon_pt_"+EtaName, "pt_{HighPt} "+EtaName, ptBin, ptMin, ptMax));

    // monitoring chi2 and Prob.Chi2
    chi2GlbTrack.push_back(ibooker.book1D("GlbMuon_chi2_"+EtaName, "#chi^{2}_{GLB} " + EtaName, chiBin, chiMin, chiMax));
    chi2probGlbTrack.push_back(ibooker.book1D("GlbMuon_chi2prob_"+EtaName, "#chi^{2}_{GLB} prob." + EtaName, chiBin, chiprobMin, chiprobMax));
    chi2Track.push_back(ibooker.book1D("TkMuon_chi2_"+EtaName, "#chi^{2}_{TK} " + EtaName, chiBin, chiMin, chiMax));
    chi2probTrack.push_back(ibooker.book1D("TkMuon_chi2prob_"+EtaName, "#chi^{2}_{TK} prob." + EtaName, chiBin, chiprobMin, chiprobMax));
    chi2StaTrack.push_back(ibooker.book1D("StaMuon_chi2_"+EtaName, "#chi^{2}_{STA} " + EtaName, chiBin, chiMin, chiMax));
    chi2probStaTrack.push_back(ibooker.book1D("StaMuon_chi2prob_"+EtaName, "#chi^{2}_{STA} prob." + EtaName, chiBin, chiprobMin, chiprobMax));
    chi2TightTrack.push_back(ibooker.book1D("TightMuon_chi2_"+EtaName, "#chi^{2}_{Tight} " + EtaName, chiBin, chiMin, chiMax));
    chi2probTightTrack.push_back(ibooker.book1D("TightMuon_chi2prob_"+EtaName, "#chi^{2}_{Tight} prob." + EtaName, chiBin, chiprobMin, chiprobMax));
    chi2LooseTrack.push_back(ibooker.book1D("LooseMuon_chi2_"+EtaName, "#chi^{2}_{Loose} " + EtaName, chiBin, chiMin, chiMax));
    chi2MediumTrack.push_back(ibooker.book1D("MediumMuon_chi2_"+EtaName, "#chi^{2}_{Medium} " + EtaName, chiBin, chiMin, chiMax));
    chi2probLooseTrack.push_back(ibooker.book1D("LooseMuon_chi2prob_"+EtaName, "#chi^{2}_{Loose} prob." + EtaName, chiBin, chiprobMin, chiprobMax));
    chi2probMediumTrack.push_back(ibooker.book1D("MediumMuon_chi2prob_"+EtaName, "#chi^{2}_{Medium} prob." + EtaName, chiBin, chiprobMin, chiprobMax));
    chi2SoftTrack.push_back(ibooker.book1D("SoftMuon_chi2_"+EtaName, "#chi^{2}_{Soft} " + EtaName, chiBin, chiMin, chiMax));
    chi2probSoftTrack.push_back(ibooker.book1D("SoftMuon_chi2prob_"+EtaName, "#chi^{2}_{Soft} prob." + EtaName, chiBin, chiprobMin, chiprobMax));
    chi2HighPtTrack.push_back(ibooker.book1D("HighPtMuon_chi2_"+EtaName, "#chi^{2}_{HighPt} " + EtaName, chiBin, chiMin, chiMax));
    chi2probHighPtTrack.push_back(ibooker.book1D("HighPtMuon_chi2prob_"+EtaName, "#chi^{2}_{HighPt} prob." + EtaName, chiBin, chiprobMin, chiprobMax));
  }
}
void MuonKinVsEtaAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
  
  LogTrace(metname)<<"[MuonKinVsEtaAnalyzer] Analyze the mu in different eta regions";
  theService->update(iSetup);
  
  edm::Handle<reco::MuonCollection> muons;
  iEvent.getByToken(theMuonCollectionLabel_,muons);
  
  // =================================================================================
  // Look for the Primary Vertex (and use the BeamSpot instead, if you can't find it):
  reco::Vertex::Point posVtx;
  reco::Vertex::Error errVtx;
  unsigned int theIndexOfThePrimaryVertex = 999.;
  
  edm::Handle<reco::VertexCollection> vertex;
  iEvent.getByToken(theVertexLabel_, vertex);
  if (vertex.isValid()){
    for (unsigned int ind=0; ind<vertex->size(); ++ind) {
      if ( (*vertex)[ind].isValid() && !((*vertex)[ind].isFake()) ) {
	theIndexOfThePrimaryVertex = ind;
	break;
      }
    }
  }

  if (theIndexOfThePrimaryVertex<100) {
    posVtx = ((*vertex)[theIndexOfThePrimaryVertex]).position();
    errVtx = ((*vertex)[theIndexOfThePrimaryVertex]).error();
  }   
  else {
    LogInfo("RecoMuonValidator") << "reco::PrimaryVertex not found, use BeamSpot position instead\n";
    
    edm::Handle<reco::BeamSpot> recoBeamSpotHandle;
    iEvent.getByToken(theBeamSpotLabel_,recoBeamSpotHandle);
    reco::BeamSpot bs = *recoBeamSpotHandle;
    
    posVtx = bs.position();
    errVtx(0,0) = bs.BeamWidthX();
    errVtx(1,1) = bs.BeamWidthY();
    errVtx(2,2) = bs.sigmaZ();
  }
  
  const reco::Vertex vtx(posVtx,errVtx);
  // ==========================================================
  
#ifdef DEBUG
  cout << "[MuonKinVsEtaAnalyzer]: Analyze the mu in different eta regions" << endl;
#endif
  if (!muons.isValid()) return;
  
  for (reco::MuonCollection::const_iterator muonIt = muons->begin(); muonIt!=muons->end(); ++muonIt){
    reco::Muon recoMu = *muonIt;
    
    for(unsigned int iEtaRegion=0;iEtaRegion<4;iEtaRegion++){
      if (iEtaRegion==0) {EtaCutMin= etaBMin; EtaCutMax=etaBMax;}
      if (iEtaRegion==1) {EtaCutMin= etaECMin; EtaCutMax=etaECMax;}
      if (iEtaRegion==2) {EtaCutMin= etaOvlpMin; EtaCutMax=etaOvlpMax;} 
      if (iEtaRegion==3) {EtaCutMin= etaBMin; EtaCutMax=etaECMax;}
    
      if(recoMu.isGlobalMuon()) {
#ifdef DEBUG
	cout << "[MuonKinVsEtaAnalyzer]: The mu is global... Filling the histos" << endl;
#endif
	LogTrace(metname)<<"[MuonKinVsEtaAnalyzer] The mu is global - filling the histos";
	reco::TrackRef recoCombinedGlbTrack = recoMu.combinedMuon();
	// get the track combinig the information from both the glb fit"
	if(fabs(recoCombinedGlbTrack->eta())>EtaCutMin && fabs(recoCombinedGlbTrack->eta())<EtaCutMax){
	  etaGlbTrack[iEtaRegion]->Fill(recoCombinedGlbTrack->eta());
	  phiGlbTrack[iEtaRegion]->Fill(recoCombinedGlbTrack->phi());
	  pGlbTrack[iEtaRegion]->Fill(recoCombinedGlbTrack->p());
	  ptGlbTrack[iEtaRegion]->Fill(recoCombinedGlbTrack->pt());
	  chi2GlbTrack[iEtaRegion]->Fill(recoCombinedGlbTrack->normalizedChi2());
	  chi2probGlbTrack[iEtaRegion]->Fill(TMath::Prob(recoCombinedGlbTrack->normalizedChi2(),recoCombinedGlbTrack->ndof()));
	}
      }

      if(recoMu.isTrackerMuon()) {
#ifdef DEBUG
	cout << "[MuonKinVsEtaAnalyzer]: The mu is tracker... Filling the histos" << endl;
#endif
	LogTrace(metname)<<"[MuonKinVsEtaAnalyzer] The mu is tracker - filling the histos";
	// get the track using only the tracker data
	reco::TrackRef recoTrack = recoMu.track();
	if(fabs(recoTrack->eta())>EtaCutMin && fabs(recoTrack->eta())<EtaCutMax){
	  etaTrack[iEtaRegion]->Fill(recoTrack->eta());
	  phiTrack[iEtaRegion]->Fill(recoTrack->phi());
	  pTrack[iEtaRegion]->Fill(recoTrack->p());
	  ptTrack[iEtaRegion]->Fill(recoTrack->pt());
	  chi2Track[iEtaRegion]->Fill(recoTrack->normalizedChi2());
	  chi2probTrack[iEtaRegion]->Fill(TMath::Prob(recoTrack->normalizedChi2(),recoTrack->ndof()));
	}
      }

      if(recoMu.isStandAloneMuon()) {
#ifdef DEBUG
	cout << "[MuonKinVsEtaAnalyzer]: The mu is standlone... Filling the histos" << endl;
#endif
	LogTrace(metname)<<"[MuonKinVsEtaAnalyzer] The mu is standalone - filling the histos";
	// get the track using only the mu spectrometer data
	reco::TrackRef recoStaTrack = recoMu.standAloneMuon();
	if(fabs(recoStaTrack->eta())>EtaCutMin && fabs(recoStaTrack->eta())<EtaCutMax){
	  etaStaTrack[iEtaRegion]->Fill(recoStaTrack->eta());
	  phiStaTrack[iEtaRegion]->Fill(recoStaTrack->phi());
	  pStaTrack[iEtaRegion]->Fill(recoStaTrack->p());
	  ptStaTrack[iEtaRegion]->Fill(recoStaTrack->pt());
	  chi2StaTrack[iEtaRegion]->Fill(recoStaTrack->normalizedChi2());
	  chi2probStaTrack[iEtaRegion]->Fill(TMath::Prob(recoStaTrack->normalizedChi2(),recoStaTrack->ndof()));
	}
      }

      if ( muon::isTightMuon(recoMu, vtx) ) {
#ifdef DEBUG
	cout << "[MuonKinVsEtaAnalyzer]: The mu is tight... Filling the histos" << endl;
#endif
	LogTrace(metname)<<"[MuonKinVsEtaAnalyzer] The mu is Tight - filling the histos";
	reco::TrackRef recoTightTrack = recoMu.combinedMuon();
	if(fabs(recoTightTrack->eta())>EtaCutMin && fabs(recoTightTrack->eta())<EtaCutMax){
	  etaTightTrack[iEtaRegion]->Fill(recoTightTrack->eta());
	  phiTightTrack[iEtaRegion]->Fill(recoTightTrack->phi());
	  pTightTrack[iEtaRegion]->Fill(recoTightTrack->p());
	  ptTightTrack[iEtaRegion]->Fill(recoTightTrack->pt());
	  chi2TightTrack[iEtaRegion]->Fill(recoTightTrack->normalizedChi2());
	  chi2probTightTrack[iEtaRegion]->Fill(TMath::Prob(recoTightTrack->normalizedChi2(),recoTightTrack->ndof()));
	}
      }


      if ( muon::isLooseMuon(recoMu) ) {
#ifdef DEBUG
	cout << "[MuonKinVsEtaAnalyzer]: The mu is Loose... Filling the histos" << endl;
#endif
	LogTrace(metname)<<"[MuonKinVsEtaAnalyzer] The mu is Loose - filling the histos";
	reco::TrackRef recoLooseTrack; 

	if ( recoMu.isGlobalMuon()) recoLooseTrack = recoMu.combinedMuon();
	else recoLooseTrack = recoMu.track();

	if(fabs(recoLooseTrack->eta())>EtaCutMin && fabs(recoLooseTrack->eta())<EtaCutMax){
	  etaLooseTrack[iEtaRegion]->Fill(recoLooseTrack->eta());
	  phiLooseTrack[iEtaRegion]->Fill(recoLooseTrack->phi());
	  pLooseTrack[iEtaRegion]->Fill(recoLooseTrack->p());
	  ptLooseTrack[iEtaRegion]->Fill(recoLooseTrack->pt());
	  chi2LooseTrack[iEtaRegion]->Fill(recoLooseTrack->normalizedChi2());
	  chi2probLooseTrack[iEtaRegion]->Fill(TMath::Prob(recoLooseTrack->normalizedChi2(),recoLooseTrack->ndof()));
	}
      }

      if ( muon::isMediumMuon(recoMu) ) {
#ifdef DEBUG
	cout << "[MuonKinVsEtaAnalyzer]: The mu is Medium... Filling the histos" << endl;
#endif
	LogTrace(metname)<<"[MuonKinVsEtaAnalyzer] The mu is Medium - filling the histos";
	reco::TrackRef recoMediumTrack; 

	if ( recoMu.isGlobalMuon()) recoMediumTrack = recoMu.combinedMuon();
	else recoMediumTrack = recoMu.track();

	if(fabs(recoMediumTrack->eta())>EtaCutMin && fabs(recoMediumTrack->eta())<EtaCutMax){
	  etaMediumTrack[iEtaRegion]->Fill(recoMediumTrack->eta());
	  phiMediumTrack[iEtaRegion]->Fill(recoMediumTrack->phi());
	  pMediumTrack[iEtaRegion]->Fill(recoMediumTrack->p());
	  ptMediumTrack[iEtaRegion]->Fill(recoMediumTrack->pt());
	  chi2MediumTrack[iEtaRegion]->Fill(recoMediumTrack->normalizedChi2());
	  chi2probMediumTrack[iEtaRegion]->Fill(TMath::Prob(recoMediumTrack->normalizedChi2(),recoMediumTrack->ndof()));
	}
      }

      if ( muon::isSoftMuon(recoMu, vtx) ) {
#ifdef DEBUG
	cout << "[MuonKinVsEtaAnalyzer]: The mu is Soft... Filling the histos" << endl;
#endif
	LogTrace(metname)<<"[MuonKinVsEtaAnalyzer] The mu is Soft - filling the histos";
	reco::TrackRef recoSoftTrack = recoMu.track();
	if(fabs(recoSoftTrack->eta())>EtaCutMin && fabs(recoSoftTrack->eta())<EtaCutMax){
	  etaSoftTrack[iEtaRegion]->Fill(recoSoftTrack->eta());
	  phiSoftTrack[iEtaRegion]->Fill(recoSoftTrack->phi());
	  pSoftTrack[iEtaRegion]->Fill(recoSoftTrack->p());
	  ptSoftTrack[iEtaRegion]->Fill(recoSoftTrack->pt());
	  chi2SoftTrack[iEtaRegion]->Fill(recoSoftTrack->normalizedChi2());
	  chi2probSoftTrack[iEtaRegion]->Fill(TMath::Prob(recoSoftTrack->normalizedChi2(),recoSoftTrack->ndof()));
	}
      }

      if ( muon::isHighPtMuon(recoMu, vtx) ) {
#ifdef DEBUG
	cout << "[MuonKinVsEtaAnalyzer]: The mu is HighPt... Filling the histos" << endl;
#endif
	LogTrace(metname)<<"[MuonKinVsEtaAnalyzer] The mu is HightPt - filling the histos";
	reco::TrackRef recoHighPtTrack = recoMu.combinedMuon();
	if(fabs(recoHighPtTrack->eta())>EtaCutMin && fabs(recoHighPtTrack->eta())<EtaCutMax){
	  etaHighPtTrack[iEtaRegion]->Fill(recoHighPtTrack->eta());
	  phiHighPtTrack[iEtaRegion]->Fill(recoHighPtTrack->phi());
	  pHighPtTrack[iEtaRegion]->Fill(recoHighPtTrack->p());
	  ptHighPtTrack[iEtaRegion]->Fill(recoHighPtTrack->pt());
	  chi2HighPtTrack[iEtaRegion]->Fill(recoHighPtTrack->normalizedChi2());
	  chi2probHighPtTrack[iEtaRegion]->Fill(TMath::Prob(recoHighPtTrack->normalizedChi2(),recoHighPtTrack->ndof()));
	}
      }
    } //end iEtaRegions
  } //end recoMu iteration
}
