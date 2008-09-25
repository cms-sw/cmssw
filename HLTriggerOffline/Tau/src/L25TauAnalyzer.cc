// Class:      L25TauAnalyzer
// Original Author:  Eduardo Luiggi, modified by Sho Maruyama
//         Created:  Fri Apr  4 16:37:44 CDT 2008
// $Id: L25TauAnalyzer.cc,v 1.4 2008/05/15 19:16:58 eluiggi Exp $
#include "HLTriggerOffline/Tau/interface/L25TauAnalyzer.h"
using namespace edm;
using namespace reco;
using namespace std;
L25TauAnalyzer::L25TauAnalyzer(const edm::ParameterSet& iConfig){
edm::Service<TFileService> fs;
  l25JetSource = iConfig.getParameter<InputTag>("l25JetSource");
  l25PtCutSource = iConfig.getParameter<InputTag>("l25PtCutSource");
  l25IsoSource = iConfig.getParameter<InputTag>("l25IsoSource");
    tauSource = iConfig.getParameter<InputTag>("tauSource");
  matchingCone = iConfig.getParameter<double>("matchingCone");
  tauEta = fs->make<TH1F>("tauEta", "PF #tau #eta;#eta",50,-3.0,3.0 );
  tauPhi = fs->make <TH1F>("tauPhi", "PF #tau #phi;#phi",50,-3.5,3.5 );
  tauEt  = fs->make<TH1F>("tauEt", "PF #tau E_{T};E_{T} GeV",50,0,80 );
  tauPt  = fs->make<TH1F>("tauPt", "PF #tau Lead p_{T};p_{T} GeV/c",50,0,40 );
  tauInvPt  = fs->make<TH1F>("tauInvPt", "PF #tau Lead 1/p_{T};1/p_{T} GeV/c",50,0,0.35 );
  tauTjDR   = fs->make<TH1F>("tauTjDR", "LPF #tau #DeltaR(Lead Trk, Jet);#DeltaR",50,0,0.15 );
  tauTrkC05  = fs->make<TH1F>("tauTrkC05", "PF #tau Tracks in Cone0.5;Tracks",8,0,8 );
  tauTrkSig  = fs->make<TH1F>("tauTrkSig", "PF #tau Tracks in Signal Cone;Tracks",8,0,8 );
  tauTrkIso  = fs->make<TH1F>("tauTrkIso", "PF #tau Tracks in Isolation Annulus;Tracks",8,0,8 );
  l25Eta = fs->make<TH1F>("l25Eta", "L2.5 #tau #eta;#eta",50,-3.0,3.0 );
  l25Phi = fs->make <TH1F>("l25Phi", "L2.5 #tau #phi;#phi",50,-3.5,3.5 );
  l25Et  = fs->make<TH1F>("l25Et", "L2.5 #tau E_{T};E_{T} GeV",50,0,80 );
  l25PtCut  = fs->make<TH1F>("l25PtCut", "Number of L2.5 #tau with Lead Trk p_{T} > 3 GeV/c;Matched PF #tau E_{T} GeV",50,0,80 );
  l25Iso  = fs->make<TH1F>("l25Iso", "Number of Isolated L2.5# tau;Matched PF #tau E_{T} GeV",50,0,80 );
  l25Pt  = fs->make<TH1F>("l25Pt", "L2.5 #tau Lead p_{T};p_{T} GeV/c",50,0,40 );
  l25InvPt  = fs->make<TH1F>("l25InvPt", "L2.5 #tau Lead 1/p_{T};Matched PF #tau 1/p_{T} GeV/c",50,0,0.35 );
  l25TjDR   = fs->make<TH1F>("l25TjDR", "L2.5 #tau #DeltaR(Lead Trk, Jet);#DeltaR",50,0,0.15 );
  l25TrkQPx  = fs->make<TH1F>("l25TrkQPx", "L2.5 #tau Quality Pixel Tracks in Cone0.5;Tracks",8,0,8 );
  leadDR   = fs->make<TH1F>("leadDR", "#DeltaR(PF Lead Trk, L2.5 Lead Trk);#DeltaR",50,0,0.02 );
  Eta = fs->make<TH2F>("Eta", "#tau #eta;PF #tau #eta;L2.5 #tau #eta",50,-3.0,3.0,50,-3.0,3.0 );
  Phi = fs->make <TH2F>("Phi", "#tau #phi;PF #tau #phi;L2.5 #tau #phi",50,-3.5,3.5,50,-3.5,3.5 );
  Et  = fs->make<TH2F>("Et", "#tau E_{T};PF #tau E_{T} GeV;L2.5 #tau E_{T} GeV",50,0,80,50,0,80 );
  Pt  = fs->make<TH2F>("Pt", "#tau Lead p_{T};PF #tau Lead p_{T} GeV/c;L2.5 #tau Lead p_{T} GeV/c",50,0,40,50,0,40 );
  TjDR   = fs->make<TH2F>("TjDR", "#tau #DeltaR(Lead Trk, Jet);PF #tau #DeltaR;L2.5 #tau #DeltaR",50,0,0.15,50,0,0.15 );
  TrkC05  = fs->make<TH2F>("TrkC05", "#tau Selected Tracks;PF #tau Tracks;L2.5 #tau Tracks",8,0,8,8,0,8 );
  TrkSig  = fs->make<TH2F>("TrkSig", "#tau Signal vs Selected Tracks;PF #tau Signal Tracks;L2.5 #tau Tracks",8,0,8,8,0,8 );
  effInvPt  = fs->make<TH1F>("effInvPt","1/#tau Lead p_{T};Matched PF #tau 1/p_{T} GeV/c;Efficiency",50,0,0.35);
  effPtCut  = fs->make<TH1F>("effCutPt","L2.5 #tau Lead p_{T} Cut Efficiency;Matched PF #tau E_{T} GeV;Efficiency",50,0,80);
  effIso    = fs->make<TH1F>("effIso","Level2.5 Isolation Efficiency;Matched PF #tau E_{T} GeV;Efficiency",50,0,80);
  effDR    = fs->make<TH1F>("effDR", "Matching Efficiency #DeltaR(PF, L2.5 #tau) < 0.3;PF #tau E_{T};Efficiency",50,0,80 );
  matchDR    = fs->make<TH1F>("matchDR", "Number of Matched #tau in #DeltaR(PF, L2.5 #tau) < 0.3;#DeltaR",50,0,80 );
}

L25TauAnalyzer::~L25TauAnalyzer(){
l25InvPt ->Sumw2();
tauInvPt ->Sumw2();
l25Iso ->Sumw2();
tauEt ->Sumw2();
matchDR ->Sumw2();
l25PtCut ->Sumw2();
effDR ->Divide(tauEt, matchDR );
effInvPt ->Divide(l25InvPt, tauInvPt );
effPtCut ->Divide(l25PtCut, tauEt );
effIso ->Divide(l25Iso, tauEt );
}

void L25TauAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup){
Handle<PFTauCollection> taus;
iEvent.getByLabel(tauSource, taus);
Handle<IsolatedTauTagInfoCollection> tags;
iEvent.getByLabel(l25JetSource, tags);
Handle<CaloJetCollection> ptJets;
iEvent.getByLabel(l25PtCutSource, ptJets);
Handle<CaloJetCollection> isoJets;
iEvent.getByLabel(l25IsoSource, isoJets);
if(&(*tags)){
for(unsigned int j = 0; j < tags->size(); j++){ // bare L2.5 Taus
for(unsigned int i = 0; i < taus->size(); i++){
if( taus->at(i).pfTauTagInfoRef().isNonnull() ){
matchDR -> Fill(taus->at(i).et() );
if(deltaR(taus->at(i), *(tags->at(j).jet())) < matchingCone ){ // dr < matchingCone
tauTrkIso -> Fill( taus->at(i).isolationPFChargedHadrCands().size() );
tauTrkC05 -> Fill( taus->at(i).pfTauTagInfoRef()->PFChargedHadrCands().size() );
tauTrkSig -> Fill( taus->at(i).signalPFChargedHadrCands().size() );
tauEt  -> Fill( taus->at(i).et()  ); 			         
tauEta -> Fill( taus->at(i).eta() );			         
tauPhi -> Fill( taus->at(i).phi() );
l25TrkQPx -> Fill( tags->at(j).selectedTracks().size() );
TrkC05 -> Fill( taus->at(i).pfTauTagInfoRef()->PFChargedHadrCands().size(), tags->at(j).selectedTracks().size()); 
TrkSig -> Fill( taus->at(i).signalPFChargedHadrCands().size(), tags->at(j).selectedTracks().size()); 
l25Eta -> Fill(tags->at(j).jet()->eta());
l25Phi -> Fill(tags->at(j).jet()->phi());
l25Et  -> Fill(tags->at(j).jet()->et());
Eta -> Fill(taus->at(i).eta(), tags->at(j).jet()->eta()); 
Phi -> Fill(taus->at(i).phi(), tags->at(j).jet()->phi()); 
Et  -> Fill(taus->at(i).et(), tags->at(j).jet()->et()); 
const TrackRef leadTrk = tags->at(j).leadingSignalTrack(); 
if(leadTrk.isNonnull() ){                                                        
l25Pt -> Fill (leadTrk->pt() );
l25TjDR -> Fill( deltaR( *(tags->at(j).jet()), *leadTrk) );
}// good l25 lead trk
if(taus->at(i).leadPFChargedHadrCand().isNonnull()){
const TrackRef leadPFTrk = taus->at(i).leadPFChargedHadrCand()->trackRef();
tauPt -> Fill(leadPFTrk->pt() );			      	      
tauInvPt -> Fill(1.0/leadPFTrk->pt() );			      	      
tauTjDR -> Fill( deltaR( taus->at(i), *leadPFTrk) );
if(leadTrk.isNonnull() ){                                                        
leadDR -> Fill( deltaR( *leadTrk, *leadPFTrk) );
TjDR -> Fill(deltaR( taus->at(i), *leadPFTrk), deltaR( *(tags->at(j).jet()), *leadTrk));
Pt -> Fill(leadPFTrk->pt(), leadTrk->pt());
l25InvPt -> Fill (1.0/leadPFTrk->pt() );
}// good pf lead cand
}// good lead cand
}// pf and l25 tau match dr < matchingCone
}// good tautag
}// for tau loop
}// for jet loop
}// non empty collection

if(&(*ptJets)){ // Leading Pt Cut > 3 GeV/c applied
for(unsigned int j = 0; j < ptJets->size(); j++){
for(unsigned int i = 0; i < taus->size(); i++){
if( taus->at(i).pfTauTagInfoRef().isNonnull() ){
if(deltaR(taus->at(i), ptJets->at(j) ) < matchingCone ){ // dr < matchingCone
l25PtCut -> Fill(taus->at(i).et() );
}// pf and l25 tau match dr < matchingCone
}// good tautag
}// for tau loop
}// for jet loop
}// non empty collection

if(&(*isoJets)){
for(unsigned int j = 0; j < isoJets->size(); j++){
for(unsigned int i = 0; i < taus->size(); i++){
if(deltaR(taus->at(i), isoJets->at(j)) < matchingCone ){ // dr < matchingCone
l25Iso -> Fill(taus->at(i).et() );
}// pf and l25 tau match dr < matchingCone
}// for tau loop
}// for jet loop
}// non empty collection

}// analyzer ends here
void L25TauAnalyzer::beginJob(const edm::EventSetup&) {}
void L25TauAnalyzer::endJob() {}
