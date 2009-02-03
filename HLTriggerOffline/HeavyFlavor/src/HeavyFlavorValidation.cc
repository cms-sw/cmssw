/**\class HeavyFlavorValidation HeavyFlavorValidation.cc HLTriggerOffline/HeavyFlavor/src/HeavyFlavorValidation.cc

 Description: Analyzer to fill Monitoring Elements for muon, quarkonium and trigger path efficiency studies (HLT/RECO, RECO/GEN)

 Implementation:
     matching is based on closest in delta R, no duplicate checks yet. Generated to Global based on momentum at IP; L1, L2, L2v to Global based on position in muon system, L3 to Global based on momentum at IP.
*/
// Original Author:  Zoltan Gecse

#include <memory>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/TriggerNames.h"

#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedCandidateFwd.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticle.h"
#include "DataFormats/L1Trigger/interface/L1MuonParticleFwd.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/HepMCCandidate/interface/GenParticle.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/Math/interface/deltaR.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/Candidate/interface/Particle.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "TLorentzVector.h"

using namespace std;
using namespace edm;
using namespace reco;
using namespace l1extra;
  
class HeavyFlavorValidation : public edm::EDAnalyzer {
  public:
    explicit HeavyFlavorValidation(const edm::ParameterSet&);
    ~HeavyFlavorValidation();
  private:
    virtual void beginJob(const edm::EventSetup&) ;
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    virtual void endJob() ;
    string dqmFolder;
    InputTag genParticlesTag;   
    InputTag l1MuonsTag;   
    InputTag l2MuonsTag;   
    InputTag l2vMuonsTag;   
    InputTag l3MuonsTag;   
    InputTag recoMuonsTag;   
    InputTag triggerResultsTag; 
    double genFilterPtMin;
    double genFilterEtaMax;
    int ptN;
    double ptMin;
    double ptMax;
    int etaN;
    double etaMin;
    double etaMax;
    int deltaEtaN;
    double deltaEtaMin;
    double deltaEtaMax;
    int deltaPhiN;
    double deltaPhiMin;
    double deltaPhiMax;
    double genGlobDeltaRMatchingCut;
    double globL1DeltaRMatchingCut;
    double globL2DeltaRMatchingCut;
    double globL2vDeltaRMatchingCut;
    double globL3DeltaRMatchingCut;
    DQMStore* dqmStore;
    map<string, MonitorElement *> muonME;
    vector<MonitorElement *> triggerPathME;
    map<string, MonitorElement *> quarkoniumME;
};

HeavyFlavorValidation::HeavyFlavorValidation(const ParameterSet& pset){
//get parameters
  dqmFolder = pset.getUntrackedParameter<string>("DQMFolder");
  genParticlesTag = pset.getParameter<InputTag>("GenParticles");
  l1MuonsTag = pset.getParameter<InputTag>("L1Muons");
  l2MuonsTag = pset.getParameter<InputTag>("L2Muons");
  l2vMuonsTag = pset.getParameter<InputTag>("L2vMuons");
  l3MuonsTag = pset.getParameter<InputTag>("L3Muons");
  recoMuonsTag = pset.getParameter<InputTag>("RecoMuons");
  triggerResultsTag = pset.getParameter<InputTag>("TriggerResults");
  genFilterPtMin = pset.getUntrackedParameter<double>("genFilterPtMin");
  genFilterEtaMax = pset.getUntrackedParameter<double>("genFilterEtaMax");
  ptN = pset.getUntrackedParameter<int>("ptN");
  ptMin = pset.getUntrackedParameter<double>("ptMin");
  ptMax = pset.getUntrackedParameter<double>("ptMax");
  etaN = pset.getUntrackedParameter<int>("etaN");
  etaMin = pset.getUntrackedParameter<double>("etaMin");
  etaMax = pset.getUntrackedParameter<double>("etaMax");
  deltaEtaN = pset.getUntrackedParameter<int>("deltaEtaN");
  deltaEtaMin = pset.getUntrackedParameter<double>("deltaEtaMin");
  deltaEtaMax = pset.getUntrackedParameter<double>("deltaEtaMax");
  deltaPhiN = pset.getUntrackedParameter<int>("deltaPhiN");
  deltaPhiMin = pset.getUntrackedParameter<double>("deltaPhiMin");
  deltaPhiMax = pset.getUntrackedParameter<double>("deltaPhiMax");
  genGlobDeltaRMatchingCut = pset.getUntrackedParameter<double>("GenGlobDeltaRMatchingCut");
  globL1DeltaRMatchingCut = pset.getUntrackedParameter<double>("GlobL1DeltaRMatchingCut");
  globL2DeltaRMatchingCut = pset.getUntrackedParameter<double>("GlobL2DeltaRMatchingCut");
  globL2vDeltaRMatchingCut = pset.getUntrackedParameter<double>("GlobL2vDeltaRMatchingCut");
  globL3DeltaRMatchingCut = pset.getUntrackedParameter<double>("GlobL3DeltaRMatchingCut");
  
//create Monitor Elements
  dqmStore = Service<DQMStore>().operator->();  
  if( !dqmStore ){
    LogError("HLTriggerOfline/HeavyFlavor") << "Could not find DQMStore service\n";
    return;
  }
  dqmStore->setVerbose(0);
//  dqmStore->cd();

//muons
  dqmStore->setCurrentFolder(dqmFolder+"/MuonEfficiencies");
  const int muonMEsize = 7;
  string muonMEnames[muonMEsize] = {
    "genMuon_genPtEta",
    "genGlobMuon_genPtEta",
    "genGlobMuon_recoPtEta",
    "genGlobL1Muon_recoPtEta",
    "genGlobL1L2Muon_recoPtEta",
    "genGlobL1L2L2vMuon_recoPtEta",
    "genGlobL1L2L2vL3Muon_recoPtEta"
  };
  for(int i=0;i<muonMEsize;i++){
    muonME[muonMEnames[i]] = dqmStore->book2D( muonMEnames[i].c_str(), muonMEnames[i].c_str(), ptN, ptMin, ptMax, etaN, etaMin, etaMax );
  }
  
//matching
  const int matchingSize = 5;
  string matchingNames[matchingSize] = {
    "genGlob_deltaEtaDeltaPhi",
    "globL1_deltaEtaDeltaPhi",
    "globL2_deltaEtaDeltaPhi",
    "globL2v_deltaEtaDeltaPhi",
    "globL3_deltaEtaDeltaPhi"
  };
  for(int i=0;i<matchingSize;i++){
    muonME[matchingNames[i]] = dqmStore->book2D( matchingNames[i].c_str(), matchingNames[i].c_str(), deltaEtaN, deltaEtaMin, deltaEtaMax, deltaPhiN, deltaPhiMin, deltaPhiMax );
  }
  
//qurkonium
  dqmStore->setCurrentFolder(dqmFolder+"/QuarkoniumEfficiencies");
  const int quarkoniumMEsize = 2;
  string quarkoniumMEnames[quarkoniumMEsize] = {
    "genGlobQuarkonium_genPt",
    "genQuarkonium_genPt"
  };
  for(int i=0;i<quarkoniumMEsize;i++){
    quarkoniumME[quarkoniumMEnames[i]] = dqmStore->book1D( quarkoniumMEnames[i], quarkoniumMEnames[i], ptN, ptMin, ptMax );
  }
}

void HeavyFlavorValidation::beginJob(const EventSetup&){
}

void HeavyFlavorValidation::analyze(const Event& iEvent, const EventSetup& iSetup){
  if( !dqmStore ){
    return;
  }
//access the containers  
  Handle<GenParticleCollection> genParticles;
  iEvent.getByLabel(genParticlesTag, genParticles);
  vector<const Candidate*> genMuons;
  const Candidate * genQQ = 0;
  if(genParticles.isValid()){
    for(GenParticleCollection::const_iterator p=genParticles->begin(); p!= genParticles->end(); ++p){
      if( (p->pdgId()==443 || p->pdgId()==553) && p->status()==2 ){
        const Candidate * m0 = p->daughter(0);
        const Candidate * m1 = p->daughter(1);
        if( abs(m0->pdgId())==13 && m0->pdgId()==-m1->pdgId()
          && m0->status()==1 && m1->status()==1 
          && m0->pt()>genFilterPtMin && m1->pt()>genFilterPtMin
          && fabs(m0->eta())<genFilterEtaMax && fabs(m1->eta())<genFilterEtaMax ){
            genMuons.push_back(p->daughter(0));
            genMuons.push_back(p->daughter(1));
            genQQ = &(*p);
            break;
        }
      }  
    }
  }else{
    LogDebug("HLTriggerOfline/HeavyFlavor")<<"Could not access GenParticleCollection"<<endl;
  }
  if(!genQQ){
    return;
  }
  
  Handle<MuonCollection> recoMuonsHandle;
  iEvent.getByLabel(recoMuonsTag, recoMuonsHandle);
  MuonCollection recoMuons;
  if(recoMuonsHandle.isValid()){
    recoMuons = *recoMuonsHandle.product();
  }else{
    LogDebug("HLTriggerOfline/HeavyFlavor")<<"Could not access reco Muons"<<endl;
  }
  
  Handle<L1MuonParticleCollection> l1MuonsHandle;
  iEvent.getByLabel(l1MuonsTag, l1MuonsHandle);
  L1MuonParticleCollection l1Muons;
  if(l1MuonsHandle.isValid()){
    l1Muons = *l1MuonsHandle.product();
  }else{
    LogDebug("HLTriggerOfline/HeavyFlavor")<<"Could not access L1Muons"<<endl;
  }
  
  Handle<TrackCollection> l2MuonsHandle;
  iEvent.getByLabel(l2MuonsTag, l2MuonsHandle);
  TrackCollection l2Muons;
  if(l2MuonsHandle.isValid()){
    l2Muons = *l2MuonsHandle.product();
  }else{
    LogDebug("HLTriggerOfline/HeavyFlavor")<<"Could not access L2Muons"<<endl;
  }
  
  Handle<TrackCollection> l2vMuonsHandle;
  iEvent.getByLabel(l2vMuonsTag, l2vMuonsHandle);
  TrackCollection l2vMuons;
  if(l2vMuonsHandle.isValid()){
    l2vMuons = *l2vMuonsHandle.product();
  }else{
    LogDebug("HLTriggerOfline/HeavyFlavor")<<"Could not access L2Muons updated at vertex"<<endl;
  }
 
  Handle<TrackCollection> l3MuonsHandle;
  iEvent.getByLabel(l3MuonsTag, l3MuonsHandle);
  TrackCollection l3Muons;
  if(l3MuonsHandle.isValid()){
    l3Muons = *l3MuonsHandle.product();
  }else{
    LogDebug("HLTriggerOfline/HeavyFlavor")<<"Could not access L3Muons"<<endl;
  }   
 
//create matching maps
  vector<int> glob_gen(genMuons.size(),-1);
  vector<int> l1_glob(recoMuons.size(),-1);
  vector<int> l2_glob(recoMuons.size(),-1);
  vector<int> l2v_glob(recoMuons.size(),-1);
  vector<int> l3_glob(recoMuons.size(),-1);
  for(size_t i=0; i<genMuons.size(); i++){
    glob_gen[i] = -1;
  }
  for(size_t i=0; i<recoMuons.size(); i++){
    l1_glob[i] = -1;
    l2_glob[i] = -1;
    l2v_glob[i] = -1;
    l3_glob[i] = -1;
  }
  
//do the matching, no duplicate check for now

//glob to gen
  for(size_t i=0; i<genMuons.size(); i++){
    double dRMin = 1.;
    for(size_t j=0; j<recoMuons.size(); j++){
      double dR = deltaR<double>(genMuons[i]->eta(), genMuons[i]->phi(), recoMuons[j].eta(), recoMuons[j].phi());
      if(dR<dRMin){
        dRMin = dR;
        glob_gen[i] = j;
      }
    }
    //we want global muons only
    if(glob_gen[i]!=-1 && !recoMuons[glob_gen[i]].isGlobalMuon()){
       glob_gen[i] = -1;
    }
    if(glob_gen[i]!=-1){
      muonME["genGlob_deltaEtaDeltaPhi"]->Fill( recoMuons[glob_gen[i]].eta()-genMuons[i]->eta(), deltaPhi<double>(recoMuons[glob_gen[i]].phi(), genMuons[i]->phi()) );
    }
    if(dRMin>genGlobDeltaRMatchingCut){
      glob_gen[i] = -1;
    }   
  }
  
//hlt to glob
  for(size_t i=0; i<genMuons.size(); i++){
    if(glob_gen[i] == -1){
      continue;
    }
//l1 to glob
    double dRMin = 1.;
    for(size_t j=0; j<l1Muons.size(); j++){
      double dR = deltaR<double>(recoMuons[glob_gen[i]].outerTrack()->innerPosition().eta(), recoMuons[glob_gen[i]].outerTrack()->innerPosition().phi(), l1Muons[j].eta(), l1Muons[j].phi());
      if(dR<dRMin){
        dRMin = dR;
        l1_glob[glob_gen[i]] = j;
      }
    }
    if(l1_glob[glob_gen[i]]!=-1){
      muonME["globL1_deltaEtaDeltaPhi"]->Fill( l1Muons[l1_glob[glob_gen[i]]].eta()-recoMuons[glob_gen[i]].outerTrack()->innerPosition().eta(), deltaPhi<double>(l1Muons[l1_glob[glob_gen[i]]].phi(), recoMuons[glob_gen[i]].outerTrack()->innerPosition().phi()) );
    }
    if(dRMin>globL1DeltaRMatchingCut){
      l1_glob[glob_gen[i]] = -1;
      continue;
    }
//l2 to glob
    dRMin = 1.;
    for(size_t j=0; j<l2Muons.size(); j++){
      double dR = deltaR<double>(recoMuons[glob_gen[i]].outerTrack()->innerPosition().eta(), recoMuons[glob_gen[i]].outerTrack()->innerPosition().phi(), l2Muons[j].innerPosition().eta(), l2Muons[j].innerPosition().phi());
      if(dR<dRMin){
        dRMin = dR;
        l2_glob[glob_gen[i]] = j;
      }
    }
    if(l2_glob[glob_gen[i]]!=-1){
      muonME["globL2_deltaEtaDeltaPhi"]->Fill( l2Muons[l2_glob[glob_gen[i]]].innerPosition().eta()-recoMuons[glob_gen[i]].outerTrack()->innerPosition().eta(), deltaPhi<double>(l2Muons[l2_glob[glob_gen[i]]].innerPosition().phi(), recoMuons[glob_gen[i]].outerTrack()->innerPosition().phi()) );
    }
    if(dRMin>globL2DeltaRMatchingCut){
      l2_glob[glob_gen[i]] = -1;
      continue;
    }
//l2v to glob
    dRMin = 1.;
    for(size_t j=0; j<l2vMuons.size(); j++){
      double dR = deltaR<double>(recoMuons[glob_gen[i]].outerTrack()->innerPosition().eta(), recoMuons[glob_gen[i]].outerTrack()->innerPosition().phi(), l2vMuons[j].innerPosition().eta(), l2vMuons[j].innerPosition().phi());
      if(dR<dRMin){
        dRMin = dR;
        l2v_glob[glob_gen[i]] = j;
      }
    }
    if(l2v_glob[glob_gen[i]]!=-1){
      muonME["globL2v_deltaEtaDeltaPhi"]->Fill( l2vMuons[l2v_glob[glob_gen[i]]].innerPosition().eta()-recoMuons[glob_gen[i]].outerTrack()->innerPosition().eta(), deltaPhi<double>(l2vMuons[l2v_glob[glob_gen[i]]].innerPosition().phi(), recoMuons[glob_gen[i]].outerTrack()->innerPosition().phi()) );
    }
    if(dRMin>globL2vDeltaRMatchingCut){
      l2v_glob[glob_gen[i]] = -1;
      continue;
    }
//l3 to glob
    dRMin = 1.;
    for(size_t j=0; j<l3Muons.size(); j++){
      double dR = deltaR<double>(recoMuons[glob_gen[i]].eta(), recoMuons[glob_gen[i]].phi(), l3Muons[j].eta(), l3Muons[j].phi());
      if(dR<dRMin){
        dRMin = dR;
        l3_glob[glob_gen[i]] = j;
      }
    }
    if(l3_glob[glob_gen[i]]!=-1){
      muonME["globL3_deltaEtaDeltaPhi"]->Fill( l3Muons[l3_glob[glob_gen[i]]].eta()-recoMuons[glob_gen[i]].eta(), deltaPhi<double>(l3Muons[l3_glob[glob_gen[i]]].phi(), recoMuons[glob_gen[i]].phi()) );
    }
    if(dRMin>globL3DeltaRMatchingCut){
      l3_glob[glob_gen[i]] = -1;
    }
  }

//fill the Monitoring Elementss

  for(size_t i=0; i<genMuons.size(); i++){
    muonME["genMuon_genPtEta"]->Fill(genMuons[i]->pt(), genMuons[i]->eta());
    if(glob_gen[i] != -1){
      muonME["genGlobMuon_genPtEta"]->Fill(genMuons[i]->pt(), genMuons[i]->eta());
      muonME["genGlobMuon_recoPtEta"]->Fill(recoMuons[glob_gen[i]].pt(), recoMuons[glob_gen[i]].eta());
      if(l1_glob[glob_gen[i]] != -1){
        muonME["genGlobL1Muon_recoPtEta"]->Fill(recoMuons[glob_gen[i]].pt(), recoMuons[glob_gen[i]].eta());
        if(l2_glob[glob_gen[i]] != -1){
          muonME["genGlobL1L2Muon_recoPtEta"]->Fill(recoMuons[glob_gen[i]].pt(), recoMuons[glob_gen[i]].eta());
          if(l2v_glob[glob_gen[i]] != -1){
            muonME["genGlobL1L2L2vMuon_recoPtEta"]->Fill(recoMuons[glob_gen[i]].pt(), recoMuons[glob_gen[i]].eta());
            if(l3_glob[glob_gen[i]] != -1){
              muonME["genGlobL1L2L2vL3Muon_recoPtEta"]->Fill(recoMuons[glob_gen[i]].pt(), recoMuons[glob_gen[i]].eta());
            }
          }
        }
      }
    }
  }

//Quarkonium Efficiencies
  quarkoniumME["genQuarkonium_genPt"]->Fill( genQQ->pt() );

  if(glob_gen[0]!=-1 && glob_gen[1]!=-1){
    quarkoniumME["genGlobQuarkonium_genPt"]->Fill( genQQ->pt() );
    
//Trigger Efficiencies wrt reco quarkonium
    TLorentzVector m0, m1, qq;
    m0.SetXYZM(recoMuons[glob_gen[0]].px(), recoMuons[glob_gen[0]].py(), recoMuons[glob_gen[0]].pz(), 0.1);
    m1.SetXYZM(recoMuons[glob_gen[1]].px(), recoMuons[glob_gen[1]].py(), recoMuons[glob_gen[1]].pz(), 0.1);
    qq = m0+m1;
    double qqPt = qq.Pt();
    
    Handle<TriggerResults> triggerResultsHandle;
    iEvent.getByLabel(triggerResultsTag, triggerResultsHandle);
    if (triggerResultsHandle.isValid()) {
      if(triggerPathME.size()==0){ //Run this only once in the beginning
        dqmStore->setCurrentFolder(dqmFolder+"/TriggerPaths");
        TriggerNames tn(*triggerResultsHandle);
        vector<string> names = tn.triggerNames();
        for (size_t i=0; i<names.size(); i++){
          triggerPathME.push_back( dqmStore->book1D(names[i], names[i], ptN, ptMin, ptMax) );
        }
        triggerPathME.push_back( dqmStore->book1D("denominator_genGlobQuarkonium_recoPt", "denominator_genGlobQuarkonium_recoPt", ptN, ptMin, ptMax) );
      }
      for (size_t i=0; i<triggerPathME.size()-1; i++){
        if(triggerResultsHandle->accept(i)){
          triggerPathME[i]->Fill(qqPt);
        }
      }
      triggerPathME[triggerPathME.size()-1]->Fill(qqPt);    
    }else{
      LogDebug("HLTriggerOfline/HeavyFlavor") << "Could not access TriggerResults"<<endl;
    }
  }
}

void HeavyFlavorValidation::endJob(){
}

HeavyFlavorValidation::~HeavyFlavorValidation(){
}

//define this as a plug-in
DEFINE_FWK_MODULE(HeavyFlavorValidation);
