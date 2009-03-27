/**\class HeavyFlavorValidation HeavyFlavorValidation.cc HLTriggerOfflineHeavyFlavor/src/HeavyFlavorValidation.cc

 Description: Analyzer to fill Monitoring Elements for muon, dimuon and trigger path efficiency studies (HLT/RECO, RECO/GEN)

 Implementation:
     matching is based on closest in delta R, no duplicates allowed. Generated to Global based on momentum at IP; L1, L2, L2v to Global based on position in muon system, L3 to Global based on momentum at IP.
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

#include "CommonTools/Utils/interface/PtComparator.h"

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
    int getMotherId( const Candidate * p );
    void match( MonitorElement * me, vector<LeafCandidate> & from, vector<LeafCandidate> & to, double deltaRMatchingCut, vector<int> & map );
    string dqmFolder;
    InputTag genParticlesTag;   
    InputTag l1MuonsTag;   
    InputTag l1MuonsTagFast;   
    InputTag l2MuonsTag;   
    InputTag l2vMuonsTag;   
    InputTag l3MuonsTag;   
    InputTag recoMuonsTag;   
    InputTag triggerResultsTag; 
    double genFilterPtMin;
    double genFilterEtaMax;
    vector<int> motherIDs;
    int massN;
    double massLower;
    double massUpper;
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
    map<string, MonitorElement *> muonMultiplicityME;
    map<string, MonitorElement *> muonME;
    vector<MonitorElement *> triggerPathME;
    map<string, MonitorElement *> dimuonME;
    map<string, MonitorElement *> massME;
    map<string, MonitorElement *> dRME;
    double muonMass;
};

HeavyFlavorValidation::HeavyFlavorValidation(const ParameterSet& pset){
//get parameters
  dqmFolder = pset.getUntrackedParameter<string>("DQMFolder");
  genParticlesTag = pset.getParameter<InputTag>("GenParticles");
  l1MuonsTag = pset.getParameter<InputTag>("L1Muons");
  l1MuonsTagFast = pset.getParameter<InputTag>("L1MuonsFast");
  l2MuonsTag = pset.getParameter<InputTag>("L2Muons");
  l2vMuonsTag = pset.getParameter<InputTag>("L2vMuons");
  l3MuonsTag = pset.getParameter<InputTag>("L3Muons");
  recoMuonsTag = pset.getParameter<InputTag>("RecoMuons");
  triggerResultsTag = pset.getParameter<InputTag>("TriggerResults");
  genFilterPtMin = pset.getUntrackedParameter<double>("genFilterPtMin");
  genFilterEtaMax = pset.getUntrackedParameter<double>("genFilterEtaMax");
  motherIDs = pset.getUntrackedParameter<vector<int> >("motherIDs");
  massN = pset.getUntrackedParameter<int>("massN");
  massLower = pset.getUntrackedParameter<double>("massLower");
  massUpper = pset.getUntrackedParameter<double>("massUpper");
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
  muonMass = 0.106;
  
//create Monitor Elements
  dqmStore = Service<DQMStore>().operator->();  
  if( !dqmStore ){
    LogError("HLTriggerOfflineHeavyFlavor") << "Could not find DQMStore service\n";
    return;
  }
  dqmStore->setVerbose(0);

//muon multiplicity
  dqmStore->setCurrentFolder(dqmFolder+"/MuonMultiplicity");
  const int muonMultiplicityMEsize = 6;
  string muonMultiplicityMEnames[muonMultiplicityMEsize] = {
    "genMuon_size",
    "globMuon_size",
    "l1Muon_size",
    "l2Muon_size",
    "l2vMuon_size",
    "l3Muon_size"
  };
  for(int i=0;i<muonMultiplicityMEsize;i++){
    muonMultiplicityME[muonMultiplicityMEnames[i]] = dqmStore->book1D( muonMultiplicityMEnames[i].c_str(), muonMultiplicityMEnames[i].c_str(), 10,-0.5,9.5 );
  }
  
//single muons
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
  dqmStore->setCurrentFolder(dqmFolder+"/MuonMatching");
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
  
//dimuons
//pt dependence  
  dqmStore->setCurrentFolder(dqmFolder+"/DimuonEfficiencies");
  const int dimuonMEsize = 7;
  string dimuonMEnames[dimuonMEsize] = {
    "genDimuon_genPt",
    "genGlobDimuon_genPt",
    "genGlobDimuon_recoPt",
    "genGlobL1Dimuon_recoPt",
    "genGlobL1L2Dimuon_recoPt",
    "genGlobL1L2L2vDimuon_recoPt",
    "genGlobL1L2L2vL3Dimuon_recoPt"
  };
  for(int i=0;i<dimuonMEsize;i++){
    dimuonME[dimuonMEnames[i]] = dqmStore->book1D( dimuonMEnames[i], dimuonMEnames[i], ptN, ptMin, ptMax );
  }
//dR dependence  
  const int dRMEsize = 12;
  string dRMEnames[dRMEsize] = {
    "genDimuon_gendR",
    "genGlobDimuon_gendR",
    "genGlobDimuon_dR",
    "genGlobL1Dimuon_dR",
    "genGlobL1L2Dimuon_dR",
    "genGlobL1L2L2vDimuon_dR",
    "genGlobL1L2L2vL3Dimuon_dR",
    "genGlobDimuon_dRpos",
    "genGlobL1Dimuon_dRpos",
    "genGlobL1L2Dimuon_dRpos",
    "genGlobL1L2L2vDimuon_dRpos",
    "genGlobL1L2L2vL3Dimuon_dRpos"
  };
  for(int i=0;i<dRMEsize;i++){
    dRME[dRMEnames[i]] = dqmStore->book1D( dRMEnames[i].c_str(), dRMEnames[i].c_str(), 50, 0., 1. );
  }

//dimuon mass resolutions  
  dqmStore->setCurrentFolder(dqmFolder+"/MassResolutions");
  const int massMEsize = 6;
  string massMEnames[massMEsize] = {
    "genDimuon_mass",
    "genGlobDimuon_mass",
    "genGlobL1Dimuon_mass",
    "genGlobL1L2Dimuon_mass",
    "genGlobL1L2L2vDimuon_mass",
    "genGlobL1L2L2vL3Dimuon_mass"
  };
  for(int i=0;i<massMEsize;i++){
    massME[massMEnames[i]] = dqmStore->book1D( massMEnames[i].c_str(), massMEnames[i].c_str(), massN, massLower, massUpper );
  }  
}

void HeavyFlavorValidation::beginJob(const EventSetup&){
}

void HeavyFlavorValidation::analyze(const Event& iEvent, const EventSetup& iSetup){
  if( !dqmStore ){
    LogDebug("HLTriggerOfflineHeavyFlavor")<<"Could not access DQM Store service"<<endl;
    return;
  }
//access the containers and create LeafCandidate copies
  vector<LeafCandidate> genMuons;
  Handle<GenParticleCollection> genParticles;
  iEvent.getByLabel(genParticlesTag, genParticles);
  if(genParticles.isValid()){
    for(GenParticleCollection::const_iterator p=genParticles->begin(); p!= genParticles->end(); ++p){
      if( p->status() == 1 && abs(p->pdgId())==13 ){
        vector<int>::iterator mother = find( motherIDs.begin(), motherIDs.end(), getMotherId( &(*p) ) );
        if( mother != motherIDs.end() ){
          genMuons.push_back( *p );
        }
      }  
    }
  }else{
    LogDebug("HLTriggerOfflineHeavyFlavor")<<"Could not access GenParticleCollection"<<endl;
  }
  sort(genMuons.begin(), genMuons.end(), GreaterByPt<LeafCandidate>());
  muonMultiplicityME["genMuon_size"]->Fill(genMuons.size());
  LogDebug("HLTriggerOfflineHeavyFlavor")<<"GenParticleCollection from "<<genParticlesTag<<" has size: "<<genMuons.size()<<endl;
  
  vector<LeafCandidate> globMuons;
  vector<LeafCandidate> globMuons_position;
  Handle<MuonCollection> recoMuonsHandle;
  iEvent.getByLabel(recoMuonsTag, recoMuonsHandle);
  if(recoMuonsHandle.isValid()){
    for(MuonCollection::const_iterator p=recoMuonsHandle->begin(); p!= recoMuonsHandle->end(); ++p){
      if(p->isGlobalMuon()){
        globMuons.push_back(*p);
        globMuons_position.push_back( LeafCandidate( p->charge(), math::XYZTLorentzVector(p->outerTrack()->innerPosition().x(), p->outerTrack()->innerPosition().y(), p->outerTrack()->innerPosition().z(), 0.) ) );
      }
    }
  }else{
    LogDebug("HLTriggerOfflineHeavyFlavor")<<"Could not access reco Muons"<<endl;
  }
  muonMultiplicityME["globMuon_size"]->Fill(globMuons.size());
  LogDebug("HLTriggerOfflineHeavyFlavor")<<"Global Muons from "<<recoMuonsTag<<" has size: "<<globMuons.size()<<endl;

  vector<LeafCandidate> l1Muons;
  Handle<L1MuonParticleCollection> l1MuonsHandle;
  iEvent.getByLabel(l1MuonsTag, l1MuonsHandle);
  //In Fast Simulation we have a different L1 name
  if(!l1MuonsHandle.isValid()){
    iEvent.getByLabel(l1MuonsTagFast, l1MuonsHandle);
  }
  if(l1MuonsHandle.isValid()){
    for(L1MuonParticleCollection::const_iterator p=l1MuonsHandle->begin(); p!= l1MuonsHandle->end(); ++p){
      l1Muons.push_back(*p);
    }
  }else{
    LogDebug("HLTriggerOfflineHeavyFlavor")<<"Could not access L1Muons"<<endl;
  }
  muonMultiplicityME["l1Muon_size"]->Fill(l1Muons.size());
  LogDebug("HLTriggerOfflineHeavyFlavor")<<"L1 Muons from "<<l1MuonsTag<<" has size: "<<l1Muons.size()<<endl;
  
  vector<LeafCandidate> l2Muons;
  vector<LeafCandidate> l2Muons_position;
  Handle<TrackCollection> l2MuonsHandle;
  iEvent.getByLabel(l2MuonsTag, l2MuonsHandle);
  if(l2MuonsHandle.isValid()){
    for(TrackCollection::const_iterator p=l2MuonsHandle->begin(); p!= l2MuonsHandle->end(); ++p){
      l2Muons.push_back( LeafCandidate( p->charge(), math::PtEtaPhiMLorentzVector(p->pt(), p->eta(), p->phi(), muonMass) ) );
      l2Muons_position.push_back( LeafCandidate( p->charge(), math::XYZTLorentzVector(p->innerPosition().x(), p->innerPosition().y(), p->innerPosition().z(), 0.) ) );
    }
  }else{
    LogDebug("HLTriggerOfflineHeavyFlavor")<<"Could not access L2Muons"<<endl;
  }
  muonMultiplicityME["l2Muon_size"]->Fill(l2Muons.size());
  LogDebug("HLTriggerOfflineHeavyFlavor")<<"L2 Muons from "<<l2MuonsTag<<" has size: "<<l2Muons.size()<<endl;
  
  vector<LeafCandidate> l2vMuons;
  vector<LeafCandidate> l2vMuons_position;
  Handle<TrackCollection> l2vMuonsHandle;
  iEvent.getByLabel(l2vMuonsTag, l2vMuonsHandle);
  if(l2vMuonsHandle.isValid()){
    for(TrackCollection::const_iterator p=l2vMuonsHandle->begin(); p!= l2vMuonsHandle->end(); ++p){
      l2vMuons.push_back( LeafCandidate( p->charge(), math::PtEtaPhiMLorentzVector(p->pt(), p->eta(), p->phi(), muonMass) ) );
      l2vMuons_position.push_back( LeafCandidate( p->charge(), math::XYZTLorentzVector(p->innerPosition().x(), p->innerPosition().y(), p->innerPosition().z(), 0.) ) );
    }
  }else{
    LogDebug("HLTriggerOfflineHeavyFlavor")<<"Could not access L2Muons updated at vertex"<<endl;
  }
  muonMultiplicityME["l2vMuon_size"]->Fill(l2vMuons.size());
  LogDebug("HLTriggerOfflineHeavyFlavor")<<"L2 updatedAtVertex Muons from "<<l2vMuonsTag<<" has size: "<<l2vMuons.size()<<endl;

  vector<LeafCandidate> l3Muons;
  Handle<TrackCollection> l3MuonsHandle;
  iEvent.getByLabel(l3MuonsTag, l3MuonsHandle);
  if(l3MuonsHandle.isValid()){
    for(TrackCollection::const_iterator p=l3MuonsHandle->begin(); p!= l3MuonsHandle->end(); ++p){
      l3Muons.push_back( LeafCandidate( p->charge(), math::PtEtaPhiMLorentzVector(p->pt(), p->eta(), p->phi(), muonMass) ) );
    }
  }else{
    LogDebug("HLTriggerOfflineHeavyFlavor")<<"Could not access L3Muons"<<endl;
  }   
  muonMultiplicityME["l3Muon_size"]->Fill(l3Muons.size());
  LogDebug("HLTriggerOfflineHeavyFlavor")<<"L3 Muons from "<<l3MuonsTag<<" has size: "<<l3Muons.size()<<endl;
 
//create matching maps
  vector<int> glob_gen(genMuons.size(),-1);
  vector<int> l1_glob(globMuons.size(),-1);
  vector<int> l2_glob(globMuons.size(),-1);
  vector<int> l2v_glob(globMuons.size(),-1);
  vector<int> l3_glob(globMuons.size(),-1);
  match( muonME["genGlob_deltaEtaDeltaPhi"], genMuons, globMuons, genGlobDeltaRMatchingCut, glob_gen );
  match( muonME["globL1_deltaEtaDeltaPhi"], globMuons_position, l1Muons ,globL1DeltaRMatchingCut, l1_glob );
  match( muonME["globL2_deltaEtaDeltaPhi"], globMuons_position, l2Muons_position, globL2DeltaRMatchingCut, l2_glob );
  match( muonME["globL2v_deltaEtaDeltaPhi"], globMuons_position, l2vMuons_position, globL2vDeltaRMatchingCut, l2v_glob );
  match( muonME["globL3_deltaEtaDeltaPhi"], globMuons, l3Muons, globL3DeltaRMatchingCut, l3_glob );

//fill single muon histograms
  for(size_t i=0; i<genMuons.size(); i++){
    muonME["genMuon_genPtEta"]->Fill(genMuons[i].pt(), genMuons[i].eta());
    if(glob_gen[i] != -1){
      muonME["genGlobMuon_genPtEta"]->Fill(genMuons[i].pt(), genMuons[i].eta());
      muonME["genGlobMuon_recoPtEta"]->Fill(globMuons[glob_gen[i]].pt(), globMuons[glob_gen[i]].eta());
      if(l1_glob[glob_gen[i]] != -1){
        muonME["genGlobL1Muon_recoPtEta"]->Fill(globMuons[glob_gen[i]].pt(), globMuons[glob_gen[i]].eta());
        if(l2_glob[glob_gen[i]] != -1){
          muonME["genGlobL1L2Muon_recoPtEta"]->Fill(globMuons[glob_gen[i]].pt(), globMuons[glob_gen[i]].eta());
          if(l2v_glob[glob_gen[i]] != -1){
            muonME["genGlobL1L2L2vMuon_recoPtEta"]->Fill(globMuons[glob_gen[i]].pt(), globMuons[glob_gen[i]].eta());
            if(l3_glob[glob_gen[i]] != -1){
              muonME["genGlobL1L2L2vL3Muon_recoPtEta"]->Fill(globMuons[glob_gen[i]].pt(), globMuons[glob_gen[i]].eta());
            }
          }
        }
      }
    }
  }

//fill dimuon histograms (highest pT, opposite charge) 
  int secondMuon = 0;
  for(size_t j=1; j<genMuons.size(); j++){
    if(genMuons[0].charge()*genMuons[j].charge()==-1){
      secondMuon = j;
      break;
    }
  }
  if(secondMuon > 0){
//two generated
    double genDimuonPt = (genMuons[0].p4()+genMuons[secondMuon].p4()).pt();
    dimuonME["genDimuon_genPt"]->Fill( genDimuonPt );
    massME["genDimuon_mass"]->Fill( (genMuons[0].p4()+genMuons[secondMuon].p4()).mass() );
//two global
    if(glob_gen[0]!=-1 && glob_gen[secondMuon]!=-1){
      dimuonME["genGlobDimuon_genPt"]->Fill( genDimuonPt );
      double globDimuonPt = (globMuons[glob_gen[0]].p4()+globMuons[glob_gen[secondMuon]].p4()).pt();
      dimuonME["genGlobDimuon_recoPt"]->Fill( globDimuonPt );
      massME["genGlobDimuon_mass"]->Fill( (globMuons[glob_gen[0]].p4()+globMuons[glob_gen[secondMuon]].p4()).mass() );
//trigger path efficiencies wrt global dimuon   
      Handle<TriggerResults> triggerResultsHandle;
      iEvent.getByLabel(triggerResultsTag, triggerResultsHandle);
      if (triggerResultsHandle.isValid()) {
        if(triggerPathME.size()==0){ //Run this only once in the beginning
          LogDebug("HLTriggerOfflineHeavyFlavor") << "Initializing trigger path names"<<endl;
          dqmStore->setCurrentFolder(dqmFolder+"/TriggerPaths");
          TriggerNames tn(*triggerResultsHandle);
          vector<string> names = tn.triggerNames();
          for (size_t i=0; i<names.size(); i++){
            triggerPathME.push_back( dqmStore->book1D(names[i], names[i], ptN, ptMin, ptMax) );
          }
          LogDebug("HLTriggerOfflineHeavyFlavor") << "Found "<<names.size()<<" paths"<<endl;
          triggerPathME.push_back( dqmStore->book1D("denominator_genGlobDimuon_recoPt", "denominator_genGlobDimuon_recoPt", ptN, ptMin, ptMax) );
        }
        for (size_t i=0; i<triggerPathME.size()-1; i++){
          if(triggerResultsHandle->accept(i)){
            triggerPathME[i]->Fill(globDimuonPt);
          }
        }
        triggerPathME[triggerPathME.size()-1]->Fill(globDimuonPt);    
      }else{
        LogDebug("HLTriggerOfflineHeavyFlavor") << "Could not access TriggerResults"<<endl;
      }
//two l1      
      if(l1_glob[glob_gen[0]]!=-1 && l1_glob[glob_gen[secondMuon]]!=-1){
        dimuonME["genGlobL1Dimuon_recoPt"]->Fill( globDimuonPt );
        massME["genGlobL1Dimuon_mass"]->Fill( (l1Muons[l1_glob[glob_gen[0]]].p4()+l1Muons[l1_glob[glob_gen[secondMuon]]].p4()).mass() );
//two l2        
        if(l2_glob[glob_gen[0]] != -1 && l2_glob[glob_gen[secondMuon]] != -1){
          dimuonME["genGlobL1L2Dimuon_recoPt"]->Fill( globDimuonPt );
          massME["genGlobL1L2Dimuon_mass"]->Fill( (l2Muons[l2_glob[glob_gen[0]]].p4()+l2Muons[l2_glob[glob_gen[secondMuon]]].p4()).mass() );
//two l2v       
          if(l2v_glob[glob_gen[0]] != -1 && l2v_glob[glob_gen[secondMuon]] != -1){
            dimuonME["genGlobL1L2L2vDimuon_recoPt"]->Fill( globDimuonPt );
            massME["genGlobL1L2L2vDimuon_mass"]->Fill( (l2vMuons[l2v_glob[glob_gen[0]]].p4()+l2vMuons[l2v_glob[glob_gen[secondMuon]]].p4()).mass() );
//two l3         
            if(l3_glob[glob_gen[0]] != -1 && l3_glob[glob_gen[secondMuon]] != -1){
              dimuonME["genGlobL1L2L2vL3Dimuon_recoPt"]->Fill( globDimuonPt );
              massME["genGlobL1L2L2vL3Dimuon_mass"]->Fill( (l3Muons[l3_glob[glob_gen[0]]].p4()+l3Muons[l3_glob[glob_gen[secondMuon]]].p4()).mass() );
            }
          }
        }
      }
    }
//fill dR histograms when both muon pT>7
    if(genMuons[0].pt()>7. && genMuons[secondMuon].pt()>7.){
      double gendR = deltaR<LeafCandidate,LeafCandidate>(genMuons[0],genMuons[secondMuon]);
      dRME["genDimuon_gendR"]->Fill( gendR );
      if(glob_gen[0]!=-1 && glob_gen[secondMuon]!=-1){
        dRME["genGlobDimuon_gendR"]->Fill( gendR );
      }
    }
    if(glob_gen[0]!=-1 && globMuons[glob_gen[0]].pt()>7. && glob_gen[secondMuon]!=-1 && globMuons[glob_gen[secondMuon]].pt()>7.){
      double dR = deltaR<LeafCandidate,LeafCandidate>(globMuons[glob_gen[0]],globMuons[glob_gen[secondMuon]]);
      double dRpos = deltaR<LeafCandidate,LeafCandidate>(globMuons_position[glob_gen[0]],globMuons_position[glob_gen[secondMuon]]);
      dRME["genGlobDimuon_dR"]->Fill( dR );
      dRME["genGlobDimuon_dRpos"]->Fill( dRpos );
      if(l1_glob[glob_gen[0]]!=-1 && l1_glob[glob_gen[secondMuon]]!=-1){
        dRME["genGlobL1Dimuon_dR"]->Fill( dR );
        dRME["genGlobL1Dimuon_dRpos"]->Fill( dRpos );
        if(l2_glob[glob_gen[0]] != -1 && l2_glob[glob_gen[secondMuon]] != -1){
          dRME["genGlobL1L2Dimuon_dR"]->Fill( dR );
          dRME["genGlobL1L2Dimuon_dRpos"]->Fill( dRpos );
          if(l2v_glob[glob_gen[0]] != -1 && l2v_glob[glob_gen[secondMuon]] != -1){
            dRME["genGlobL1L2L2vDimuon_dR"]->Fill( dR );
            dRME["genGlobL1L2L2vDimuon_dRpos"]->Fill( dRpos );
            if(l3_glob[glob_gen[0]] != -1 && l3_glob[glob_gen[secondMuon]] != -1){
              dRME["genGlobL1L2L2vL3Dimuon_dR"]->Fill( dR );
              dRME["genGlobL1L2L2vL3Dimuon_dRpos"]->Fill( dRpos );
            }
          }
        }
      }
    }
  }    
}

void HeavyFlavorValidation::endJob(){
}

int HeavyFlavorValidation::getMotherId( const Candidate * p ){
  const Candidate* mother = p->mother();
  if( mother ){
    if( mother->pdgId() == p->pdgId() ){
      return getMotherId(mother);
    }else{
      return mother->pdgId();
    }
  }else{
    return 0;
  }
}

void HeavyFlavorValidation::match( MonitorElement * me, vector<LeafCandidate> & from, vector<LeafCandidate> & to, double dRMatchingCut, vector<int> & map ){
  vector<double> dR(from.size());
  for(size_t i=0; i<from.size(); i++){
    map[i] = -1;
    dR[i] = 10.;
    //find closest
    for(size_t j=0; j<to.size(); j++){
      double dRtmp = deltaR<double>(from[i].eta(), from[i].phi(), to[j].eta(), to[j].phi());
      if( dRtmp < dR[i] ){
        dR[i] = dRtmp;
        map[i] = j;
      }
    }
    //fill matching histo
    if( map[i] != -1 ){
      me->Fill( to[map[i]].eta()-from[i].eta(), deltaPhi<double>(to[map[i]].phi(), from[i].phi()) );
    }
    //apply matching cut
    if( dR[i] > dRMatchingCut ){
      map[i] = -1;
    }
    //remove duplication
    if( map[i] != -1 ){
      for(size_t k=0; k<i; k++){
        if( map[k] != -1 && map[i] == map[k] ){
          if( dR[i] < dR[k] ){
            map[k] = -1;
          }else{
            map[i] = -1;
          }
          break;
        }
      }
    }
  }
}

HeavyFlavorValidation::~HeavyFlavorValidation(){
}

//define this as a plug-in
DEFINE_FWK_MODULE(HeavyFlavorValidation);
