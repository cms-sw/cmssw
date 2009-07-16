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
#include "DataFormats/HLTReco/interface/TriggerEventWithRefs.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

#include "CommonTools/Utils/interface/PtComparator.h"

#include "TLorentzVector.h"

using namespace std;
using namespace edm;
using namespace reco;
using namespace l1extra;
using namespace trigger;

class HeavyFlavorValidation : public edm::EDAnalyzer {
  public:
    explicit HeavyFlavorValidation(const edm::ParameterSet&);
    ~HeavyFlavorValidation();
  private:
    virtual void beginJob(const edm::EventSetup&) ;
    virtual void analyze(const edm::Event&, const edm::EventSetup&);
    virtual void endJob() ;
    bool containsIndex(vector<RecoChargedCandidateRef> &v, size_t i);
    int getMotherId( const Candidate * p );
    void match( MonitorElement * me, vector<LeafCandidate> & from, vector<LeafCandidate> & to, double deltaRMatchingCut, vector<int> & map );
    string processName;
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
    vector<float> muonPtBins;
    vector<float> dimuonPtBins;
    vector<float> muonEtaBins;
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
//     map<string, MonitorElement *> muonMultiplicityME;
    vector<map<string, MonitorElement *> > pathME;
    map<string, MonitorElement *> offlineME;
//     vector<MonitorElement *> triggerPathME;
//     map<string, MonitorElement *> dimuonME;
    map<string, MonitorElement *> massME;
    map<string, MonitorElement *> dRME;
    map<string, MonitorElement *> matchingME;
    double muonMass;
    vector<vector<string> > filterNames;
    vector<int> pathIndices;
};

HeavyFlavorValidation::HeavyFlavorValidation(const ParameterSet& pset){
//get parameters
  processName = pset.getUntrackedParameter<string>("ProcessName");
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
  vector<double> tmp = pset.getUntrackedParameter<vector<double> >("muonPtBins");
  for(size_t i=0;i<tmp.size();i++){
    muonPtBins.push_back(tmp[i]);
  }
  tmp = pset.getUntrackedParameter<vector<double> >("dimuonPtBins");
  for(size_t i=0;i<tmp.size();i++){
    dimuonPtBins.push_back(tmp[i]);
  }
  tmp = pset.getUntrackedParameter<vector<double> >("muonEtaBins");
  for(size_t i=0;i<tmp.size();i++){
    muonEtaBins.push_back(tmp[i]);
  }
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
  
//discover HLT configuration
  HLTConfigProvider hltConfig;
  hltConfig.init(processName);
  vector<string> triggerNames = hltConfig.triggerNames();
  for( size_t i = 0; i < triggerNames.size(); i++) {
    TString triggerName = triggerNames[i];
    if (triggerName.Contains("Mu") && !triggerName.Contains("Iso") && !TString(triggerName(4,triggerName.Length())).Contains("_") && !(TString(triggerName(triggerName.First("Mu")+2,triggerName.Length())).Atoi()>3) ){ 
      vector<string> filters(4);
      filters[0] = triggerNames[i];
      vector<string> moduleNames = hltConfig.moduleLabels( triggerNames[i] );
      for( size_t j = 0; j < moduleNames.size(); j++) {
        TString name = moduleNames[j];
        if(name.Contains("Filtered") && !name.Contains("Iso")){
          if(name.Contains("L1")){
            filters[1] = name;
          }
          if(name.Contains("L2")){
            filters[2] = name;
          }
          if(name.Contains("L3")){
            filters[3] = name;
          }
        }
      }
      filterNames.push_back(filters);
      pathIndices.push_back(i);
    }
  }
  for(size_t i=0; i<filterNames.size(); i++){
    LogDebug("HLTriggerOfflineHeavyFlavor")<<"Trigger Path: "<<filterNames[i][0]<<" Filters: "<<filterNames[i][1]<<",  "<<filterNames[i][2]<<", "<<filterNames[i][3]<<endl;
  }

//create Monitor Elements
  dqmStore = Service<DQMStore>().operator->();  
  if( !dqmStore ){
    LogError("HLTriggerOfflineHeavyFlavor") << "Could not find DQMStore service\n";
    return;
  }
  dqmStore->setVerbose(0);
 
//per path monitoring elements
  const int muonMultiplicityMEsize = 3;
  string muonMultiplicityMEnames[muonMultiplicityMEsize] = {
    "l1Muon_size",
    "l2vMuon_size",
    "l3Muon_size"
  };
  const int muonMEsize = 3;
  string muonMEnames[muonMEsize] = {
    "genGlobL1Muon_recoPtEta",
    "genGlobL1L2L2vMuon_recoPtEta",
    "genGlobL1L2L2vL3Muon_recoPtEta"
  };
  const int dimuonMEsize = 4;
  string dimuonMEnames[dimuonMEsize] = {
    "genGlobL1Dimuon_recoPt",
    "genGlobL1L2L2vDimuon_recoPt",
    "genGlobL1L2L2vL3Dimuon_recoPt",
    "genGlobDimuonPath_recoPt"
  };
  for(size_t path=0; path<filterNames.size(); path++){
    dqmStore->setCurrentFolder((dqmFolder+"/")+filterNames[path][0]);
    map<string, MonitorElement *> tmp;
    for(int i=0;i<muonMultiplicityMEsize;i++){
      tmp[muonMultiplicityMEnames[i]] = dqmStore->book1D( muonMultiplicityMEnames[i], muonMultiplicityMEnames[i], 10,-0.5,9.5 );
    }
    for(int i=0;i<muonMEsize;i++){
      tmp[muonMEnames[i]] = dqmStore->book2D( muonMEnames[i], muonMEnames[i], muonPtBins.size()-1, &muonPtBins[0], muonEtaBins.size()-1, &muonEtaBins[0] );
    }
    for(int i=0;i<dimuonMEsize;i++){
      tmp[dimuonMEnames[i]] = dqmStore->book1D( dimuonMEnames[i], dimuonMEnames[i], dimuonPtBins.size()-1, &dimuonPtBins[0] );
    }
    pathME.push_back(tmp);
  }
  
//per event offline monitoring elements
  dqmStore->setCurrentFolder(dqmFolder+"/OfflineMuons");
  const int offlineMuonMultiplicityMEsize = 6;
  string offlineMuonMultiplicityMEnames[offlineMuonMultiplicityMEsize] = {
    "genMuon_size",
    "globMuon_size",
    "l1Muon_size",
    "l2Muon_size",
    "l2vMuon_size",
    "l3Muon_size"
  };
  for(int i=0;i<offlineMuonMultiplicityMEsize;i++){
    offlineME[offlineMuonMultiplicityMEnames[i]] = dqmStore->book1D( offlineMuonMultiplicityMEnames[i], offlineMuonMultiplicityMEnames[i], 10,-0.5,9.5 );
  }
  const int offlineMEsize = 6;
  string offlineMEnames[offlineMEsize] = {
    "genMuon_genPtEta",
    "genGlobMuon_genPtEta",
    "genGlobMuon_recoPtEta",
    "genDimuon_genPt",
    "genGlobDimuon_genPt",
    "genGlobDimuon_recoPt"
  };
  for(int i=0;i<3;i++){
    offlineME[offlineMEnames[i]] = dqmStore->book2D( offlineMEnames[i], offlineMEnames[i], muonPtBins.size()-1, &muonPtBins[0], muonEtaBins.size()-1, &muonEtaBins[0] );
  }
  for(int i=3;i<offlineMEsize;i++){
    offlineME[offlineMEnames[i]] = dqmStore->book1D( offlineMEnames[i], offlineMEnames[i], dimuonPtBins.size()-1, &dimuonPtBins[0] );
  }

//dR dependence  
  dqmStore->setCurrentFolder(dqmFolder+"/dR");
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
    dRME[dRMEnames[i]] = dqmStore->book1D( dRMEnames[i], dRMEnames[i], 50, 0., 1. );
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
    matchingME[matchingNames[i]] = dqmStore->book2D( matchingNames[i], matchingNames[i], deltaEtaN, deltaEtaMin, deltaEtaMax, deltaPhiN, deltaPhiMin, deltaPhiMax );
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
    massME[massMEnames[i]] = dqmStore->book1D( massMEnames[i], massMEnames[i], massN, massLower, massUpper );
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
      if( p->status() == 1 && abs(p->pdgId())==13 && find( motherIDs.begin(), motherIDs.end(), getMotherId(&(*p)) )!=motherIDs.end() ){
        genMuons.push_back( *p );
      }  
    }
  }else{
    LogDebug("HLTriggerOfflineHeavyFlavor")<<"Could not access GenParticleCollection"<<endl;
  }
  sort(genMuons.begin(), genMuons.end(), GreaterByPt<LeafCandidate>());
  offlineME["genMuon_size"]->Fill(genMuons.size());
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
  offlineME["globMuon_size"]->Fill(globMuons.size());
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
  offlineME["l1Muon_size"]->Fill(l1Muons.size());
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
  offlineME["l2Muon_size"]->Fill(l2Muons.size());
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
  offlineME["l2vMuon_size"]->Fill(l2vMuons.size());
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
  offlineME["l3Muon_size"]->Fill(l3Muons.size());
  LogDebug("HLTriggerOfflineHeavyFlavor")<<"L3 Muons from "<<l3MuonsTag<<" has size: "<<l3Muons.size()<<endl;
 
//create matching maps
  vector<int> glob_gen(genMuons.size(),-1);
  vector<int> l1_glob(globMuons.size(),-1);
  vector<int> l2_glob(globMuons.size(),-1);
  vector<int> l2v_glob(globMuons.size(),-1);
  vector<int> l3_glob(globMuons.size(),-1);
  match( matchingME["genGlob_deltaEtaDeltaPhi"], genMuons, globMuons, genGlobDeltaRMatchingCut, glob_gen );
  match( matchingME["globL1_deltaEtaDeltaPhi"], globMuons_position, l1Muons ,globL1DeltaRMatchingCut, l1_glob );
  match( matchingME["globL2_deltaEtaDeltaPhi"], globMuons_position, l2Muons_position, globL2DeltaRMatchingCut, l2_glob );
  match( matchingME["globL2v_deltaEtaDeltaPhi"], globMuons_position, l2vMuons_position, globL2vDeltaRMatchingCut, l2v_glob );
  match( matchingME["globL3_deltaEtaDeltaPhi"], globMuons, l3Muons, globL3DeltaRMatchingCut, l3_glob );

//get the trigger event and copy trigger decisions 
  vector<vector<L1MuonParticleRef> > l1Cands;
  vector<vector<RecoChargedCandidateRef> > l2Cands;
  vector<vector<RecoChargedCandidateRef> > l3Cands;
  for(size_t path=0; path<filterNames.size(); path++){
    l1Cands.push_back( vector<L1MuonParticleRef>() );
    l2Cands.push_back( vector<RecoChargedCandidateRef>() );
    l3Cands.push_back( vector<RecoChargedCandidateRef>() );
  }  
  
  Handle<TriggerEventWithRefs> rawTriggerEvent;
  iEvent.getByLabel( "hltTriggerSummaryRAW", rawTriggerEvent );
  if( rawTriggerEvent.isValid() ){
    for(size_t path=0; path<filterNames.size(); path++){
      size_t indexL1 = rawTriggerEvent->filterIndex(InputTag( filterNames[path][1], "", processName ));
      if ( indexL1 < rawTriggerEvent->size() ){
          rawTriggerEvent->getObjects( indexL1, TriggerL1Mu, l1Cands[path] );
      }  
      pathME[path]["l1Muon_size"]->Fill(l1Cands[path].size());
      size_t indexL2 = rawTriggerEvent->filterIndex(InputTag( filterNames[path][2], "", processName ));
      if ( indexL2 < rawTriggerEvent->size() ){
          rawTriggerEvent->getObjects( indexL2, TriggerMuon, l2Cands[path] );
      }
      pathME[path]["l2vMuon_size"]->Fill(l2Cands[path].size());
      size_t indexL3 = rawTriggerEvent->filterIndex(InputTag( filterNames[path][3], "", processName ));
      if ( indexL3 < rawTriggerEvent->size() ){
          rawTriggerEvent->getObjects( indexL3, TriggerMuon, l3Cands[path] );
      }
      pathME[path]["l3Muon_size"]->Fill(l3Cands[path].size());
    }
  }else{
    LogDebug("HLTriggerOfflineHeavyFlavor")<<"Could not access rawTriggerEvent"<<endl;
  }
    
//fill histos
  for(size_t i=0; i<genMuons.size(); i++){
    offlineME["genMuon_genPtEta"]->Fill(genMuons[i].pt(), genMuons[i].eta());
    if(glob_gen[i] != -1){
      offlineME["genGlobMuon_genPtEta"]->Fill(genMuons[i].pt(), genMuons[i].eta());
      offlineME["genGlobMuon_recoPtEta"]->Fill(globMuons[glob_gen[i]].pt(), globMuons[glob_gen[i]].eta());
      for(size_t path=0; path<filterNames.size(); path++){  
        if(l1_glob[glob_gen[i]] != -1 && find(l1Cands[path].begin(), l1Cands[path].end(), L1MuonParticleRef(l1MuonsHandle,l1_glob[glob_gen[i]])) != l1Cands[path].end() ){
          pathME[path]["genGlobL1Muon_recoPtEta"]->Fill(globMuons[glob_gen[i]].pt(), globMuons[glob_gen[i]].eta());
          if(l2v_glob[glob_gen[i]] != -1 && containsIndex(l2Cands[path], l2v_glob[glob_gen[i]]) ){
            pathME[path]["genGlobL1L2L2vMuon_recoPtEta"]->Fill(globMuons[glob_gen[i]].pt(), globMuons[glob_gen[i]].eta());
            if(l3_glob[glob_gen[i]] != -1 && containsIndex(l3Cands[path], l3_glob[glob_gen[i]]) ){
              pathME[path]["genGlobL1L2L2vL3Muon_recoPtEta"]->Fill(globMuons[glob_gen[i]].pt(), globMuons[glob_gen[i]].eta());
            }
          }
        }
      }
    }
  }
  
//trigger path efficiencies wrt global dimuon   
  Handle<TriggerResults> triggerResultsHandle;
  iEvent.getByLabel(triggerResultsTag, triggerResultsHandle);
  if( !triggerResultsHandle.isValid() ){
    LogDebug("HLTriggerOfflineHeavyFlavor") << "Could not access TriggerResults"<<endl;
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
    offlineME["genDimuon_genPt"]->Fill( genDimuonPt );
    massME["genDimuon_mass"]->Fill( (genMuons[0].p4()+genMuons[secondMuon].p4()).mass() );
//two global
    if(glob_gen[0]!=-1 && glob_gen[secondMuon]!=-1){
      offlineME["genGlobDimuon_genPt"]->Fill( genDimuonPt );
      double globDimuonPt = (globMuons[glob_gen[0]].p4()+globMuons[glob_gen[secondMuon]].p4()).pt();
      offlineME["genGlobDimuon_recoPt"]->Fill( globDimuonPt );
      massME["genGlobDimuon_mass"]->Fill( (globMuons[glob_gen[0]].p4()+globMuons[glob_gen[secondMuon]].p4()).mass() );
      for(size_t path=0; path<filterNames.size(); path++){  
        if(triggerResultsHandle.isValid() && triggerResultsHandle->accept(pathIndices[path])){
          pathME[path]["genGlobDimuonPath_recoPt"]->Fill(globDimuonPt);
        }
//two l1      
        if( l1_glob[glob_gen[0]]!=-1 
          && find(l1Cands[path].begin(), l1Cands[path].end(), L1MuonParticleRef(l1MuonsHandle,l1_glob[glob_gen[0]])) != l1Cands[path].end()
          && l1_glob[glob_gen[secondMuon]]!=-1
          && find(l1Cands[path].begin(), l1Cands[path].end(), L1MuonParticleRef(l1MuonsHandle,l1_glob[glob_gen[secondMuon]])) != l1Cands[path].end()
          ){
          pathME[path]["genGlobL1Dimuon_recoPt"]->Fill( globDimuonPt );
          massME["genGlobL1Dimuon_mass"]->Fill( (l1Muons[l1_glob[glob_gen[0]]].p4()+l1Muons[l1_glob[glob_gen[secondMuon]]].p4()).mass() );
//two l2v       
          if( l2v_glob[glob_gen[0]] != -1 && containsIndex(l2Cands[path], l2v_glob[glob_gen[0]])
            && l2v_glob[glob_gen[secondMuon]] != -1 && containsIndex(l2Cands[path], l2v_glob[glob_gen[secondMuon]])
            ){
            pathME[path]["genGlobL1L2L2vDimuon_recoPt"]->Fill( globDimuonPt );
            massME["genGlobL1L2L2vDimuon_mass"]->Fill( (l2vMuons[l2v_glob[glob_gen[0]]].p4()+l2vMuons[l2v_glob[glob_gen[secondMuon]]].p4()).mass() );
//two l3         
            if(l3_glob[glob_gen[0]] != -1  && containsIndex(l3Cands[path], l3_glob[glob_gen[0]])
              && l3_glob[glob_gen[secondMuon]] != -1 && containsIndex(l3Cands[path], l3_glob[glob_gen[secondMuon]])
              ){
              pathME[path]["genGlobL1L2L2vL3Dimuon_recoPt"]->Fill( globDimuonPt );
              massME["genGlobL1L2L2vL3Dimuon_mass"]->Fill( (l3Muons[l3_glob[glob_gen[0]]].p4()+l3Muons[l3_glob[glob_gen[secondMuon]]].p4()).mass() );
            }
          }
        }
      }
    }
//fill dR histograms when both muon pT>0
    if(genMuons[0].pt()>0. && genMuons[secondMuon].pt()>0.){
      double gendR = deltaR<LeafCandidate,LeafCandidate>(genMuons[0],genMuons[secondMuon]);
      dRME["genDimuon_gendR"]->Fill( gendR );
      if(glob_gen[0]!=-1 && glob_gen[secondMuon]!=-1){
        dRME["genGlobDimuon_gendR"]->Fill( gendR );
      }
    }
    if(glob_gen[0]!=-1 && globMuons[glob_gen[0]].pt()>0. && glob_gen[secondMuon]!=-1 && globMuons[glob_gen[secondMuon]].pt()>0.){
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

bool HeavyFlavorValidation::containsIndex(vector<RecoChargedCandidateRef> &v, size_t i){
  bool result = false;
  for(size_t i=0; i<v.size(); i++){
    if(v[i].key() == i) result = true;
  }
  return result;
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
