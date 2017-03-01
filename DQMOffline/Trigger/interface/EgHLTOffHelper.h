#ifndef DQMOFFLINE_TRIGGER_EGHLTOFFHELPER
#define DQMOFFLINE_TRIGGER_EGHLTOFFHELPER

//class: EgHLTOffHelper (Egamma HLT offline helper)
//
//author: Sam Harper (July 2008)
//
//
//aim: to hide temporary place holder code away from the rest of the system
//
//implimentation: currently no isolation producers or electron selection cut meets my needs
//                while I would like to use a central tool, for now I'm cludging my own as
//                placeholders


#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/CaloTowers/interface/CaloTower.h"
#include "DataFormats/CaloTowers/interface/CaloTowerFwd.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"

#include "DQMOffline/Trigger/interface/EgHLTOffEvt.h"
#include "DQMOffline/Trigger/interface/EgHLTOffEle.h"
#include "DQMOffline/Trigger/interface/EgHLTOffPho.h"
#include "DQMOffline/Trigger/interface/EgHLTOffEgSel.h"
#include "DQMOffline/Trigger/interface/EgHLTTrigCodes.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "MagneticField/Engine/interface/MagneticField.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

class EgammaHLTTrackIsolation;
class HLTConfigProvider;
class EcalSeverityLevelAlgo;

namespace egHLT {

  class OffHelper {

  private:

    OffEgSel eleLooseCuts_; //loose selection cuts (loose has no relation to other 'loose' cuts)
    OffEgSel eleCuts_; //normal selection cuts
    OffEgSel phoLooseCuts_; //loose selection cuts (loose has no relation to other 'loose' cuts)
    OffEgSel phoCuts_; //normal selection cuts
    
    std::vector<std::pair<TrigCodes::TrigBitSet,OffEgSel> > trigCuts_;//non sorted vector (for now)
    
    
    edm::EDGetTokenT <EcalRecHitCollection>  ecalRecHitsEBToken;
    edm::EDGetTokenT <EcalRecHitCollection>  ecalRecHitsEEToken;
    edm::EDGetTokenT <reco::CaloJetCollection>  caloJetsToken;
    edm::EDGetTokenT <reco::TrackCollection>  isolTrkToken;
    edm::EDGetTokenT <HBHERecHitCollection>  hbheHitsToken;
    edm::EDGetTokenT <HFRecHitCollection>  hfHitsToken;
    edm::EDGetTokenT <trigger::TriggerEvent> triggerSummaryToken;
    edm::EDGetTokenT <reco::GsfElectronCollection>  electronsToken;
    edm::EDGetTokenT <reco::PhotonCollection>  photonsToken;
    edm::EDGetTokenT <reco::BeamSpot>  beamSpotToken;
    edm::EDGetTokenT <CaloTowerCollection>  caloTowersToken;
    edm::EDGetTokenT <edm::TriggerResults>  trigResultsToken;
    edm::EDGetTokenT <reco::VertexCollection>  vertexToken;

    edm::ESHandle<CaloGeometry> caloGeom_;
    edm::ESHandle<CaloTopology> caloTopology_;
    edm::ESHandle<MagneticField> magField_;
    edm::ESHandle<EcalSeverityLevelAlgo> ecalSeverityLevel_;

    edm::Handle<EcalRecHitCollection> ebRecHits_;
    edm::Handle<EcalRecHitCollection> eeRecHits_; 
    edm::Handle<HFRecHitCollection> hfHits_;
    edm::Handle<HBHERecHitCollection> hbheHits_;
    edm::Handle<reco::TrackCollection> isolTrks_;

    edm::Handle<trigger::TriggerEvent> trigEvt_;
    edm::Handle<reco::PhotonCollection> recoPhos_;
    edm::Handle<reco::GsfElectronCollection> recoEles_;
    edm::Handle<std::vector<reco::CaloJet> > recoJets_;
    
    edm::Handle<reco::BeamSpot> beamSpot_;
    edm::Handle<CaloTowerCollection> caloTowers_;
   
    edm::Handle<edm::TriggerResults> trigResults_;

    edm::Handle<reco::VertexCollection> recoVertices_;
    
 

    std::string hltTag_;
    std::vector<std::string> hltFiltersUsed_;
    std::vector<std::pair<std::string,int> > hltFiltersUsedWithNrCandsCut_; //stores the filter name + number of candidates required to pass that filter for it to accept
    std::vector<std::pair<std::string,std::string> > l1PreAndSeedFilters_; //filter names of a l1 prescaler and the corresponding l1 seed filter
    std::vector<std::string> l1PreScaledPaths_;//l1 pre-scaled path names
    std::vector<std::string> l1PreScaledFilters_;//l1 pre scale filters

    //allow us to recompute e/gamma HLT isolations (note we also have em and hcal but they have to be declared for every event)
    //which is awkward and I havent thought of a good way around it yet
    EgammaHLTTrackIsolation* hltEleTrkIsolAlgo_;
    EgammaHLTTrackIsolation* hltPhoTrkIsolAlgo_;

    //our hlt isolation parameters...
    //ecal
    double hltEMIsolOuterCone_;
    double hltEMIsolInnerConeEB_;
    double hltEMIsolEtaSliceEB_;
    double hltEMIsolEtMinEB_;
    double hltEMIsolEMinEB_;
    double hltEMIsolInnerConeEE_;
    double hltEMIsolEtaSliceEE_;
    double hltEMIsolEtMinEE_;
    double hltEMIsolEMinEE_;
    //tracker
    double hltPhoTrkIsolPtMin_;
    double hltPhoTrkIsolOuterCone_;
    double hltPhoTrkIsolInnerCone_;
    double hltPhoTrkIsolZSpan_;
    double hltPhoTrkIsolRSpan_;
    bool hltPhoTrkIsolCountTrks_;
    double hltEleTrkIsolPtMin_;
    double hltEleTrkIsolOuterCone_;
    double hltEleTrkIsolInnerCone_;
    double hltEleTrkIsolZSpan_;
    double hltEleTrkIsolRSpan_;
    //hcal
    double hltHadIsolOuterCone_;
    double hltHadIsolInnerCone_;
    double hltHadIsolEtMin_;
    int hltHadIsolDepth_;
    //flags to disable calculations if same as reco (saves time)
    bool calHLTHcalIsol_;
    bool calHLTEmIsol_;
    bool calHLTEleTrkIsol_;
    bool calHLTPhoTrkIsol_;
    
    
    std::vector<edm::ParameterSet> trigCutParams_; //probably the least bad option

  private: //disabling copy / assignment
    OffHelper & operator=(const OffHelper&) = delete;
    OffHelper(const OffHelper&) = delete;
    
  public:
    OffHelper(): eleLooseCuts_(),eleCuts_(),phoLooseCuts_(),phoCuts_(),hltEleTrkIsolAlgo_(NULL),hltPhoTrkIsolAlgo_(NULL){}
    ~OffHelper();
    
    void setup(const edm::ParameterSet& conf, edm::ConsumesCollector && iC);
    void setupTriggers(const HLTConfigProvider& config,const std::vector<std::string>& hltFiltersUsed, const TrigCodes& trigCodes);

    //int is the error code, 0 = no error
    //it should never throw, print to screen or crash, this is the only error reporting it does
    int makeOffEvt(const edm::Event& edmEvent,const edm::EventSetup& setup,egHLT::OffEvt& offEvent,const TrigCodes& trigCodes);
    
    int getHandles(const edm::Event& event,const edm::EventSetup& setup);
    int fillOffEleVec(std::vector<OffEle>& offEles);
    int fillOffPhoVec(std::vector<OffPho>& offPhos);
    int setTrigInfo(const edm::Event & edmEvent, egHLT::OffEvt& offEvent, const TrigCodes& trigCodes);

    void fillIsolData(const reco::GsfElectron& ele,OffEle::IsolData& isolData);
    void fillClusShapeData(const reco::GsfElectron& ele,OffEle::ClusShapeData& clusShapeData);
    void fillHLTData(const reco::GsfElectron& ele,OffEle::HLTData& hltData);    

    void fillIsolData(const reco::Photon& pho,OffPho::IsolData& isolData);
    void fillClusShapeData(const reco::Photon& pho,OffPho::ClusShapeData& clusShapeData);
    void fillHLTDataPho(const reco::Photon& pho,OffPho::HLTData& hltData);    

    //tempory debugging functions
    const trigger::TriggerEvent* trigEvt()const{return trigEvt_.product();}
    const std::vector<std::pair<TrigCodes::TrigBitSet,OffEgSel> >& trigCuts()const{return trigCuts_;}
    
    
    template<class T> static bool getHandle(const edm::Event& event,const edm::EDGetTokenT<T>& token,edm::Handle<T>& handle);
    
  };
  

  template<class T> bool OffHelper::getHandle(const edm::Event& event,const edm::EDGetTokenT<T>& token, edm::Handle<T>& handle)
  {
  
    bool success=event.getByToken(token,handle);
    return success &&  handle.product();
    

  }
}

#endif
