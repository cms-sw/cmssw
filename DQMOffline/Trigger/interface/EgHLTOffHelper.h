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
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"
//#include "DataFormats/EgammaCandidates/interface/PhotonID.h"
//#include "DataFormats/EgammaCandidates/interface/PhotonIDFwd.h"
//#include "DataFormats/EgammaCandidates/interface/PhotonIDAssociation.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

#include "DQMOffline/Trigger/interface/EgHLTOffEvt.h"
#include "DQMOffline/Trigger/interface/EgHLTOffEle.h"
#include "DQMOffline/Trigger/interface/EgHLTOffPho.h"
#include "DQMOffline/Trigger/interface/EgHLTOffEgSel.h"
#include "DQMOffline/Trigger/interface/EgHLTTrigCodes.h"

#include "Geometry/Records/interface/CaloGeometryRecord.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/CaloTopology.h"

#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"


class EgammaHLTHcalIsolation;
class EgammaHLTTrackIsolation;

namespace egHLT {

  class OffHelper {

  private:
    OffEgSel eleLooseCuts_; //loose selection cuts (loose has no relation to other 'loose' cuts)
    OffEgSel eleCuts_; //normal selection cuts
    OffEgSel phoLooseCuts_; //loose selection cuts (loose has no relation to other 'loose' cuts)
    OffEgSel phoCuts_; //normal selection cuts
    
    std::vector<std::pair<TrigCodes::TrigBitSet,OffEgSel> > trigCuts_;//non sorted vector (for now)
    
    //does anybody else think its ridicious that we need handles to the CaloGeometry and Topology as well as all the read out ecal barrel / endcap hits to calculated a standard id variable which to be perfectly honest should be accessible from the electron directly.
    //as you may have guessed the following six members are to enable us to calculate sigmaEtaEta, with the first two being the tags needed
    edm::InputTag ecalRecHitsEBTag_;
    edm::InputTag ecalRecHitsEETag_;
    edm::InputTag caloJetsTag_;
    edm::InputTag isolTrkTag_;
    edm::InputTag hbheHitsTag_;
    edm::InputTag hfHitsTag_;
    edm::InputTag triggerSummaryLabel_;
    edm::InputTag electronsTag_;
    edm::InputTag photonsTag_;
    edm::InputTag eleEcalIsolTag_;
    edm::InputTag eleHcalDepth1IsolTag_;
    edm::InputTag eleHcalDepth2IsolTag_;
    edm::InputTag eleTrkIsolTag_;
    //edm::InputTag phoIDTag_;

    edm::ESHandle<CaloGeometry> caloGeom_;
    edm::ESHandle<CaloTopology> caloTopology_;
    
    edm::Handle<EcalRecHitCollection> ebRecHits_;
    edm::Handle<EcalRecHitCollection> eeRecHits_; 
    edm::Handle<HFRecHitCollection> hfHits_;
    edm::Handle<HBHERecHitCollection> hbheHits_;
    edm::Handle<reco::TrackCollection> isolTrks_;

    edm::Handle<trigger::TriggerEvent> trigEvt_;
    edm::Handle<reco::PhotonCollection> recoPhos_;
    edm::Handle<reco::GsfElectronCollection> recoEles_;
    edm::Handle<std::vector<reco::CaloJet> > recoJets_;
    // edm::Handle<reco::PhotonIDAssociationCollection> photonIDMap_;

    edm::Handle<edm::ValueMap<double> > eleEcalIsol_;
    edm::Handle<edm::ValueMap<double> > eleHcalDepth1Isol_;
    edm::Handle<edm::ValueMap<double> > eleHcalDepth2Isol_;
    edm::Handle<edm::ValueMap<double> > eleTrkIsol_;

    std::string hltTag_;
    std::vector<std::string> hltFiltersUsed_;

    //allow us to recompute e/gamma HLT isolations
    //cant do ECAL isolation at the moment as we would need island clusters 
    EgammaHLTHcalIsolation* hltHcalIsolAlgo_;
    EgammaHLTTrackIsolation* hltEleTrkIsolAlgo_;
    EgammaHLTTrackIsolation* hltPhoTrkIsolAlgo_;

  private: //disabling copy / assignment
    OffHelper& operator=(const OffHelper& rhs){return *this;}
    OffHelper(const OffHelper& rhs){}
    
  public:
    OffHelper():eleLooseCuts_(),eleCuts_(),phoLooseCuts_(),phoCuts_(),hltHcalIsolAlgo_(NULL),hltEleTrkIsolAlgo_(NULL),hltPhoTrkIsolAlgo_(NULL){}
    ~OffHelper();
    
    void setup(const edm::ParameterSet& conf,const std::vector<std::string>& hltFiltersUsed);

    //int is the error code, 0 = no error
    //it should never throw, print to screen or crash, this is the only error reporting it does
    int makeOffEvt(const edm::Event& edmEvent,const edm::EventSetup& setup,egHLT::OffEvt& offEvent);
    
    
    int getHandles(const edm::Event& event,const edm::EventSetup& setup);
    int fillOffEleVec(std::vector<OffEle>& offEles);
    int fillOffPhoVec(std::vector<OffPho>& offPhos);
    int setTrigInfo(egHLT::OffEvt& offEvent);

    //
    template<class T> static bool getHandle(const edm::Event& event,const edm::InputTag& tag,edm::Handle<T>& handle);
    
  };
  

  template<class T> bool OffHelper::getHandle(const edm::Event& event,const edm::InputTag& tag,edm::Handle<T>& handle)
  {
  
    bool success=event.getByLabel(tag,handle);
    return success &&  handle.product();
    

  }
}

#endif
