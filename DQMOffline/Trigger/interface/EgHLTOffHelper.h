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

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/ClusterShape.h"
#include "DataFormats/EgammaReco/interface/ClusterShapeFwd.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"
#include "DataFormats/EgammaReco/interface/BasicClusterShapeAssociation.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "DQMOffline/Trigger/interface/EgHLTOffEleSel.h"

#include "FWCore/ParameterSet/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"

class CaloGeometry;
class CaloTopology;


class EgHLTOffHelper {

 private:
  EgHLTOffEleSel tagCuts_; //cuts applied to tags
  EgHLTOffEleSel probeCuts_; //cuts applied to probes
  EgHLTOffEleSel cuts_; //normal selection cuts

  //needed for pre 2_1_X releases
  //edm::InputTag barrelShapeAssocProd_;
  //edm::InputTag endcapShapeAssocProd_;
  //edm::Handle<reco::BasicClusterShapeAssociationCollection> clusterShapeHandleBarrel_;
  // edm::Handle<reco::BasicClusterShapeAssociationCollection> clusterShapeHandleEndcap_;

   //does anybody else think its ridicious that we need handles to the CaloGeometry and Topology as well as all the read out ecal barrel / endcap hits to calculated a standard id variable which to be perfectly honest should be accessible from the electron directly.
  //as you may have guessed the following six members are to enable us to calculate sigmaEtaEta, with the first two being the tags needed
  edm::InputTag ecalRecHitsEBTag_;
  edm::InputTag ecalRecHitsEETag_;
  edm::InputTag caloJetsTag_;
  const CaloGeometry* caloGeom_;
  const CaloTopology* caloTopology_;
  const EcalRecHitCollection* ebRecHits_;
  const EcalRecHitCollection* eeRecHits_;
  const std::vector<reco::CaloJet>* jets_;


 public:
  EgHLTOffHelper():tagCuts_(),probeCuts_(),cuts_(){}
  ~EgHLTOffHelper(){}

  void setup(const edm::ParameterSet& conf);

  void getHandles(const edm::Event& event,const edm::EventSetup& setup);
  
  const std::vector<reco::CaloJet>* jets()const{return jets_;}

  void fillEgHLTOffEleVec(edm::Handle<reco::GsfElectronCollection> gsfElectrons,std::vector<EgHLTOffEle>& egHLTOffEles);
  
  //ripped of from the electronIDAlgo (there must be a better way, I *cannot* believe that there isnt a better way)
  //incidently they came up with a new way in 2_1_X, making this redundant. The new way is acutally worse... 
  const reco::ClusterShape* getClusterShape(const reco::GsfElectron* electron);
  
  
 
			  


};

#endif
