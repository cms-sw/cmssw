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

#include "DQMOffline/Trigger/interface/EgHLTOffEleSel.h"

#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Framework/interface/Event.h"

class EgHLTOffHelper {

 private:
  EgHLTOffEleSel tagCuts_; //cuts applied to tags
  EgHLTOffEleSel probeCuts_; //cuts applied to probes
  EgHLTOffEleSel cuts_; //normal selection cuts

  edm::InputTag barrelShapeAssocProd_;
  edm::InputTag endcapShapeAssocProd_;
  
  edm::Handle<reco::BasicClusterShapeAssociationCollection> clusterShapeHandleBarrel_;
  edm::Handle<reco::BasicClusterShapeAssociationCollection> clusterShapeHandleEndcap_;


 public:
  EgHLTOffHelper():tagCuts_(),probeCuts_(),cuts_(){}
  ~EgHLTOffHelper(){}

  void setup(const edm::ParameterSet& conf);

  void getHandles(const edm::Event& event);
  

  void fillEgHLTOffEleVec(edm::Handle<reco::GsfElectronCollection> gsfElectrons,std::vector<EgHLTOffEle>& egHLTOffEles);
  
  //ripped of from the electronIDAlgo (there must be a better way, I *cannot* believe that there isnt a better way)
  const reco::ClusterShape* getClusterShape(const reco::GsfElectron* electron);
  
  
			  
			  


};

#endif
