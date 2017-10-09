#ifndef RecoEcal_EgammaClusterProducers_EcalDigiSelector_h_
#define RecoEcal_EgammaClusterProducers_EcalDigiSelector_h_

#include <memory>
#include <vector>
#include <map>
#include <string>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h" 

//


class EcalDigiSelector : public edm::stream::EDProducer<> 
{
  
 public:
  
  EcalDigiSelector(const edm::ParameterSet& ps);
   
  void produce(edm::Event&, const edm::EventSetup&);
  

 private:
 
  std::string selectedEcalEBDigiCollection_;
  std::string selectedEcalEEDigiCollection_;

  edm::EDGetTokenT<reco::SuperClusterCollection> barrelSuperClusterProducer_;
  edm::EDGetTokenT<reco::SuperClusterCollection> endcapSuperClusterProducer_;

  // input configuration
  edm::EDGetTokenT<EcalRecHitCollection> EcalEBRecHitToken_;
  edm::EDGetTokenT<EcalRecHitCollection> EcalEERecHitToken_;
  edm::EDGetTokenT<EBDigiCollection> EcalEBDigiToken_;
  edm::EDGetTokenT<EEDigiCollection> EcalEEDigiToken_;


 

  double cluster_pt_thresh_;
  double single_cluster_thresh_;
  int nclus_sel_;

};


#endif
