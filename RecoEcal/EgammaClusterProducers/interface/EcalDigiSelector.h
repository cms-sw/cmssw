#ifndef RecoEcal_EgammaClusterProducers_EcalDigiSelector_h_
#define RecoEcal_EgammaClusterProducers_EcalDigiSelector_h_

#include <memory>
#include <vector>
#include <map>
#include <string>
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"


//


class EcalDigiSelector : public edm::EDProducer 
{
  
 public:
  
  EcalDigiSelector(const edm::ParameterSet& ps);
  
  virtual ~EcalDigiSelector();
  
  virtual void produce(edm::Event&, const edm::EventSetup&);
  
 private:
 
  std::string selectedEcalEBDigiCollection_;
  std::string selectedEcalEEDigiCollection_;

  edm::InputTag barrelSuperClusterProducer_;
  edm::InputTag endcapSuperClusterProducer_;

  // input configuration
  edm::InputTag EcalEBDigiTag_;
  edm::InputTag EcalEEDigiTag_;

  edm::InputTag EcalEBRecHitTag_;
  edm::InputTag EcalEERecHitTag_;
  
  double cluster_pt_thresh_;
  double single_cluster_thresh_;
  int nclus_sel_;


};


#endif
