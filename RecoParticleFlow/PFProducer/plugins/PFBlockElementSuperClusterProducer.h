#ifndef RecoParticleFlow_PFProducer_PFBlockElementSuperClusterProducer_H
#define RecoParticleFlow_PFProducer_PFBlockElementSuperClusterProducer_H
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include <iostream>
#include <string>
#include <map>



class PFBlockElementSuperClusterProducer : public edm::EDProducer
{
 public:
  explicit PFBlockElementSuperClusterProducer(const edm::ParameterSet&);
  ~PFBlockElementSuperClusterProducer();
  
  virtual void produce(edm::Event &, const edm::EventSetup&);
  virtual void beginRun(edm::Run & run,const edm::EventSetup & c);

 private:
  std::vector<edm::InputTag> inputTagSuperClusters_;
  std::string outputName_; 

};
#endif
