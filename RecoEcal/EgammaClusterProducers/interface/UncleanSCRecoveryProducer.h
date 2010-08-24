#ifndef RecoEcal_EgammaClusterProducers_UncleanSCRecoveryProducer_h_
#define RecoEcal_EgammaClusterProducers_UncleanSCRecoveryProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEcal/EgammaClusterAlgos/interface/HybridClusterAlgo.h"


class UncleanSCRecoveryProducer : public edm::EDProducer 
{
  
  public:

      UncleanSCRecoveryProducer(const edm::ParameterSet& ps);

      ~UncleanSCRecoveryProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);
      
  private:
      
      std::string  cleanBcCollection_; 
      std::string  cleanBcProducer_; 
      std::string  cleanScCollection_; 
      std::string  cleanScProducer_; 
      // the uncleaned collection
      std::string  uncleanBcCollection_;
      std::string  uncleanBcProducer_;
      std::string  uncleanScCollection_;
      std::string  uncleanScProducer_;
      // the names of the products to be produced:
      std::string  bcCollection_;     
      std::string  scCollection_;     

      HybridClusterAlgo::DebugLevel debugL;

};


#endif


