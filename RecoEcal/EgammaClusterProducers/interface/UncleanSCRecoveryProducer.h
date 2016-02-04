#ifndef RecoEcal_EgammaClusterProducers_UncleanSCRecoveryProducer_h_
#define RecoEcal_EgammaClusterProducers_UncleanSCRecoveryProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Utilities/interface/InputTag.h"

// $Id: UncleanSCRecoveryProducer.h,v 1.5 2011/07/18 18:05:40 argiro Exp $

class UncleanSCRecoveryProducer : public edm::EDProducer 
{
  
  public:

      UncleanSCRecoveryProducer(const edm::ParameterSet& ps);

      ~UncleanSCRecoveryProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);
      
  private:
      // the clean collection      
      edm::InputTag  cleanBcCollection_; 
      edm::InputTag  cleanScCollection_; 
      // the uncleaned collection
      edm::InputTag uncleanBcCollection_;
      edm::InputTag uncleanScCollection_;
      // the names of the products to be produced:
      std::string  bcCollection_;     
      std::string  scCollection_;     


};


#endif


