#ifndef RecoEcal_EgammaClusterProducers_CleanAndMergeProducer_h_
#define RecoEcal_EgammaClusterProducers_CleanAndMergeProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEcal/EgammaCoreTools/interface/ClusterShapeAlgo.h"


class CleanAndMergeProducer : public edm::EDProducer 
{
  
  public:

      CleanAndMergeProducer(const edm::ParameterSet& ps);

      ~CleanAndMergeProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);
      
  private:
      

      edm::InputTag cleanScInputTag_;
      edm::InputTag uncleanScInputTag_;
     
      // the names of the products to be produced:
      std::string  bcCollection_;     
      std::string  scCollection_;     
      std::string  refScCollection_;  
      // other collections
      std::string hitproducer_;
      std::string hitcollection_;


};


#endif


