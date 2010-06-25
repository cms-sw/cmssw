#ifndef RecoEcal_EgammaClusterProducers_CleanAndMergeProducer_h_
#define RecoEcal_EgammaClusterProducers_CleanAndMergeProducer_h_

#include <memory>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "RecoEcal/EgammaClusterAlgos/interface/HybridClusterAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/ClusterShapeAlgo.h"


class CleanAndMergeProducer : public edm::EDProducer 
{
  
  public:

      CleanAndMergeProducer(const edm::ParameterSet& ps);

      ~CleanAndMergeProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);
      
  private:
      
      std::string  cleanBcCollection_; 
      std::string  cleanBcProducer_; 
      std::string  cleanScCollection_; 
      std::string  cleanScProducer_; 
      //std::string  cleanClShapeAssoc_; 
      // the uncleaned collection
      std::string  uncleanBcCollection_;
      std::string  uncleanBcProducer_;
      std::string  uncleanScCollection_;
      std::string  uncleanScProducer_;
      //std::string  uncleanClShapeAssoc_; 
      // the names of the products to be produced:
      std::string  bcCollection_;     
      std::string  scCollection_;     
      std::string  cShapeCollection_; 
      std::string  clShapeAssoc_;     
      std::string  refScCollection_;  
      // other collections
      std::string hitproducer_;
      std::string hitcollection_;


      HybridClusterAlgo::DebugLevel debugL;

      ClusterShapeAlgo shapeAlgo_; // cluster shape algorithm


};


#endif


