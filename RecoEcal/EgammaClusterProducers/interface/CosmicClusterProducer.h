#ifndef RecoEcal_EgammaClusterProducers_CosmicClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_CosmicClusterProducer_h_

#include <memory>
#include <time.h>
#include <vector> //TEMP JHAUPT 4-27

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "RecoEcal/EgammaClusterAlgos/interface/CosmicClusterAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEcal/EgammaCoreTools/interface/ClusterShapeAlgo.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"

//


class CosmicClusterProducer : public edm::EDProducer 
{
  public:

      CosmicClusterProducer(const edm::ParameterSet& ps);

      ~CosmicClusterProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:

      int nMaxPrintout_; // max # of printouts
      int nEvt_;         // internal counter of events

      CosmicClusterAlgo::VerbosityLevel verbosity;

      std::string barrelHitProducer_;
      std::string endcapHitProducer_;
      std::string barrelHitCollection_;
      std::string endcapHitCollection_;

	  std::string barrelUHitProducer_;
      std::string endcapUHitProducer_;
      std::string barrelUHitCollection_;
      std::string endcapUHitCollection_;
	  
      std::string barrelClusterCollection_;
      std::string endcapClusterCollection_;

      std::string clustershapecollectionEB_;
      std::string clustershapecollectionEE_;

      //BasicClusterShape AssociationMap
      std::string barrelClusterShapeAssociation_;
      std::string endcapClusterShapeAssociation_; 

      PositionCalc posCalculator_; // position calculation algorithm
      ClusterShapeAlgo shapeAlgo_; // cluster shape algorithm
      CosmicClusterAlgo * island_p;

      bool counterExceeded() const { return ((nEvt_ > nMaxPrintout_) || (nMaxPrintout_ < 0)); }

      const EcalRecHitCollection * getCollection(edm::Event& evt,
                                                 const std::string& hitProducer_,
                                                 const std::string& hitCollection_);

	  const EcalUncalibratedRecHitCollection * getUCollection(edm::Event& evt,
                                                 const std::string& hitProducer_,
                                                 const std::string& hitCollection_);
												 
      void clusterizeECALPart(edm::Event &evt, const edm::EventSetup &es,
                              const std::string& hitProducer,
                              const std::string& hitCollection,
							  const std::string& uhitProducer,
                              const std::string& uhitCollection,
                              const std::string& clusterCollection,
			      const std::string& clusterShapeAssociation,
                              const CosmicClusterAlgo::EcalPart& ecalPart);

      void outputValidationInfo(reco::CaloClusterPtrVector &clusterPtrVector);
	  
	 
};


#endif
