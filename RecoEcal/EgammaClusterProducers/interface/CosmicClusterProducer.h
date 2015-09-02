#ifndef RecoEcal_EgammaClusterProducers_CosmicClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_CosmicClusterProducer_h_

#include <memory>
#include <time.h>
#include <vector> //TEMP JHAUPT 4-27

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
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


class CosmicClusterProducer : public edm::stream::EDProducer<>
{
  public:

      CosmicClusterProducer(const edm::ParameterSet& ps);

      ~CosmicClusterProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:

      int nMaxPrintout_; // max # of printouts
      int nEvt_;         // internal counter of events

      CosmicClusterAlgo::VerbosityLevel verbosity;

      edm::EDGetTokenT<EcalRecHitCollection> ebHitsToken_;
      edm::EDGetTokenT<EcalRecHitCollection> eeHitsToken_;

      edm::EDGetTokenT<EcalUncalibratedRecHitCollection> ebUHitsToken_;
      edm::EDGetTokenT<EcalUncalibratedRecHitCollection> eeUHitsToken_;
	  
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

      											 
      void clusterizeECALPart(edm::Event &evt, const edm::EventSetup &es,
			      const edm::EDGetTokenT<EcalRecHitCollection>& hitsToken,
			      const edm::EDGetTokenT<EcalUncalibratedRecHitCollection>& uhitsToken,       
                              const std::string& clusterCollection,
			      const std::string& clusterShapeAssociation,
                              const CosmicClusterAlgo::EcalPart& ecalPart);

      void outputValidationInfo(reco::CaloClusterPtrVector &clusterPtrVector);
	  
	 
};


#endif
