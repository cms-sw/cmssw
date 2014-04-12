#ifndef RecoEcal_EgammaClusterProducers_Multi5x5ClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_Multi5x5ClusterProducer_h_

#include <memory>
#include <time.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEcal/EgammaClusterAlgos/interface/Multi5x5ClusterAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

#include "DataFormats/EgammaReco/interface/BasicClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"

//


class Multi5x5ClusterProducer : public edm::stream::EDProducer<> 
{
  public:

      Multi5x5ClusterProducer(const edm::ParameterSet& ps);

      ~Multi5x5ClusterProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:

      int nMaxPrintout_; // max # of printouts
      int nEvt_;         // internal counter of events

      // cluster which regions
      bool doBarrel_;
      bool doEndcap_;

      edm::EDGetTokenT<EcalRecHitCollection> barrelHitToken_;
	  edm::EDGetTokenT<EcalRecHitCollection> endcapHitToken_;

      std::string barrelClusterCollection_;
      std::string endcapClusterCollection_;

      PositionCalc posCalculator_; // position calculation algorithm
      Multi5x5ClusterAlgo * island_p;

      bool counterExceeded() const { return ((nEvt_ > nMaxPrintout_) || (nMaxPrintout_ < 0)); }

      const EcalRecHitCollection * getCollection(edm::Event& evt,
                                                 const edm::EDGetTokenT<EcalRecHitCollection>& token );


      void clusterizeECALPart(edm::Event &evt, const edm::EventSetup &es,
                              const edm::EDGetTokenT<EcalRecHitCollection>& token, 
                              const std::string& clusterCollection,
                              const reco::CaloID::Detectors detector);

      void outputValidationInfo(reco::CaloClusterPtrVector &clusterPtrVector);
};


#endif
