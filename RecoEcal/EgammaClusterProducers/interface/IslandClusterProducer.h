#ifndef RecoEcal_EgammaClusterProducers_IslandClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_IslandClusterProducer_h_

#include <memory>
#include <time.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEcal/EgammaClusterAlgos/interface/IslandClusterAlgo.h"

#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

//


class IslandClusterProducer : public edm::EDProducer 
{
  
  public:

      IslandClusterProducer(const edm::ParameterSet& ps);

      ~IslandClusterProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:

      int nMaxPrintout_; // max # of printouts
      int nEvt_;         // internal counter of events
 
      IslandClusterAlgo::VerbosityLevel verbosity;

      std::string barrelHitProducer_;
      std::string endcapHitProducer_;
      std::string barrelHitCollection_;
      std::string endcapHitCollection_;

      std::string barrelClusterCollection_;
      std::string endcapClusterCollection_;

      // Position correction parameters
      std::string clustershapecollectionEB_;
      std::string clustershapecollectionEE_;

      bool clustershape_logweighted;
      float clustershape_x0;
      float clustershape_t0;
      float clustershape_w0;

      IslandClusterAlgo * island_p;

      bool counterExceeded() const { return ((nEvt_ > nMaxPrintout_) || (nMaxPrintout_ < 0)); }

      const EcalRecHitCollection * getCollection(edm::Event& evt,
                                                 const std::string& hitProducer_,
                                                 const std::string& hitCollection_);


      void clusterizeECALPart(edm::Event &evt, const edm::EventSetup &es,
			      const std::string& hitProducer,
			      const std::string& hitCollection,
			      const std::string& clusterCollection,
			      const IslandClusterAlgo::EcalPart& ecalPart);

      void outputValidationInfo(reco::BasicClusterRefVector &clusterRefVector);
};


#endif
