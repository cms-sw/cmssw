#ifndef RecoEcal_EgammaClusterProducers_EgammaHLTIslandClusterProducer_h_
#define RecoEcal_EgammaClusterProducers_EgammaHLTIslandClusterProducer_h_

#include <memory>
#include <time.h>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "RecoEcal/EgammaClusterAlgos/interface/IslandClusterAlgo.h"
#include "RecoEcal/EgammaCoreTools/interface/PositionCalc.h"
#include "RecoEcal/EgammaCoreTools/interface/ClusterShapeAlgo.h"

#include "Geometry/CaloTopology/interface/CaloSubdetectorTopology.h"

//


class EgammaHLTIslandClusterProducer : public edm::EDProducer 
{
  public:

      EgammaHLTIslandClusterProducer(const edm::ParameterSet& ps);

      ~EgammaHLTIslandClusterProducer();

      virtual void produce(edm::Event&, const edm::EventSetup&);

   private:

      int nMaxPrintout_; // max # of printouts
      int nEvt_;         // internal counter of events

      IslandClusterAlgo::VerbosityLevel verbosity;

      bool doBarrel_;
      bool doEndcaps_;
      bool doIsolated_;

      edm::InputTag barrelHitProducer_;
      edm::InputTag endcapHitProducer_;
      std::string barrelHitCollection_;
      std::string endcapHitCollection_;

      std::string barrelClusterCollection_;
      std::string endcapClusterCollection_;

      edm::InputTag l1TagIsolated_;
      edm::InputTag l1TagNonIsolated_;
      double l1LowerThr_;
      double l1UpperThr_;
      double l1LowerThrIgnoreIsolation_;

      double regionEtaMargin_;
      double regionPhiMargin_;

      PositionCalc posCalculator_; // position calculation algorithm
      IslandClusterAlgo * island_p;

      bool counterExceeded() const { return ((nEvt_ > nMaxPrintout_) || (nMaxPrintout_ < 0)); }

      const EcalRecHitCollection * getCollection(edm::Event& evt,
                                                 const std::string& hitProducer_,
                                                 const std::string& hitCollection_);


      void clusterizeECALPart(edm::Event &evt, const edm::EventSetup &es,
                              const std::string& hitProducer,
                              const std::string& hitCollection,
                              const std::string& clusterCollection,
                              const std::vector<EcalEtaPhiRegion>& regions,
                              const IslandClusterAlgo::EcalPart& ecalPart);

};


#endif
