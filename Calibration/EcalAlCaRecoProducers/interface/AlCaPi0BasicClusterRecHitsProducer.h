/**

 Description: Producer for EcalRecHits to be used for pi0 ECAL calibration. ECAL Barrel RecHits and Basic 
              clusters are involved.

 Implementation:
     <Notes on implementation>
*/
//
// Original Author:  Vladimir Litvine


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "TrackingTools/TrackAssociator/interface/TimerStack.h"

//
// class declaration
//

class AlCaPi0BasicClusterRecHitsProducer : public edm::EDProducer {
   public:
      explicit AlCaPi0BasicClusterRecHitsProducer(const edm::ParameterSet&);
      ~AlCaPi0BasicClusterRecHitsProducer();


      virtual void produce(edm::Event &, const edm::EventSetup&);
   private:
      // ----------member data ---------------------------

 std::string ecalHitsProducer_;
 std::string barrelHits_;
 std::string pi0BarrelHits_;
 std::string islandBCProd_;
 std::string islandBCColl_;

 int gammaCandEtaSize_;
 int gammaCandPhiSize_;

 double selePtGammaOne_;
 double selePtGammaTwo_;
 double selePtPi0_;
 double seleMinvMaxPi0_;
 double seleMinvMinPi0_;

 std::map<DetId, EcalRecHit> *recHitsEB_map;

};
