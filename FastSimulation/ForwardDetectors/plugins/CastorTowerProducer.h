#ifndef FastSimulation_CaloRecHitsProducer_CastorTowerProducer_h
#define FastSimulation_CaloRecHitsProducer_CastorTowerProducer_h

// Castorobject includes
#include "DataFormats/CastorReco/interface/CastorTower.h"
#include "DataFormats/CastorReco/interface/CastorCell.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/Point3D.h"


//
// class decleration
//

class CastorTowerProducer : public edm::EDProducer {
   public:
      explicit CastorTowerProducer(const edm::ParameterSet&);
      ~CastorTowerProducer();

   private:
      virtual void beginRun(edm::Run&, edm::EventSetup const&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      virtual void endRun() ;
      
      // ----------member data ---------------------------
      typedef math::XYZPointD Point;
      typedef ROOT::Math::RhoEtaPhiPoint TowerPoint;
      typedef std::vector<reco::CastorTower> CastorTowerCollection;
};

#endif


