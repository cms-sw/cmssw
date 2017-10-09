#ifndef FastSimulation_ForwardDetectors_CastorFastTowerProducer_h
#define FastSimulation_ForwardDetectors_CastorFastTowerProducer_h

// Castorobject includes
#include "DataFormats/CastorReco/interface/CastorTower.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/Point3D.h"

#include "FWCore/Framework/interface/stream/EDProducer.h"


//
// class decleration
//

class CastorFastTowerProducer : public edm::stream::EDProducer <> {
   public:
      explicit CastorFastTowerProducer(const edm::ParameterSet&);
      ~CastorFastTowerProducer();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      double make_noise();
      
      // ----------member data ---------------------------
      typedef math::XYZPointD Point;
      typedef ROOT::Math::RhoEtaPhiPoint TowerPoint;
      typedef std::vector<reco::CastorTower> CastorTowerCollection;
};

#endif


