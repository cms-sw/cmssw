#ifndef FastSimulation_ForwardDetectors_CastorFastClusterProducer_h
#define FastSimulation_ForwardDetectors_CastorFastClusterProducer_h

#include "FWCore/Framework/interface/stream/EDProducer.h"

// Castorobject includes
#include "DataFormats/CastorReco/interface/CastorCluster.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/Point3D.h"



//
// class decleration
//

class CastorFastClusterProducer : public edm::stream::EDProducer <>{
   public:
      explicit CastorFastClusterProducer(const edm::ParameterSet&);
      ~CastorFastClusterProducer();

   private:
      virtual void produce(edm::Event&, const edm::EventSetup&) override;
      double make_noise();
      
      // ----------member data ---------------------------
      typedef math::XYZPointD Point;
      typedef ROOT::Math::RhoEtaPhiPoint ClusterPoint;
      typedef std::vector<reco::CastorCluster> CastorClusterCollection;
};

#endif


