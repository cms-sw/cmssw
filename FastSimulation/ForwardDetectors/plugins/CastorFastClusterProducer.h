#ifndef FastSimulation_ForwardDetectors_CastorFastClusterProducer_h
#define FastSimulation_ForwardDetectors_CastorFastClusterProducer_h

// Castorobject includes
#include "DataFormats/CastorReco/interface/CastorCluster.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/Math/interface/Point3D.h"


//
// class decleration
//

class CastorFastClusterProducer : public edm::EDProducer {
   public:
      explicit CastorFastClusterProducer(const edm::ParameterSet&);
      ~CastorFastClusterProducer();

   private:
      virtual void beginRun(edm::Run&, edm::EventSetup const&) ;
      virtual void produce(edm::Event&, const edm::EventSetup&);
      double make_noise();
      virtual void endRun() ;
      
      // ----------member data ---------------------------
      typedef math::XYZPointD Point;
      typedef ROOT::Math::RhoEtaPhiPoint ClusterPoint;
      typedef std::vector<reco::CastorCluster> CastorClusterCollection;
};

#endif


