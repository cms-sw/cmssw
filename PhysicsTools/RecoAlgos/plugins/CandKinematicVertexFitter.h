#ifndef RecoAlgos_CandKalmanVertexFitter_h
#define RecoAlgos_CandKalmanVertexFitter_h
#include "PhysicsTools/UtilAlgos/interface/EventSetupInitTrait.h"
#include "PhysicsTools/RecoCandUtils/interface/CandKinematicVertexFitter.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace reco {
  namespace modules {
    struct CandKinematicVertexFitterEventSetupInit {
      static void init(CandKinematicVertexFitter & fitter, 
		       const edm::Event & evt,
		       const edm::EventSetup& es) { 
	edm::ESHandle<MagneticField> h;
	es.get<IdealMagneticFieldRecord>().get(h);
	fitter.set(h.product());
      }
    };

    template<>
    struct EventSetupInit<CandKinematicVertexFitter> {
      typedef CandKinematicVertexFitterEventSetupInit type;
    };
  }
}

#endif
