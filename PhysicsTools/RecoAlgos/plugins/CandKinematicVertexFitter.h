#ifndef RecoAlgos_CandKalmanVertexFitter_h
#define RecoAlgos_CandKalmanVertexFitter_h
#include "CommonTools/UtilAlgos/interface/EventSetupInitTrait.h"
#include "PhysicsTools/RecoUtils/interface/CandKinematicVertexFitter.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "SimGeneral/HepPDTRecord/interface/ParticleDataTable.h"
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
	edm::ESHandle<ParticleDataTable> pdt;
	es.getData(pdt); 
	fitter.set(pdt.product());
      }
    };

    template<>
    struct EventSetupInit<CandKinematicVertexFitter> {
      typedef CandKinematicVertexFitterEventSetupInit type;
    };
  }
}

#endif
