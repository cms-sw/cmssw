#ifndef RecoAlgos_CandKalmanVertexFitter_h
#define RecoAlgos_CandKalmanVertexFitter_h
#include "PhysicsTools/UtilAlgos/interface/EventSetupInitTrait.h"
#include "PhysicsTools/RecoCandUtils/interface/CandCommonVertexFitter.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace reco {
  namespace modules {
    template<typename Fitter>
    struct CandVertexFitterEventSetupInit {
      static void init( CandCommonVertexFitter<Fitter> & fitter, 
			const edm::EventSetup& es ) { 
	edm::ESHandle<MagneticField> h;
	es.get<IdealMagneticFieldRecord>().get( h );
	fitter.set( h.product() );
      }
    };

    template<typename Fitter>
    struct EventSetupInit<CandCommonVertexFitter<Fitter> > {
      typedef CandVertexFitterEventSetupInit<Fitter> type;
    };
  }
}

#endif
