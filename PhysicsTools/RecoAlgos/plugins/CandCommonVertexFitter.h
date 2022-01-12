#ifndef RecoAlgos_CandKalmanVertexFitter_h
#define RecoAlgos_CandKalmanVertexFitter_h
#include "CommonTools/UtilAlgos/interface/EventSetupInitTrait.h"
#include "PhysicsTools/RecoUtils/interface/CandCommonVertexFitter.h"
#include "MagneticField/Records/interface/IdealMagneticFieldRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace reco {
  namespace modules {
    template <typename Fitter>
    struct CandVertexFitterEventSetupInit {
      explicit CandVertexFitterEventSetupInit(edm::ConsumesCollector iC) : magToken_(iC.esConsumes()) {}
      void init(CandCommonVertexFitter<Fitter>& fitter, const edm::Event& evt, const edm::EventSetup& es) {
        fitter.set(&es.getData(magToken_));
      }
      edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magToken_;
    };

    template <typename Fitter>
    struct EventSetupInit<CandCommonVertexFitter<Fitter> > {
      typedef CandVertexFitterEventSetupInit<Fitter> type;
    };
  }  // namespace modules
}  // namespace reco

#endif
