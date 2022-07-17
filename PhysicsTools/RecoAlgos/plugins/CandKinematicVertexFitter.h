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
      explicit CandKinematicVertexFitterEventSetupInit(edm::ConsumesCollector iC)
          : magToken_(iC.esConsumes()), pdtToken_(iC.esConsumes()) {}

      void init(CandKinematicVertexFitter& fitter, const edm::Event& evt, const edm::EventSetup& es) {
        fitter.set(&es.getData(magToken_));
        fitter.set(&es.getData(pdtToken_));
      }

      edm::ESGetToken<MagneticField, IdealMagneticFieldRecord> magToken_;
      edm::ESGetToken<ParticleDataTable, edm::DefaultRecord> pdtToken_;
    };

    template <>
    struct EventSetupInit<CandKinematicVertexFitter> {
      typedef CandKinematicVertexFitterEventSetupInit type;
    };
  }  // namespace modules
}  // namespace reco

#endif
