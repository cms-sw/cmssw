#ifndef IsolationAlgos_CalIsolationAlgo_h
#define IsolationAlgos_CalIsolationAlgo_h
/* Partial spacialization of parameter set adapeter helper
 *
 */
#include "PhysicsTools/IsolationAlgos/interface/CalIsolationAlgo.h"
#include "PhysicsTools/IsolationAlgos/interface/IsolationProducer.h"
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

namespace helper {

  template <typename Alg>
  struct BFieldIsolationAlgorithmSetup {
    using ESConsumesToken = edm::ESGetToken<MagneticField, IdealMagneticFieldRecord>;
    static ESConsumesToken esConsumes(edm::ConsumesCollector cc) {
      return cc.esConsumes<MagneticField, IdealMagneticFieldRecord>();
    }
    static void init(Alg& algo, const edm::EventSetup& es, const ESConsumesToken& token) {
      algo.setBfield(&es.getData(token));
    }
  };

  template <typename T1, typename C2>
  struct IsolationAlgorithmSetup<CalIsolationAlgo<T1, C2> > {
    typedef BFieldIsolationAlgorithmSetup<CalIsolationAlgo<T1, C2> > type;
  };
}  // namespace helper

namespace reco {
  namespace modules {

    template <typename T, typename C>
    struct ParameterAdapter<CalIsolationAlgo<T, C> > {
      static CalIsolationAlgo<T, C> make(const edm::ParameterSet& cfg) {
        bool propagate = cfg.template getParameter<bool>("PropagateToCal");
        double r = 0.0, minz = 0.0, maxz = 0.0;
        bool material = false;
        //allow for undefined propagation-parameters, if no propagation is wanted
        if (propagate) {
          r = cfg.template getParameter<double>("CalRadius");
          minz = cfg.template getParameter<double>("CalMinZ");
          maxz = cfg.template getParameter<double>("CalMaxZ");
          material = cfg.template getParameter<bool>("IgnoreMaterial");
        }
        return CalIsolationAlgo<T, C>(cfg.template getParameter<double>("dRMin"),
                                      cfg.template getParameter<double>("dRMax"),
                                      propagate,
                                      r,
                                      minz,
                                      maxz,
                                      material);
      }
    };
  }  // namespace modules
}  // namespace reco

#endif
