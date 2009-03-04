#ifndef RecoAlgos_KalmanVertexFitter_h
#define RecoAlgos_KalmanVertexFitter_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "RecoVertex/GaussianSumVertexFit/interface/GsfVertexFitter.h"

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<GsfVertexFitter> {
      static  GsfVertexFitter make( const edm::ParameterSet & cfg ) {
        return GsfVertexFitter( cfg );
      }
    };

  }
}

#endif
