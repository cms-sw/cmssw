#ifndef RecoAlgos_KalmanVertexFitter_h
#define RecoAlgos_KalmanVertexFitter_h
#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"
#include "RecoVertex/KalmanVertexFit/interface/KalmanVertexFitter.h"

const bool refitTracks = true;

namespace reco {
  namespace modules {
    
    template<>
    struct ParameterAdapter<KalmanVertexFitter> {
      static KalmanVertexFitter make( const edm::ParameterSet & cfg ) {
        return KalmanVertexFitter( refitTracks );
      }
    };

  }
}

#endif
