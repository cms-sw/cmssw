#include "PhysicsTools/RecoAlgos/plugins/ConstrainedFitCandProducer.h"
#include "PhysicsTools/RecoAlgos/plugins/CandCommonVertexFitter.h"
#include "PhysicsTools/RecoAlgos/plugins/KalmanVertexFitter.h"

typedef ConstrainedFitCandProducer<CandCommonVertexFitter<KalmanVertexFitter> > KalmanVertexFitCandProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( KalmanVertexFitCandProducer );


