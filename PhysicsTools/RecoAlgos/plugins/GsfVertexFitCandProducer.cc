#include "PhysicsTools/RecoAlgos/plugins/ConstrainedFitCandProducer.h"
#include "PhysicsTools/RecoAlgos/plugins/CandCommonVertexFitter.h"
#include "PhysicsTools/RecoAlgos/plugins/GsfVertexFitter.h"

typedef ConstrainedFitCandProducer<CandCommonVertexFitter<GsfVertexFitter> > GsfVertexFitCandProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GsfVertexFitCandProducer );


