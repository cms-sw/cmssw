#include "PhysicsTools/RecoAlgos/plugins/ConstrainedFitCandProducer.h"
#include "PhysicsTools/RecoAlgos/plugins/CandCommonVertexFitter.h"
#include "PhysicsTools/RecoAlgos/plugins/GsfVertexFitter.h"
#include "DataFormats/Candidate/interface/CompositeCandidate.h"

typedef ConstrainedFitCandProducer<CandCommonVertexFitter<GsfVertexFitter>,
                                   std::vector<reco::CompositeCandidate> > GsfVertexFitCompositeCandProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GsfVertexFitCompositeCandProducer );


