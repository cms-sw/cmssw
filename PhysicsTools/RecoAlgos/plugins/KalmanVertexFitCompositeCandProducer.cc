#include "PhysicsTools/RecoAlgos/plugins/ConstrainedFitCandProducer.h"
#include "PhysicsTools/RecoAlgos/plugins/CandCommonVertexFitter.h"
#include "PhysicsTools/RecoAlgos/plugins/KalmanVertexFitter.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"

typedef ConstrainedFitCandProducer<CandCommonVertexFitter<KalmanVertexFitter>,
                                   edm::View<reco::Candidate>,
                                   std::vector<reco::VertexCompositeCandidate> > KalmanVertexFitCompositeCandProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( KalmanVertexFitCompositeCandProducer );


