#include "PhysicsTools/RecoAlgos/plugins/ConstrainedFitCandProducer.h"
#include "PhysicsTools/RecoAlgos/plugins/CandKinematicVertexFitter.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"

typedef ConstrainedFitCandProducer<CandKinematicVertexFitter,
                                   edm::View<reco::Candidate>,
                                   std::vector<reco::VertexCompositeCandidate> > KinematicVertexFitCompositeCandProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(KinematicVertexFitCompositeCandProducer);


