#include "PhysicsTools/RecoAlgos/plugins/ConstrainedFitCandProducer.h"
#include "PhysicsTools/RecoAlgos/plugins/CandCommonVertexFitter.h"
#include "PhysicsTools/RecoAlgos/plugins/GsfVertexFitter.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"

typedef ConstrainedFitCandProducer<CandCommonVertexFitter<GsfVertexFitter>,
				   edm::View<reco::Candidate>,
                                   std::vector<reco::VertexCompositeCandidate> > GsfVertexFitCompositeCandProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GsfVertexFitCompositeCandProducer );


