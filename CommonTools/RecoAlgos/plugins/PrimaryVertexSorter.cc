#include "CommonTools/RecoAlgos/plugins/PrimaryVertexSorter.h"
#include "DataFormats/RecoCandidate/interface/RecoChargedRefCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

typedef  PrimaryVertexSorter<std::vector<reco::RecoChargedRefCandidate> > RecoChargedRefCandidatePrimaryVertexSorter ;
DEFINE_FWK_MODULE(RecoChargedRefCandidatePrimaryVertexSorter);
typedef  PrimaryVertexSorter<std::vector<reco::PFCandidate> > PFCandidatePrimaryVertexSorter;
DEFINE_FWK_MODULE(PFCandidatePrimaryVertexSorter);
