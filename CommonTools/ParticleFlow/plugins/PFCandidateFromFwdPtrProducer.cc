#include "CommonTools/UtilAlgos/interface/ProductFromFwdPtrProducer.h"
#include "DataFormats/JetReco/interface/PFJet.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "CommonTools/ParticleFlow/interface/PFCandidateWithSrcPtrFactory.h"

typedef edm::ProductFromFwdPtrProducer<reco::PFCandidate, reco::PFCandidateWithSrcPtrFactory>
    PFCandidateFromFwdPtrProducer;
typedef edm::ProductFromFwdPtrProducer<reco::PFJet> PFJetFromFwdPtrProducer;

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(PFCandidateFromFwdPtrProducer);
DEFINE_FWK_MODULE(PFJetFromFwdPtrProducer);
