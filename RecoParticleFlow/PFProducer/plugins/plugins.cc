#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/Merger.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

#include "CommonTools/UtilAlgos/interface/ProductFromFwdPtrProducer.h"
#include "CommonTools/ParticleFlow/interface/PFCandidateWithSrcPtrFactory.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "CommonTools/UtilAlgos/interface/FwdPtrProducer.h"
#include "CommonTools/ParticleFlow/interface/PFCandidateFwdPtrFactory.h"

typedef Merger<reco::PFCandidateCollection> PFCandidateListMerger;
typedef edm::ProductFromFwdPtrProducer<reco::PFCandidate, reco::PFCandidateWithSrcPtrFactory>
    PFCandidateProductFromFwdPtrProducer;
typedef edm::FwdPtrProducer<reco::PFCandidate, reco::PFCandidateFwdPtrFactory> PFCandidateFwdPtrProducer;

template <>
void PFCandidateListMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("src",
                                       {
                                           edm::InputTag("collection1"),
                                           edm::InputTag("collection2"),
                                       });
  descriptions.add("pfCandidateListMerger", desc);
}

DEFINE_FWK_MODULE(PFCandidateListMerger);
DEFINE_FWK_MODULE(PFCandidateProductFromFwdPtrProducer);
DEFINE_FWK_MODULE(PFCandidateFwdPtrProducer);
