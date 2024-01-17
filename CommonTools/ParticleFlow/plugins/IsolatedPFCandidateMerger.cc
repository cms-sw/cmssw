#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/IsolatedPFCandidate.h"
#include "CommonTools/UtilAlgos/interface/Merger.h"

typedef Merger<reco::IsolatedPFCandidateCollection> IsolatedPFCandidateMerger;

template <>
void IsolatedPFCandidateMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("src",
                                       {
                                           edm::InputTag("collection1"),
                                           edm::InputTag("collection2"),
                                       });
  descriptions.add("isolatedPFCandidateMerger", desc);
}

DEFINE_FWK_MODULE(IsolatedPFCandidateMerger);
