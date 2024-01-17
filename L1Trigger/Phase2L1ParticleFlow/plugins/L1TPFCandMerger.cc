#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/Merger.h"

typedef Merger<std::vector<l1t::PFCandidate>> L1TPFCandMerger;

template <>
void L1TPFCandMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("src",
                                       {
                                           edm::InputTag("collection1"),
                                           edm::InputTag("collection2"),
                                       });
  descriptions.add("l1TPFCandMerger", desc);
}

DEFINE_FWK_MODULE(L1TPFCandMerger);
