#include <memory>
#include <vector>
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class V0EventSelector : public edm::stream::EDFilter<> {
public:
  explicit V0EventSelector(const edm::ParameterSet&);
  ~V0EventSelector() override = default;

  bool filter(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> vccToken_;
  const unsigned int minNumCandidates_;
};

V0EventSelector::V0EventSelector(const edm::ParameterSet& iConfig)
    : vccToken_{consumes<reco::VertexCompositeCandidateCollection>(
          iConfig.getParameter<edm::InputTag>("vertexCompositeCandidates"))},
      minNumCandidates_{iConfig.getParameter<unsigned int>("minCandidates")} {}

bool V0EventSelector::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<reco::VertexCompositeCandidateCollection> vccHandle;
  iEvent.getByToken(vccToken_, vccHandle);

  return vccHandle->size() >= minNumCandidates_;
}

void V0EventSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("vertexCompositeCandidates", edm::InputTag("generalV0Candidates:Kshort"));
  desc.add<unsigned int>("minCandidates", 1);  // Change '1' to your desired minimum number of candidates
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(V0EventSelector);
