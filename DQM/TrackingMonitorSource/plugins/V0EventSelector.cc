#include <memory>
#include <vector>
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/View.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/InputTag.h"

class V0EventSelector : public edm::stream::EDFilter<> {
public:
  explicit V0EventSelector(const edm::ParameterSet&);
  ~V0EventSelector() override = default;

  bool filter(edm::Event&, const edm::EventSetup&) override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::EDGetTokenT<reco::VertexCompositeCandidateCollection> vccToken_;
  const unsigned int minNumCandidates_;
  const double massMin_;
  const double massMax_;
};

V0EventSelector::V0EventSelector(const edm::ParameterSet& iConfig)
    : vccToken_{consumes<reco::VertexCompositeCandidateCollection>(
          iConfig.getParameter<edm::InputTag>("vertexCompositeCandidates"))},
      minNumCandidates_{iConfig.getParameter<unsigned int>("minCandidates")},
      massMin_{iConfig.getParameter<double>("massMin")},
      massMax_{iConfig.getParameter<double>("massMax")} {
  produces<reco::VertexCompositeCandidateCollection>();
}

bool V0EventSelector::filter(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<reco::VertexCompositeCandidateCollection> vccHandle;
  iEvent.getByToken(vccToken_, vccHandle);
  auto filteredVCC = std::make_unique<reco::VertexCompositeCandidateCollection>();

  // early return if the input collection is empty
  if (!vccHandle.isValid()) {
    iEvent.put(std::move(filteredVCC));
    return false;
  }

  for (const auto& vcc : *vccHandle) {
    if (vcc.mass() >= massMin_ && vcc.mass() <= massMax_) {
      filteredVCC->push_back(vcc);
    }
  }

  bool pass = filteredVCC->size() >= minNumCandidates_;
  iEvent.put(std::move(filteredVCC));

  return pass;
}

void V0EventSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("vertexCompositeCandidates", edm::InputTag("generalV0Candidates:Kshort"));
  desc.add<unsigned int>("minCandidates", 1)->setComment("Minimum number of candidates required");
  desc.add<double>("massMin", 0.0)->setComment("Minimum mass in GeV");
  desc.add<double>("massMax", std::numeric_limits<double>::max())->setComment("Maximum mass in GeV");
  descriptions.addWithDefaultLabel(desc);
}

// Define this module as a plug-in
DEFINE_FWK_MODULE(V0EventSelector);
