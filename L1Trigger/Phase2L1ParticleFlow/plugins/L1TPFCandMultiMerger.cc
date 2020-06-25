#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/CloneTrait.h"
#include <vector>

class L1TPFCandMultiMerger : public edm::global::EDProducer<> {
public:
  explicit L1TPFCandMultiMerger(const edm::ParameterSet&);
  ~L1TPFCandMultiMerger() override;

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  std::vector<std::string> instances_;
  std::vector<edm::EDGetTokenT<std::vector<l1t::PFCandidate>>> tokens_;
};

L1TPFCandMultiMerger::L1TPFCandMultiMerger(const edm::ParameterSet& iConfig)
    : instances_(iConfig.getParameter<std::vector<std::string>>("labelsToMerge")) {
  const std::vector<edm::InputTag>& pfProducers = iConfig.getParameter<std::vector<edm::InputTag>>("pfProducers");
  tokens_.reserve(instances_.size() * pfProducers.size());
  for (unsigned int ii = 0, ni = instances_.size(); ii < ni; ++ii) {
    for (const edm::InputTag& tag : pfProducers) {
      tokens_.push_back(
          consumes<std::vector<l1t::PFCandidate>>(edm::InputTag(tag.label(), instances_[ii], tag.process())));
    }
    produces<std::vector<l1t::PFCandidate>>(instances_[ii]);
  }
}

L1TPFCandMultiMerger::~L1TPFCandMultiMerger() {}

void L1TPFCandMultiMerger::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  edm::Handle<std::vector<l1t::PFCandidate>> handle;
  for (unsigned int ii = 0, it = 0, ni = instances_.size(), np = tokens_.size() / ni; ii < ni; ++ii) {
    auto out = std::make_unique<std::vector<l1t::PFCandidate>>();
    for (unsigned int ip = 0; ip < np; ++ip, ++it) {
      iEvent.getByToken(tokens_[it], handle);
      out->insert(out->end(), handle->begin(), handle->end());
    }
    iEvent.put(std::move(out), instances_[ii]);
  }
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TPFCandMultiMerger);
