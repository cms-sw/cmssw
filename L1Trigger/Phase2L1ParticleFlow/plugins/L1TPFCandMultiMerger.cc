#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/transform.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "DataFormats/Common/interface/CloneTrait.h"
#include <vector>

class L1TPFCandMultiMerger : public edm::global::EDProducer<> {
public:
  explicit L1TPFCandMultiMerger(const edm::ParameterSet&);
  ~L1TPFCandMultiMerger() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  std::vector<std::string> instances_, regionalInstances_;
  std::vector<bool> alsoRegional_;  // aligned with instances_
  std::vector<edm::EDGetTokenT<l1t::PFCandidateCollection>> tokens_;
  std::vector<edm::EDGetTokenT<l1t::PFCandidateRegionalOutput>> regionalTokens_;
};

L1TPFCandMultiMerger::L1TPFCandMultiMerger(const edm::ParameterSet& iConfig)
    : instances_(iConfig.getParameter<std::vector<std::string>>("labelsToMerge")),
      regionalInstances_(iConfig.getParameter<std::vector<std::string>>("regionalLabelsToMerge")) {
  const std::vector<edm::InputTag>& pfProducers = iConfig.getParameter<std::vector<edm::InputTag>>("pfProducers");
  tokens_.reserve(instances_.size() * pfProducers.size());
  for (const std::string& instance : instances_) {
    for (const edm::InputTag& tag : pfProducers) {
      tokens_.push_back(consumes<l1t::PFCandidateCollection>(edm::InputTag(tag.label(), instance, tag.process())));
    }
    produces<l1t::PFCandidateCollection>(instance);
    // check if regional output is needed too
    if (std::find(regionalInstances_.begin(), regionalInstances_.end(), instance) != regionalInstances_.end()) {
      alsoRegional_.push_back(true);
      for (const edm::InputTag& tag : pfProducers) {
        regionalTokens_.push_back(
            consumes<l1t::PFCandidateRegionalOutput>(edm::InputTag(tag.label(), instance + "Regional", tag.process())));
      }
      produces<l1t::PFCandidateRegionalOutput>(instance + "Regional");
    } else {
      alsoRegional_.push_back(false);
    }
  }
  // check that regional output is not requested without the standard one
  for (const std::string& instance : regionalInstances_) {
    auto match = std::find(instances_.begin(), instances_.end(), instance);
    if (match == instances_.end()) {
      throw cms::Exception("Configuration", "The regional label '" + instance + "' is not in labelsToMerge\n");
    }
  }
}

L1TPFCandMultiMerger::~L1TPFCandMultiMerger() {}

void L1TPFCandMultiMerger::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  edm::Handle<l1t::PFCandidateCollection> handle;
  edm::Handle<l1t::PFCandidateRegionalOutput> regionalHandle;
  unsigned int ninstances = instances_.size(), nproducers = tokens_.size() / ninstances;
  std::vector<int> keys;
  for (unsigned int ii = 0, it = 0, irt = 0; ii < ninstances; ++ii) {
    auto out = std::make_unique<l1t::PFCandidateCollection>();
    std::unique_ptr<l1t::PFCandidateRegionalOutput> regout;
    if (alsoRegional_[ii]) {
      auto refprod = iEvent.getRefBeforePut<l1t::PFCandidateCollection>(instances_[ii]);
      regout = std::make_unique<l1t::PFCandidateRegionalOutput>(edm::RefProd<l1t::PFCandidateCollection>(refprod));
    }
    for (unsigned int ip = 0; ip < nproducers; ++ip, ++it) {
      iEvent.getByToken(tokens_[it], handle);
      unsigned int offset = out->size();
      out->insert(out->end(), handle->begin(), handle->end());
      if (alsoRegional_[ii]) {
        iEvent.getByToken(regionalTokens_[irt++], regionalHandle);
        const auto& src = *regionalHandle;
        for (unsigned int ireg = 0, nreg = src.nRegions(); ireg < nreg; ++ireg) {
          auto region = src.region(ireg);
          keys.clear();
          for (auto iter = region.begin(), iend = region.end(); iter != iend; ++iter) {
            keys.push_back(iter.idx() + offset);
          }
          regout->addRegion(keys, src.eta(ireg), src.phi(ireg));
        }
      }
    }
    iEvent.put(std::move(out), instances_[ii]);
    if (alsoRegional_[ii]) {
      iEvent.put(std::move(regout), instances_[ii] + "Regional");
    }
  }
}

void L1TPFCandMultiMerger::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // l1tLayer1
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("pfProducers",
                                       {
                                           edm::InputTag("l1tLayer1Barrel"),
                                           edm::InputTag("l1tLayer1HGCal"),
                                           edm::InputTag("l1tLayer1HGCalNoTK"),
                                           edm::InputTag("l1tLayer1HF"),
                                       });
  desc.add<std::vector<std::string>>("labelsToMerge",
                                     {
                                         "PF",
                                         "Puppi",
                                         "Calo",
                                         "TK",
                                     });
  desc.add<std::vector<std::string>>("regionalLabelsToMerge",
                                     {
                                         "Puppi",
                                     });
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TPFCandMultiMerger);
