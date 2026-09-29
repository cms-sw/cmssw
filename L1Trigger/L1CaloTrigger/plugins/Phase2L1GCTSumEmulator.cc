/* 
 * Description: Phase 2 GCT SumCard emulator
 * Author: Mila Bileska
 */

#include <ap_int.h>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

#include "L1Trigger/L1CaloTrigger/interface/GCTSum_h.h"
#include "L1Trigger/L1CaloTrigger/interface/GCTSum_cpp.h"
#include "L1Trigger/L1CaloTrigger/interface/GCTSumToGT_h.h"
#include "L1Trigger/L1CaloTrigger/interface/GCTSumToGT_cpp.h"
#include "L1Trigger/L1CaloTrigger/interface/bitonicSort32_GCT_h.h"
#include "L1Trigger/L1CaloTrigger/interface/bitonicSort32_GCT_cpp.h"

namespace {
  constexpr unsigned int kWordsPerLink = 9;
  constexpr unsigned int kSideInputLinks = 12;
  constexpr unsigned int kSideOutputLinks = 3;
  constexpr unsigned int kInputLinks = 2 * kSideInputLinks;
  constexpr unsigned int kOutputLinks = 2 * kSideOutputLinks;
}  // namespace

class Phase2L1GCTSumEmulator : public edm::stream::EDProducer<> {
public:
  explicit Phase2L1GCTSumEmulator(const edm::ParameterSet&);
  ~Phase2L1GCTSumEmulator() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  std::array<edm::EDGetTokenT<std::vector<uint64_t>>, kInputLinks> inputLinkTokens_;
  bool debug_;
};

Phase2L1GCTSumEmulator::Phase2L1GCTSumEmulator(const edm::ParameterSet& iConfig)
    : debug_(iConfig.getParameter<bool>("debug")) {
  const auto inputLinks = iConfig.getParameter<std::vector<edm::InputTag>>("inputLinks");
  if (inputLinks.size() != inputLinkTokens_.size()) {
    throw cms::Exception("Phase2L1GCTSumEmulator")
        << "Expected exactly " << inputLinkTokens_.size() << " input links (12 positive eta + 12 negative eta)";
  }

  for (unsigned int i = 0; i < inputLinkTokens_.size(); ++i) {
    inputLinkTokens_[i] = consumes<std::vector<uint64_t>>(inputLinks[i]);
  }

  for (unsigned int i = 0; i < kOutputLinks; ++i) {
    produces<std::vector<uint64_t>>(std::string("LinkOut") + std::to_string(i));
    produces<std::vector<uint64_t>>(std::string("SumLinkOut") + std::to_string(i));
  }
}

void Phase2L1GCTSumEmulator::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  (void)iSetup;

  std::array<ap_uint<576>, kSideInputLinks> link_in_pos{};
  std::array<ap_uint<576>, kSideInputLinks> link_in_neg{};
  std::array<ap_uint<576>, kSideOutputLinks> link_out_pos{};
  std::array<ap_uint<576>, kSideOutputLinks> link_out_neg{};
  std::array<ap_uint<576>, kOutputLinks> link_in_gt{};
  std::array<ap_uint<576>, kOutputLinks> link_out_gt{};

  for (unsigned int i = 0; i < kSideInputLinks; ++i) {
    edm::Handle<std::vector<uint64_t>> handle;
    iEvent.getByToken(inputLinkTokens_[i], handle);
    if (!handle.isValid()) {
      throw cms::Exception("Phase2L1GCTSumEmulator") << "GCTSum positive-eta input link " << i << " is missing";
    }
    if (handle->size() != kWordsPerLink) {
      throw cms::Exception("Phase2L1GCTSumEmulator")
          << "GCTSum positive-eta input link " << i << " has " << handle->size() << " words; expected "
          << kWordsPerLink;
    }

    ap_uint<576> packed = 0;
    for (unsigned int word = 0; word < kWordsPerLink; ++word) {
      packed.range((word * 64) + 63, word * 64) = (*handle)[word];
    }
    link_in_pos[i] = packed;
  }

  for (unsigned int i = 0; i < kSideInputLinks; ++i) {
    edm::Handle<std::vector<uint64_t>> handle;
    iEvent.getByToken(inputLinkTokens_[kSideInputLinks + i], handle);
    if (!handle.isValid()) {
      throw cms::Exception("Phase2L1GCTSumEmulator") << "GCTSum negative-eta input link " << i << " is missing";
    }
    if (handle->size() != kWordsPerLink) {
      throw cms::Exception("Phase2L1GCTSumEmulator")
          << "GCTSum negative-eta input link " << i << " has " << handle->size() << " words; expected "
          << kWordsPerLink;
    }

    ap_uint<576> packed = 0;
    for (unsigned int word = 0; word < kWordsPerLink; ++word) {
      packed.range((word * 64) + 63, word * 64) = (*handle)[word];
    }
    link_in_neg[i] = packed;
  }

  p2gctsum::algo_top(link_in_pos.data(), link_out_pos.data());
  p2gctsum::algo_top(link_in_neg.data(), link_out_neg.data());

  for (unsigned int i = 0; i < kOutputLinks; ++i) {
    const ap_uint<576> sumLink = (i < kSideOutputLinks) ? link_out_pos[i] : link_out_neg[i - kSideOutputLinks];
    auto outWords = std::make_unique<std::vector<uint64_t>>();
    outWords->reserve(kWordsPerLink);

    for (unsigned int word = 0; word < kWordsPerLink; ++word) {
      outWords->push_back(sumLink.range((word * 64) + 63, word * 64).to_uint64());
    }

    iEvent.put(std::move(outWords), std::string("SumLinkOut") + std::to_string(i));
  }

  link_in_gt[0] = link_out_pos[0];
  link_in_gt[1] = link_out_pos[1];
  link_in_gt[2] = link_out_pos[2];
  link_in_gt[3] = link_out_neg[0];
  link_in_gt[4] = link_out_neg[1];
  link_in_gt[5] = link_out_neg[2];

  p2gctsumGT::algo_top_GT(link_in_gt.data(), link_out_gt.data());

  // These are the six event-level logical GT links. The firmware's TMUX6 and
  // 72-GT/6-scouting fanout are board transport stages that replicate and
  // serialize this same payload; they are intentionally not separate CMSSW
  // physics products.
  for (unsigned int i = 0; i < kOutputLinks; ++i) {
    auto outWords = std::make_unique<std::vector<uint64_t>>();
    outWords->reserve(kWordsPerLink);

    for (unsigned int word = 0; word < kWordsPerLink; ++word) {
      outWords->push_back(link_out_gt[i].range((word * 64) + 63, word * 64).to_uint64());
    }

    if (debug_) {
      edm::LogVerbatim("Phase2L1GCTSumEmulator") << "Output link " << i << " contains " << outWords->size()
                                                  << " words";
    }

    iEvent.put(std::move(outWords), std::string("LinkOut") + std::to_string(i));
  }
}

void Phase2L1GCTSumEmulator::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("inputLinks", std::vector<edm::InputTag>());
  desc.add<bool>("debug", false);
  descriptions.add("l1tPhase2L1GCTSumEmulator", desc);
}

DEFINE_FWK_MODULE(Phase2L1GCTSumEmulator);
