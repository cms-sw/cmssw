/**
 * BoardDataWriter for validation with hardware. Currently only writing the algo bits is implemented.
 **/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "L1Trigger/DemonstratorTools/interface/BoardDataWriter.h"
#include "L1Trigger/DemonstratorTools/interface/utilities.h"

#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDMException.h"

#include "DataFormats/L1Trigger/interface/P2GTAlgoBlock.h"

#include "ap_int.h"

#include <vector>
#include <algorithm>
#include <string>
#include <array>

using namespace l1t;

class L1GTFinOrBoardWriter : public edm::one::EDAnalyzer<> {
public:
  explicit L1GTFinOrBoardWriter(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  unsigned int eventCounter_;
  const unsigned int maxEvents_;
  const std::array<unsigned int, 3> channelsLow_;
  const std::array<unsigned int, 3> channelsMid_;
  const std::array<unsigned int, 3> channelsHigh_;
  const unsigned int channelFinOr_;
  const edm::EDGetTokenT<P2GTAlgoBlockMap> algoBlocksToken_;
  l1t::demo::BoardDataWriter boardDataWriter_;

  std::map<l1t::demo::LinkId, std::vector<ap_uint<64>>> linkData_;
  std::size_t tmuxCounter_;
};

template <typename T, std::size_t N>
static std::array<T, N> convert(std::vector<T> vec, const char* name) {
  if (vec.size() != N) {
    throw edm::Exception(edm::errors::Configuration)
        << "The parameter '" << name << "' should have " << N << " elements, but has " << vec.size()
        << " elements in the configuration.\n";
  }
  std::array<T, N> a;
  std::copy_n(std::make_move_iterator(vec.begin()), N, a.begin());
  return a;
}

L1GTFinOrBoardWriter::L1GTFinOrBoardWriter(const edm::ParameterSet& config)
    : eventCounter_(0),
      maxEvents_(config.getUntrackedParameter<unsigned int>("maxEvents")),
      channelsLow_(convert<unsigned int, 3>(config.getUntrackedParameter<std::vector<unsigned int>>("channelsLow"),
                                            "channelsLow")),
      channelsMid_(convert<unsigned int, 3>(config.getUntrackedParameter<std::vector<unsigned int>>("channelsMid"),
                                            "channelsMid")),
      channelsHigh_(convert<unsigned int, 3>(config.getUntrackedParameter<std::vector<unsigned int>>("channelsHigh"),
                                             "channelsHigh")),
      channelFinOr_(config.getUntrackedParameter<unsigned int>("channelFinOr")),
      algoBlocksToken_(consumes<P2GTAlgoBlockMap>(config.getUntrackedParameter<edm::InputTag>("algoBlocksTag"))),
      boardDataWriter_(l1t::demo::parseFileFormat(config.getUntrackedParameter<std::string>("patternFormat")),
                       config.getUntrackedParameter<std::string>("filename"),
                       config.getUntrackedParameter<std::string>("fileExtension"),
                       9,
                       2,
                       config.getUntrackedParameter<unsigned int>("maxFrames"),
                       [&]() {
                         l1t::demo::BoardDataWriter::ChannelMap_t channelMap;
                         channelMap.insert({l1t::demo::LinkId{"BeforeBxMaskAndPrescaleLow", channelsLow_[0]},
                                            {l1t::demo::ChannelSpec{2, 0, 0}, {channelsLow_[0]}}});
                         channelMap.insert({l1t::demo::LinkId{"BeforePrescaleLow", channelsLow_[1]},
                                            {l1t::demo::ChannelSpec{2, 0, 0}, {channelsLow_[1]}}});
                         channelMap.insert({l1t::demo::LinkId{"FinalLow", channelsLow_[2]},
                                            {l1t::demo::ChannelSpec{2, 0, 0}, {channelsLow_[2]}}});

                         channelMap.insert({l1t::demo::LinkId{"BeforeBxMaskAndPrescaleMid", channelsMid_[0]},
                                            {l1t::demo::ChannelSpec{2, 0, 0}, {channelsMid_[0]}}});
                         channelMap.insert({l1t::demo::LinkId{"BeforePrescaleMid", channelsMid_[1]},
                                            {l1t::demo::ChannelSpec{2, 0, 0}, {channelsMid_[1]}}});
                         channelMap.insert({l1t::demo::LinkId{"FinalMid", channelsMid_[2]},
                                            {l1t::demo::ChannelSpec{2, 0, 0}, {channelsMid_[2]}}});

                         channelMap.insert({l1t::demo::LinkId{"BeforeBxMaskAndPrescaleHigh", channelsHigh_[0]},
                                            {l1t::demo::ChannelSpec{2, 0, 0}, {channelsHigh_[0]}}});
                         channelMap.insert({l1t::demo::LinkId{"BeforePrescaleHigh", channelsHigh_[1]},
                                            {l1t::demo::ChannelSpec{2, 0, 0}, {channelsHigh_[1]}}});
                         channelMap.insert({l1t::demo::LinkId{"FinalHigh", channelsHigh_[2]},
                                            {l1t::demo::ChannelSpec{2, 0, 0}, {channelsHigh_[2]}}});

                         channelMap.insert({l1t::demo::LinkId{"FinOr", channelFinOr_},
                                            {l1t::demo::ChannelSpec{2, 0, 0}, {channelFinOr_}}});

                         return channelMap;
                       }()),
      linkData_(),
      tmuxCounter_(0) {}

void L1GTFinOrBoardWriter::analyze(const edm::Event& event, const edm::EventSetup& iSetup) {
  const P2GTAlgoBlockMap& algoBlocks = event.get(algoBlocksToken_);

  auto algoBlockIt = algoBlocks.begin();

  if (tmuxCounter_ == 0) {
    linkData_[l1t::demo::LinkId{"BeforeBxMaskAndPrescaleLow", channelsLow_[0]}] = std::vector<ap_uint<64>>(18, 0);
    linkData_[l1t::demo::LinkId{"BeforePrescaleLow", channelsLow_[1]}] = std::vector<ap_uint<64>>(18, 0);
    linkData_[l1t::demo::LinkId{"FinalLow", channelsLow_[2]}] = std::vector<ap_uint<64>>(18, 0);
    linkData_[l1t::demo::LinkId{"BeforeBxMaskAndPrescaleMid", channelsMid_[0]}] = std::vector<ap_uint<64>>(18, 0);
    linkData_[l1t::demo::LinkId{"BeforePrescaleMid", channelsMid_[1]}] = std::vector<ap_uint<64>>(18, 0);
    linkData_[l1t::demo::LinkId{"FinalMid", channelsMid_[2]}] = std::vector<ap_uint<64>>(18, 0);
    linkData_[l1t::demo::LinkId{"BeforeBxMaskAndPrescaleHigh", channelsHigh_[0]}] = std::vector<ap_uint<64>>(18, 0);
    linkData_[l1t::demo::LinkId{"BeforePrescaleHigh", channelsHigh_[1]}] = std::vector<ap_uint<64>>(18, 0);
    linkData_[l1t::demo::LinkId{"FinalHigh", channelsHigh_[2]}] = std::vector<ap_uint<64>>(18, 0);
    linkData_[l1t::demo::LinkId{"FinOr", channelFinOr_}] = std::vector<ap_uint<64>>(18, 0);
  }

  for (std::size_t word = 0; word < 9; word++) {
    for (std::size_t idx = 0; idx < 64 && algoBlockIt != algoBlocks.end(); idx++, algoBlockIt++) {
      auto& [alogName, algoBlock] = *algoBlockIt;
      linkData_[l1t::demo::LinkId{"BeforeBxMaskAndPrescaleLow", channelsLow_[0]}][word + tmuxCounter_ * 9].set(
          idx, algoBlock.decisionBeforeBxMaskAndPrescale());
      linkData_[l1t::demo::LinkId{"BeforePrescaleLow", channelsLow_[1]}][word + tmuxCounter_ * 9].set(
          idx, algoBlock.decisionBeforePrescale());
      linkData_[l1t::demo::LinkId{"FinalLow", channelsLow_[2]}][word + tmuxCounter_ * 9].set(idx,
                                                                                             algoBlock.decisionFinal());
    }
  }

  for (std::size_t word = 0; word < 9; word++) {
    for (std::size_t idx = 0; idx < 64 && algoBlockIt != algoBlocks.end(); idx++, algoBlockIt++) {
      auto& [alogName, algoBlock] = *algoBlockIt;
      linkData_[l1t::demo::LinkId{"BeforeBxMaskAndPrescaleMid", channelsMid_[0]}][word + tmuxCounter_ * 9].set(
          idx, algoBlock.decisionBeforeBxMaskAndPrescale());
      linkData_[l1t::demo::LinkId{"BeforePrescaleMid", channelsMid_[1]}][word + tmuxCounter_ * 9].set(
          idx, algoBlock.decisionBeforePrescale());
      linkData_[l1t::demo::LinkId{"FinalMid", channelsMid_[2]}][word + tmuxCounter_ * 9].set(idx,
                                                                                             algoBlock.decisionFinal());
    }
  }

  for (std::size_t word = 0; word < 9; word++) {
    for (std::size_t idx = 0; idx < 64 && algoBlockIt != algoBlocks.end(); idx++, algoBlockIt++) {
      auto& [algoName, algoBlock] = *algoBlockIt;
      linkData_[l1t::demo::LinkId{"BeforeBxMaskAndPrescaleHigh", channelsHigh_[0]}][word + tmuxCounter_ * 9].set(
          idx, algoBlock.decisionBeforeBxMaskAndPrescale());
      linkData_[l1t::demo::LinkId{"BeforePrescaleHigh", channelsHigh_[1]}][word + tmuxCounter_ * 9].set(
          idx, algoBlock.decisionBeforePrescale());
      linkData_[l1t::demo::LinkId{"FinalHigh", channelsHigh_[2]}][word + tmuxCounter_ * 9].set(
          idx, algoBlock.decisionFinal());
    }
  }

  bool vetoed = false, vetoedPreview = false;
  int finOrByTypes = 0, finOrPreviewByTypes = 0;
  for (auto algoBlockIt = algoBlocks.begin(); algoBlockIt != algoBlocks.end(); algoBlockIt++) {
    auto& [alogName, algoBlock] = *algoBlockIt;
    vetoed |= (algoBlock.isVeto() && algoBlock.decisionFinal());
    vetoedPreview |= (algoBlock.isVeto() && algoBlock.decisionFinalPreview());
    finOrByTypes |= algoBlock.decisionFinal() ? algoBlock.triggerTypes() : 0;
    finOrPreviewByTypes |= algoBlock.decisionFinalPreview() ? algoBlock.triggerTypes() : 0;
  }

  // Add FinOrTrigger bits per https://gitlab.cern.ch/cms-cactus/phase2/firmware/gt-final-or#output-finor-bits
  ap_uint<64> finOrBits(0);
  finOrBits(7, 0) = finOrByTypes;
  finOrBits(15, 8) = finOrPreviewByTypes;
  finOrBits(23, 16) = vetoed ? 0 : finOrByTypes;
  finOrBits(31, 24) = vetoedPreview ? 0 : finOrPreviewByTypes;

  linkData_[l1t::demo::LinkId{"FinOr", channelFinOr_}][0 + tmuxCounter_ * 9] = finOrBits;

  if (tmuxCounter_ == 1) {
    boardDataWriter_.addEvent(l1t::demo::EventData(linkData_));
  }

  tmuxCounter_ = (tmuxCounter_ + 1) % 2;

  eventCounter_++;

  if (maxEvents_ != 0 && eventCounter_ == maxEvents_) {
    boardDataWriter_.flush();
    eventCounter_ = 0;
  }
}

void L1GTFinOrBoardWriter::endJob() {
  if (tmuxCounter_ == 1) {
    boardDataWriter_.addEvent(l1t::demo::EventData(linkData_));
  }

  boardDataWriter_.flush();
}

void L1GTFinOrBoardWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::string>("filename");
  desc.addUntracked<std::string>("fileExtension", "txt");
  desc.addUntracked<edm::InputTag>("algoBlocksTag");
  desc.addUntracked<std::vector<unsigned int>>("channelsLow");
  desc.addUntracked<std::vector<unsigned int>>("channelsMid");
  desc.addUntracked<std::vector<unsigned int>>("channelsHigh");
  desc.addUntracked<unsigned int>("channelFinOr");
  desc.addUntracked<unsigned int>("maxFrames", 1024);
  desc.addUntracked<unsigned int>("maxEvents", 0);
  desc.addUntracked<std::string>("patternFormat", "EMPv2");

  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(L1GTFinOrBoardWriter);
