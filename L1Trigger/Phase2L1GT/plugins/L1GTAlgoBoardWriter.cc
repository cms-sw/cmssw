/**
 * AlgoBoardDataWriter for validation with hardware.
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

#include "DataFormats/L1Trigger/interface/P2GTAlgoBlock.h"

#include "ap_int.h"

#include <vector>
#include <algorithm>
#include <string>

using namespace l1t;

class L1GTAlgoBoardWriter : public edm::one::EDAnalyzer<> {
public:
  explicit L1GTAlgoBoardWriter(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  const std::array<unsigned int, 2> channels_;
  const std::array<unsigned long long, 9> algoBitMask_;
  const edm::EDGetTokenT<P2GTAlgoBlockMap> algoBlocksToken_;
  l1t::demo::BoardDataWriter boardDataWriter_;

  std::map<l1t::demo::LinkId, std::vector<ap_uint<64>>> linkData_;
  std::size_t tmuxCounter_;
};

L1GTAlgoBoardWriter::L1GTAlgoBoardWriter(const edm::ParameterSet& config)
    : channels_(config.getParameter<std::array<unsigned int, 2>>("channels")),
      algoBitMask_(config.getParameter<std::array<unsigned long long, 9>>("algoBitMask")),
      algoBlocksToken_(consumes<P2GTAlgoBlockMap>(config.getParameter<edm::InputTag>("algoBlocksTag"))),
      boardDataWriter_(
          l1t::demo::parseFileFormat(config.getParameter<std::string>("patternFormat")),
          config.getParameter<std::string>("outputFilename"),
          config.getParameter<std::string>("outputFileExtension"),
          9,
          2,
          config.getParameter<unsigned int>("maxLines"),
          [](const std::array<unsigned int, 2>& channels) {
            l1t::demo::BoardDataWriter::ChannelMap_t channelMap;
            for (unsigned int channel : channels) {
              channelMap.insert({l1t::demo::LinkId{"Algos", channel}, {l1t::demo::ChannelSpec{2, 0, 0}, {channel}}});
            }
            return channelMap;
          }(channels_)),
      linkData_(),
      tmuxCounter_(0) {}

void L1GTAlgoBoardWriter::analyze(const edm::Event& event, const edm::EventSetup& iSetup) {
  const P2GTAlgoBlockMap& algoBlocks = event.get(algoBlocksToken_);

  auto algoBlockIt = algoBlocks.begin();
  auto algoMaskIt = algoBitMask_.begin();

  for (unsigned int channel : channels_) {
    if (tmuxCounter_ == 0) {
      linkData_[{"Algos", channel}] = std::vector<ap_uint<64>>(18, 0);
    }

    for (std::size_t word = 0; word < 9; word++) {
      ap_uint<64> mask = algoMaskIt != algoBitMask_.end() ? *algoMaskIt++ : ~static_cast<unsigned long long>(0);

      for (std::size_t idx = 0; idx < 64 && algoBlockIt != algoBlocks.end(); idx++, algoBlockIt++) {
        auto& [algoName, algoBlock] = *algoBlockIt;
        linkData_[{"Algos", channel}][word + tmuxCounter_ * 9].set(
            idx, algoBlock.decisionBeforeBxMaskAndPrescale() && mask.bit(idx));
      }
    }
  }

  if (tmuxCounter_ == 1) {
    boardDataWriter_.addEvent(l1t::demo::EventData(linkData_));
  }

  tmuxCounter_ = (tmuxCounter_ + 1) % 2;
}

void L1GTAlgoBoardWriter::endJob() {
  if (tmuxCounter_ == 1) {
    boardDataWriter_.addEvent(l1t::demo::EventData(linkData_));
  }

  boardDataWriter_.flush();
}

void L1GTAlgoBoardWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("outputFilename");
  desc.add<std::string>("outputFileExtension", "txt");
  desc.add<edm::InputTag>("algoBlocksTag");
  desc.add<std::vector<unsigned int>>("channels");
  desc.add<std::vector<unsigned long long>>("algoBitMask",
                                            {0xffffffffffffffffull,
                                             0xffffffffffffffffull,
                                             0xffffffffffffffffull,
                                             0xffffffffffffffffull,
                                             0xffffffffffffffffull,
                                             0xffffffffffffffffull,
                                             0xffffffffffffffffull,
                                             0xffffffffffffffffull,
                                             0xffffffffffffffffull});
  desc.add<unsigned int>("maxLines", 1024);
  desc.add<std::string>("patternFormat", "EMPv2");

  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(L1GTAlgoBoardWriter);
