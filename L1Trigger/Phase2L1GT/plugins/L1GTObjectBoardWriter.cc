/**
 * BoardDataWriter for input/output patterns of upstream objects.
 **/

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "L1Trigger/DemonstratorTools/interface/BoardDataWriter.h"
#include "L1Trigger/DemonstratorTools/interface/utilities.h"
#include "L1Trigger/DemonstratorTools/interface/codecs/etsums.h"
#include "L1Trigger/DemonstratorTools/interface/codecs/htsums.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

#include "DataFormats/L1Trigger/interface/TkJetWord.h"
#include "DataFormats/L1Trigger/interface/VertexWord.h"

#include "DataFormats/L1TMuonPhase2/interface/SAMuon.h"
#include "DataFormats/L1TMuonPhase2/interface/TrackerMuon.h"

#include "DataFormats/L1TParticleFlow/interface/PFJet.h"
#include "DataFormats/L1TCorrelator/interface/TkEmFwd.h"
#include "DataFormats/L1TCorrelator/interface/TkEm.h"
#include "DataFormats/L1TCorrelator/interface/TkElectronFwd.h"
#include "DataFormats/L1TCorrelator/interface/TkElectron.h"
#include "DataFormats/L1TParticleFlow/interface/PFTau.h"

#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"

#include "L1GTChannelMapping.h"

#include <vector>
#include <algorithm>
#include <string>
#include <type_traits>
#include <optional>

using namespace l1t;

class L1GTObjectBoardWriter : public edm::one::EDAnalyzer<> {
public:
  explicit L1GTObjectBoardWriter(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  enum BufferType { INPUT, OUTPUT };

  const BufferType bufferFileType_;
  unsigned int eventCounter_;
  unsigned int maxEvents_;
  const GTOutputChannelMap_t outputChannelDef_;
  demo::BoardDataWriter boardDataWriter_;

  // From upstream
  const edm::EDGetTokenT<TkJetWordCollection> gttPromptJetToken_;
  const edm::EDGetTokenT<TkJetWordCollection> gttDisplacedJetToken_;
  const edm::EDGetTokenT<std::vector<EtSum>> gttPromptHtSumToken_;
  const edm::EDGetTokenT<std::vector<EtSum>> gttDisplacedHtSumToken_;
  const edm::EDGetTokenT<std::vector<EtSum>> gttEtSumToken_;
  const edm::EDGetTokenT<VertexWordCollection> gttPrimaryVertexToken_;

  const edm::EDGetTokenT<SAMuonCollection> gmtSaPromptMuonToken_;
  const edm::EDGetTokenT<SAMuonCollection> gmtSaDisplacedMuonToken_;
  const edm::EDGetTokenT<TrackerMuonCollection> gmtTkMuonToken_;

  const edm::EDGetTokenT<PFJetCollection> cl2JetSC4Token_;
  const edm::EDGetTokenT<PFJetCollection> cl2JetSC8Token_;
  const edm::EDGetTokenT<TkEmCollection> cl2PhotonToken_;
  const edm::EDGetTokenT<TkElectronCollection> cl2ElectronToken_;
  const edm::EDGetTokenT<PFTauCollection> cl2TauToken_;
  const edm::EDGetTokenT<std::vector<EtSum>> cl2EtSumToken_;
  const edm::EDGetTokenT<std::vector<EtSum>> cl2HtSumToken_;
};

L1GTObjectBoardWriter::L1GTObjectBoardWriter(const edm::ParameterSet& config)
    : bufferFileType_(config.getParameter<std::string>("bufferFileType") == "input" ? INPUT : OUTPUT),
      eventCounter_(0),
      maxEvents_(config.getParameter<unsigned int>("maxEvents")),
      outputChannelDef_(config.getParameter<std::string>("platform") == "VU13P" ? OUTPUT_CHANNELS_VU13P
                                                                                : OUTPUT_CHANNELS_VU9P),
      boardDataWriter_(demo::parseFileFormat(config.getParameter<std::string>("patternFormat")),
                       config.getParameter<std::string>("filename"),
                       config.getParameter<std::string>("fileExtension"),
                       9,
                       1,
                       config.getParameter<unsigned int>("maxFrames"),
                       [&]() {
                         if (bufferFileType_ == INPUT) {
                           return config.getParameter<std::string>("platform") == "VU13P" ? INPUT_CHANNEL_MAP_VU13P
                                                                                          : INPUT_CHANNEL_MAP_VU9P;
                         } else {
                           demo::BoardDataWriter::ChannelMap_t channelMap;
                           for (const auto& [name, start, end] : outputChannelDef_) {
                             for (std::size_t i = start; i < end; i++) {
                               channelMap.insert({{name, i - start}, {{1, 0}, {i}}});
                             }
                           }
                           return channelMap;
                         }
                       }()),
      gttPromptJetToken_(consumes<TkJetWordCollection>(config.getParameter<edm::InputTag>("GTTPromptJets"))),
      gttDisplacedJetToken_(consumes<TkJetWordCollection>(config.getParameter<edm::InputTag>("GTTDisplacedJets"))),
      gttPromptHtSumToken_(consumes<std::vector<EtSum>>(config.getParameter<edm::InputTag>("GTTPromptHtSum"))),
      gttDisplacedHtSumToken_(consumes<std::vector<EtSum>>(config.getParameter<edm::InputTag>("GTTDisplacedHtSum"))),
      gttEtSumToken_(consumes<std::vector<EtSum>>(config.getParameter<edm::InputTag>("GTTEtSum"))),
      gttPrimaryVertexToken_(consumes<VertexWordCollection>(config.getParameter<edm::InputTag>("GTTPrimaryVert"))),
      gmtSaPromptMuonToken_(consumes<SAMuonCollection>(config.getParameter<edm::InputTag>("GMTSaPromptMuons"))),
      gmtSaDisplacedMuonToken_(consumes<SAMuonCollection>(config.getParameter<edm::InputTag>("GMTSaDisplacedMuons"))),
      gmtTkMuonToken_(consumes<TrackerMuonCollection>(config.getParameter<edm::InputTag>("GMTTkMuons"))),
      cl2JetSC4Token_(consumes<PFJetCollection>(config.getParameter<edm::InputTag>("CL2JetsSC4"))),
      cl2JetSC8Token_(consumes<PFJetCollection>(config.getParameter<edm::InputTag>("CL2JetsSC8"))),
      cl2PhotonToken_(consumes<TkEmCollection>(config.getParameter<edm::InputTag>("CL2Photons"))),
      cl2ElectronToken_(consumes<TkElectronCollection>(config.getParameter<edm::InputTag>("CL2Electrons"))),
      cl2TauToken_(consumes<PFTauCollection>(config.getParameter<edm::InputTag>("CL2Taus"))),
      cl2EtSumToken_(consumes<std::vector<EtSum>>(config.getParameter<edm::InputTag>("CL2EtSum"))),
      cl2HtSumToken_(consumes<std::vector<EtSum>>(config.getParameter<edm::InputTag>("CL2HtSum"))) {}

template <typename T, P2GTCandidate::ObjectType type = P2GTCandidate::Undefined>
static std::vector<ap_uint<64>> packCollection(const std::vector<T>& collection) {
  std::vector<ap_uint<64>> packed;
  std::optional<ap_uint<64>> next_packed;

  for (std::size_t idx = 0; idx < collection.size() && idx < 12; idx++) {
    const T& obj = collection[idx];
    if constexpr (std::is_same_v<T, TkJetWord>) {
      ap_uint<128> word = obj.tkJetWord();
      packed.emplace_back(word(63, 0));
      packed.emplace_back(word(127, 64));
    } else if constexpr (std::is_same_v<T, EtSum>) {
      if constexpr (type == P2GTCandidate::GTTEtSum) {
        packed.emplace_back(l1t::demo::codecs::encodeEtSum(obj));
      } else if constexpr (type == P2GTCandidate::GTTPromptHtSum || type == P2GTCandidate::GTTDisplacedHtSum) {
        packed.emplace_back(l1t::demo::codecs::encodeHtSum(obj));
      } else if constexpr (type == P2GTCandidate::CL2EtSum) {
        l1gt::Sum sum{true /* valid */, obj.pt(), obj.phi() / l1gt::Scales::ETAPHI_LSB, 0 /* scalar sum */};
        packed.emplace_back(sum.pack_ap());
      } else if constexpr (type == P2GTCandidate::CL2HtSum) {
        // Make interfaces great again!
        const EtSum& ht = collection[0];
        const EtSum& mht = collection[1];

        l1gt::Sum sum{true /* valid */, mht.pt(), mht.phi() / l1gt::Scales::ETAPHI_LSB, ht.pt()};
        packed.emplace_back(sum.pack_ap());
      }
      break;
    } else if constexpr (std::is_same_v<T, VertexWord>) {
      packed.emplace_back(obj.vertexWord());
    } else if constexpr (std::is_same_v<T, SAMuon>) {
      packed.emplace_back(obj.word());
    } else if constexpr (std::is_same_v<T, TrackerMuon>) {
      std::array<uint64_t, 2> word = obj.word();
      if (next_packed.has_value()) {
        packed.emplace_back(word[1] << 32 | next_packed.value());
        next_packed.reset();
      } else {
        next_packed = word[1];
      }

      packed.emplace_back(word[0]);
    } else if constexpr (std::is_same_v<T, PFJet>) {
      packed.emplace_back(obj.encodedJet()[0]);
      packed.emplace_back(obj.encodedJet()[1]);
    } else if constexpr (std::is_same_v<T, TkEm> || std::is_same_v<T, TkElectron>) {
      ap_uint<96> word = obj.template egBinaryWord<96>();
      if (next_packed.has_value()) {
        packed.emplace_back(word(95, 64) << 32 | next_packed.value());
        next_packed.reset();
      } else {
        next_packed = word(95, 64);
      }

      packed.emplace_back(word(63, 0));
    } else if constexpr (std::is_same_v<T, PFTau>) {
      std::array<uint64_t, 2> word = obj.encodedTau();
      if (next_packed.has_value()) {
        packed.emplace_back(word[1] << 32 | next_packed.value());
        next_packed.reset();
      } else {
        next_packed = word[1];
      }

      packed.emplace_back(word[0]);
    }
  }

  // Filling up remaining words with 0
  if constexpr (std::is_same_v<T, TkJetWord> || std::is_same_v<T, PFJet>) {
    while (packed.size() < 24) {
      packed.emplace_back(0);
    }
  } else if constexpr (std::is_same_v<T, TrackerMuon> || std::is_same_v<T, TkEm> || std::is_same_v<T, TkElectron> ||
                       std::is_same_v<T, PFTau>) {
    while (packed.size() < 18) {
      packed.emplace_back(0);
    }
  } else if constexpr (std::is_same_v<T, SAMuon> || std::is_same_v<T, VertexWord>) {
    while (packed.size() < 12) {
      packed.emplace_back(0);
    }
  } else if constexpr (std::is_same_v<T, EtSum>) {
    if (packed.size() < 1) {
      packed.emplace_back(0);
    }
  }

  return packed;
}

template <typename T>
static std::vector<T> operator+(std::vector<T>&& lhs, std::vector<T>&& rhs) {
  std::vector<T> concat;
  concat.reserve(lhs.size() + rhs.size());
  std::move(lhs.begin(), lhs.end(), std::back_inserter(concat));
  std::move(rhs.begin(), rhs.end(), std::back_inserter(concat));
  return concat;
}

void L1GTObjectBoardWriter::analyze(const edm::Event& event, const edm::EventSetup&) {
  const TkJetWordCollection& gttPromptJets = event.get(gttPromptJetToken_);
  const TkJetWordCollection& gttDisplacedJets = event.get(gttDisplacedJetToken_);
  const std::vector<EtSum>& gttPromptHtSum = event.get(gttPromptHtSumToken_);
  const std::vector<EtSum>& gttDisplacedHtSum = event.get(gttDisplacedHtSumToken_);
  const std::vector<EtSum>& gttEtSum = event.get(gttEtSumToken_);
  const VertexWordCollection& gttPrimaryVertices = event.get(gttPrimaryVertexToken_);
  const SAMuonCollection& gmtSaPromptMuons = event.get(gmtSaPromptMuonToken_);
  const SAMuonCollection& gmtSaDisplacedMuons = event.get(gmtSaDisplacedMuonToken_);
  const TrackerMuonCollection& gmtTkMuons = event.get(gmtTkMuonToken_);
  const PFJetCollection& cl2JetsSC4 = event.get(cl2JetSC4Token_);
  const PFJetCollection& cl2JetsSC8 = event.get(cl2JetSC8Token_);
  const TkEmCollection& cl2Photons = event.get(cl2PhotonToken_);
  const TkElectronCollection& cl2Electrons = event.get(cl2ElectronToken_);
  const PFTauCollection& cl2Taus = event.get(cl2TauToken_);
  const std::vector<EtSum>& cl2EtSum = event.get(cl2EtSumToken_);
  const std::vector<EtSum>& cl2HtSum = event.get(cl2HtSumToken_);

  if (bufferFileType_ == INPUT) {
    boardDataWriter_.addEvent(
        demo::EventData{{{{"GTT", 1},
                          packCollection(gttPromptJets) + packCollection(gttDisplacedJets) +
                              packCollection<EtSum, P2GTCandidate::GTTPromptHtSum>(gttPromptHtSum) +
                              packCollection<EtSum, P2GTCandidate::GTTDisplacedHtSum>(gttDisplacedHtSum) +
                              packCollection<EtSum, P2GTCandidate::GTTEtSum>(gttEtSum)},
                         {{"GTT", 2}, std::vector<ap_uint<64>>(18, 0)},
                         {{"CL2", 1},
                          packCollection(cl2JetsSC4) + packCollection<EtSum, P2GTCandidate::CL2HtSum>(cl2HtSum) +
                              packCollection<EtSum, P2GTCandidate::CL2EtSum>(cl2EtSum) + packCollection(cl2JetsSC8)},
                         {{"CL2", 2}, packCollection(cl2Taus)},
                         {{"GCT", 1}, std::vector<ap_uint<64>>(50, 0)},
                         {{"GMT", 1},
                          packCollection(gmtSaPromptMuons) + packCollection(gmtSaDisplacedMuons) +
                              packCollection(gmtTkMuons) + std::vector<ap_uint<64>>(12, 0)},
                         {{"CL2", 3}, packCollection(cl2Electrons) + packCollection(cl2Photons)},
                         {{"GTT", 3}, std::vector<ap_uint<64>>(38, 0)},
                         {{"GTT", 4}, std::vector<ap_uint<64>>(36, 0) + packCollection(gttPrimaryVertices)}}});
  } else {
    std::map<demo::LinkId, std::vector<ap_uint<64>>> eventData;

    for (const auto& [name, start, end] : outputChannelDef_) {
      std::vector<ap_uint<64>> data;

      if (std::string("GTTPromptJets") == name) {
        data = packCollection(gttPromptJets);
      } else if (std::string("GTTDisplacedJets") == name) {
        data = packCollection(gttDisplacedJets);
      } else if (std::string("GTTPromptHtSum") == name) {
        data = packCollection<EtSum, P2GTCandidate::GTTPromptHtSum>(gttPromptHtSum);
      } else if (std::string("GTTDisplacedHtSum") == name) {
        data = packCollection<EtSum, P2GTCandidate::GTTDisplacedHtSum>(gttDisplacedHtSum);
      } else if (std::string("GTTEtSum") == name) {
        data = packCollection<EtSum, P2GTCandidate::GTTEtSum>(gttEtSum);
      } else if (std::string("GTTPrimaryVert") == name) {
        data = packCollection(gttPrimaryVertices);
      } else if (std::string("GMTSaPromptMuons") == name) {
        data = packCollection(gmtSaPromptMuons);
      } else if (std::string("GMTSaDisplacedMuons") == name) {
        data = packCollection(gmtSaDisplacedMuons);
      } else if (std::string("GMTTkMuons") == name) {
        data = packCollection(gmtTkMuons);
      } else if (std::string("CL2JetsSC4") == name) {
        data = packCollection(cl2JetsSC4);
      } else if (std::string("CL2JetsSC8") == name) {
        data = packCollection(cl2JetsSC8);
      } else if (std::string("CL2Photons") == name) {
        data = packCollection(cl2Photons);
      } else if (std::string("CL2Electrons") == name) {
        data = packCollection(cl2Electrons);
      } else if (std::string("CL2Taus") == name) {
        data = packCollection(cl2Taus);
      } else if (std::string("CL2EtSum") == name) {
        data = packCollection<EtSum, P2GTCandidate::CL2EtSum>(cl2EtSum);
      } else if (std::string("CL2HtSum") == name) {
        data = packCollection<EtSum, P2GTCandidate::CL2HtSum>(cl2HtSum);
      }

      std::size_t numChannels = end - start;
      for (std::size_t i = start; i < end; i++) {
        for (std::size_t j = i - start; j < data.size(); j += numChannels) {
          eventData[{name, i - start}].push_back(data[j]);
        }

        while (eventData[{name, i - start}].size() < 9) {
          eventData[{name, i - start}].push_back(0);
        }
      }
    }

    boardDataWriter_.addEvent(eventData);
  }

  eventCounter_++;

  if (maxEvents_ != 0 && eventCounter_ == maxEvents_) {
    boardDataWriter_.flush();
    eventCounter_ = 0;
  }
}

void L1GTObjectBoardWriter::endJob() { boardDataWriter_.flush(); }

void L1GTObjectBoardWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::string>("filename");
  desc.add<std::string>("fileExtension", "txt");
  desc.add<unsigned int>("maxFrames", 1024);
  desc.add<unsigned int>("maxEvents", 0);
  desc.add<std::string>("patternFormat", "EMPv2");
  desc.ifValue(edm::ParameterDescription<std::string>("platform", "VU13P", true),
               edm::allowedValues<std::string>("VU9P", "VU13P"));
  desc.ifValue(edm::ParameterDescription<std::string>("bufferFileType", "input", true),
               edm::allowedValues<std::string>("input", "output"));

  desc.add<edm::InputTag>("GTTPromptJets");
  desc.add<edm::InputTag>("GTTDisplacedJets");
  desc.add<edm::InputTag>("GTTPromptHtSum");
  desc.add<edm::InputTag>("GTTDisplacedHtSum");
  desc.add<edm::InputTag>("GTTEtSum");
  desc.add<edm::InputTag>("GTTPrimaryVert");

  desc.add<edm::InputTag>("GMTSaPromptMuons");
  desc.add<edm::InputTag>("GMTSaDisplacedMuons");
  desc.add<edm::InputTag>("GMTTkMuons");

  desc.add<edm::InputTag>("CL2JetsSC4");
  desc.add<edm::InputTag>("CL2JetsSC8");
  desc.add<edm::InputTag>("CL2Photons");
  desc.add<edm::InputTag>("CL2Electrons");
  desc.add<edm::InputTag>("CL2Taus");
  desc.add<edm::InputTag>("CL2EtSum");
  desc.add<edm::InputTag>("CL2HtSum");

  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(L1GTObjectBoardWriter);
