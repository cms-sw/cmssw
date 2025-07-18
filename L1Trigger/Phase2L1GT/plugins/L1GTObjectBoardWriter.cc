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

#include <vector>
#include <algorithm>
#include <string>
#include <type_traits>
#include <optional>
#include <array>
#include <unordered_map>

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
  std::unordered_map<std::string, std::size_t> numChannels_;
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

template <typename T1, typename T2>
static std::vector<T1> vconvert(std::vector<T2> ivec) {
  return std::vector<T1>(ivec.begin(), ivec.end());
}

static constexpr std::array<const char*, 27> AVAILABLE_COLLECTIONS{{"GTTPromptJets",
                                                                    "GTTDisplacedJets",
                                                                    "GTTPromptHtSum",
                                                                    "GTTDisplacedHtSum",
                                                                    "GTTEtSum",
                                                                    "GTTHadronicTaus",
                                                                    "CL2JetsSC4",
                                                                    "CL2JetsSC8",
                                                                    "CL2Taus",
                                                                    "CL2HtSum",
                                                                    "CL2EtSum",
                                                                    "GCTNonIsoEg",
                                                                    "GCTIsoEg",
                                                                    "GCTJets",
                                                                    "GCTTaus",
                                                                    "GCTHtSum",
                                                                    "GCTEtSum",
                                                                    "GMTSaPromptMuons",
                                                                    "GMTSaDisplacedMuons",
                                                                    "GMTTkMuons",
                                                                    "GMTTopo",
                                                                    "CL2Electrons",
                                                                    "CL2Photons",
                                                                    "GTTPhiCandidates",
                                                                    "GTTRhoCandidates",
                                                                    "GTTBsCandidates",
                                                                    "GTTPrimaryVert"}};

L1GTObjectBoardWriter::L1GTObjectBoardWriter(const edm::ParameterSet& config)
    : bufferFileType_(config.getUntrackedParameter<std::string>("bufferFileType") == "input" ? INPUT : OUTPUT),
      eventCounter_(0),
      maxEvents_(config.getUntrackedParameter<unsigned int>("maxEvents")),
      numChannels_(),
      boardDataWriter_(
          demo::parseFileFormat(config.getUntrackedParameter<std::string>("patternFormat")),
          config.getUntrackedParameter<std::string>("filename"),
          config.getUntrackedParameter<std::string>("fileExtension"),
          9,
          1,
          config.getUntrackedParameter<unsigned int>("maxFrames"),
          [&]() {
            if (bufferFileType_ == INPUT) {
              const edm::ParameterSet& iChannels = config.getUntrackedParameterSet("InputChannels");
              demo::BoardDataWriter::ChannelMap_t channelMap;

              channelMap.insert(
                  {{"GCT", 1},
                   {{6, 0},
                    vconvert<std::size_t>(iChannels.getUntrackedParameter<std::vector<unsigned int>>("GCT_1"))}});
              channelMap.insert(
                  {{"GMT", 1},
                   {{18, 0},
                    vconvert<std::size_t>(iChannels.getUntrackedParameter<std::vector<unsigned int>>("GMT_1"))}});
              channelMap.insert(
                  {{"GTT", 1},
                   {{6, 0},
                    vconvert<std::size_t>(iChannels.getUntrackedParameter<std::vector<unsigned int>>("GTT_1"))}});
              channelMap.insert(
                  {{"GTT", 2},
                   {{6, 0},
                    vconvert<std::size_t>(iChannels.getUntrackedParameter<std::vector<unsigned int>>("GTT_2"))}});
              channelMap.insert(
                  {{"GTT", 3},
                   {{6, 0},
                    vconvert<std::size_t>(iChannels.getUntrackedParameter<std::vector<unsigned int>>("GTT_3"))}});
              channelMap.insert(
                  {{"GTT", 4},
                   {{6, 0},
                    vconvert<std::size_t>(iChannels.getUntrackedParameter<std::vector<unsigned int>>("GTT_4"))}});
              channelMap.insert(
                  {{"CL2", 1},
                   {{6, 0},
                    vconvert<std::size_t>(iChannels.getUntrackedParameter<std::vector<unsigned int>>("CL2_1"))}});
              channelMap.insert(
                  {{"CL2", 2},
                   {{6, 0},
                    vconvert<std::size_t>(iChannels.getUntrackedParameter<std::vector<unsigned int>>("CL2_2"))}});
              channelMap.insert(
                  {{"CL2", 3},
                   {{6, 0},
                    vconvert<std::size_t>(iChannels.getUntrackedParameter<std::vector<unsigned int>>("CL2_3"))}});

              return channelMap;
            } else {
              const edm::ParameterSet& oChannels = config.getUntrackedParameterSet("OutputChannels");
              demo::BoardDataWriter::ChannelMap_t channelMap;
              for (const char* name : AVAILABLE_COLLECTIONS) {
                std::vector<unsigned int> channels = oChannels.getUntrackedParameter<std::vector<unsigned int>>(name);
                for (std::size_t i = 0; i < channels.size(); i++) {
                  channelMap.insert({{name, i}, {{1, 0}, {channels.at(i)}}});
                }
                numChannels_.insert({name, channels.size()});
              }
              return channelMap;
            }
          }()),
      gttPromptJetToken_(consumes<TkJetWordCollection>(config.getUntrackedParameter<edm::InputTag>("GTTPromptJets"))),
      gttDisplacedJetToken_(
          consumes<TkJetWordCollection>(config.getUntrackedParameter<edm::InputTag>("GTTDisplacedJets"))),
      gttPromptHtSumToken_(consumes<std::vector<EtSum>>(config.getUntrackedParameter<edm::InputTag>("GTTPromptHtSum"))),
      gttDisplacedHtSumToken_(
          consumes<std::vector<EtSum>>(config.getUntrackedParameter<edm::InputTag>("GTTDisplacedHtSum"))),
      gttEtSumToken_(consumes<std::vector<EtSum>>(config.getUntrackedParameter<edm::InputTag>("GTTEtSum"))),
      gttPrimaryVertexToken_(
          consumes<VertexWordCollection>(config.getUntrackedParameter<edm::InputTag>("GTTPrimaryVert"))),
      gmtSaPromptMuonToken_(
          consumes<SAMuonCollection>(config.getUntrackedParameter<edm::InputTag>("GMTSaPromptMuons"))),
      gmtSaDisplacedMuonToken_(
          consumes<SAMuonCollection>(config.getUntrackedParameter<edm::InputTag>("GMTSaDisplacedMuons"))),
      gmtTkMuonToken_(consumes<TrackerMuonCollection>(config.getUntrackedParameter<edm::InputTag>("GMTTkMuons"))),
      cl2JetSC4Token_(consumes<PFJetCollection>(config.getUntrackedParameter<edm::InputTag>("CL2JetsSC4"))),
      cl2JetSC8Token_(consumes<PFJetCollection>(config.getUntrackedParameter<edm::InputTag>("CL2JetsSC8"))),
      cl2PhotonToken_(consumes<TkEmCollection>(config.getUntrackedParameter<edm::InputTag>("CL2Photons"))),
      cl2ElectronToken_(consumes<TkElectronCollection>(config.getUntrackedParameter<edm::InputTag>("CL2Electrons"))),
      cl2TauToken_(consumes<PFTauCollection>(config.getUntrackedParameter<edm::InputTag>("CL2Taus"))),
      cl2EtSumToken_(consumes<std::vector<EtSum>>(config.getUntrackedParameter<edm::InputTag>("CL2EtSum"))),
      cl2HtSumToken_(consumes<std::vector<EtSum>>(config.getUntrackedParameter<edm::InputTag>("CL2HtSum"))) {}

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
      if (next_packed) {
        packed.emplace_back(next_packed.value());
        next_packed.reset();
      } else {
        packed.emplace_back(0);
      }
    }
  } else if constexpr (std::is_same_v<T, SAMuon> || std::is_same_v<T, VertexWord>) {
    while (packed.size() < 12) {
      packed.emplace_back(0);
    }
  } else if constexpr (std::is_same_v<T, EtSum>) {
    if (packed.empty()) {
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
                         {{"GTT", 3}, std::vector<ap_uint<64>>(39, 0)},
                         {{"GTT", 4}, std::vector<ap_uint<64>>(36, 0) + packCollection(gttPrimaryVertices)}}});
  } else {
    std::map<demo::LinkId, std::vector<ap_uint<64>>> eventData;

    for (const char* name : AVAILABLE_COLLECTIONS) {
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

      for (std::size_t i = 0; i < numChannels_.at(name); i++) {
        for (std::size_t j = i; j < data.size(); j += numChannels_.at(name)) {
          eventData[{name, i}].push_back(data[j]);
        }

        while (eventData[{name, i}].size() < 9) {
          eventData[{name, i}].push_back(0);
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
  desc.addUntracked<std::string>("filename");
  desc.addUntracked<std::string>("fileExtension", "txt");
  desc.addUntracked<unsigned int>("maxFrames", 1024);
  desc.addUntracked<unsigned int>("maxEvents", 0);
  desc.addUntracked<std::string>("patternFormat", "EMPv2");
  desc.ifValue(edm::ParameterDescription<std::string>("bufferFileType", "input", false),
               edm::allowedValues<std::string>("input", "output"));

  desc.addUntracked<edm::InputTag>("GTTPromptJets");
  desc.addUntracked<edm::InputTag>("GTTDisplacedJets");
  desc.addUntracked<edm::InputTag>("GTTPromptHtSum");
  desc.addUntracked<edm::InputTag>("GTTDisplacedHtSum");
  desc.addUntracked<edm::InputTag>("GTTEtSum");
  desc.addUntracked<edm::InputTag>("GTTPrimaryVert");

  desc.addUntracked<edm::InputTag>("GMTSaPromptMuons");
  desc.addUntracked<edm::InputTag>("GMTSaDisplacedMuons");
  desc.addUntracked<edm::InputTag>("GMTTkMuons");

  desc.addUntracked<edm::InputTag>("CL2JetsSC4");
  desc.addUntracked<edm::InputTag>("CL2JetsSC8");
  desc.addUntracked<edm::InputTag>("CL2Photons");
  desc.addUntracked<edm::InputTag>("CL2Electrons");
  desc.addUntracked<edm::InputTag>("CL2Taus");
  desc.addUntracked<edm::InputTag>("CL2EtSum");
  desc.addUntracked<edm::InputTag>("CL2HtSum");

  edm::ParameterSetDescription inputChannelDesc;
  inputChannelDesc.addUntracked<std::vector<unsigned int>>("GCT_1");
  inputChannelDesc.addUntracked<std::vector<unsigned int>>("GMT_1");
  inputChannelDesc.addUntracked<std::vector<unsigned int>>("GTT_1");
  inputChannelDesc.addUntracked<std::vector<unsigned int>>("GTT_2");
  inputChannelDesc.addUntracked<std::vector<unsigned int>>("GTT_3");
  inputChannelDesc.addUntracked<std::vector<unsigned int>>("GTT_4");
  inputChannelDesc.addUntracked<std::vector<unsigned int>>("CL2_1");
  inputChannelDesc.addUntracked<std::vector<unsigned int>>("CL2_2");
  inputChannelDesc.addUntracked<std::vector<unsigned int>>("CL2_3");

  desc.addOptionalUntracked<edm::ParameterSetDescription>("InputChannels", inputChannelDesc);

  edm::ParameterSetDescription outputChannelDesc;
  for (const char* name : AVAILABLE_COLLECTIONS) {
    outputChannelDesc.addUntracked<std::vector<unsigned int>>(name);
  }

  desc.addOptionalUntracked<edm::ParameterSetDescription>("OutputChannels", outputChannelDesc);

  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(L1GTObjectBoardWriter);
