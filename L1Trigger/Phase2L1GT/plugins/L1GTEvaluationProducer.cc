/**  Produces Pseudo-random upstream objects converts them into P2GTCandidates and 
 *   writes them into buffer files. Intended to more thoroughly test the whole
 *   phase space of Phase-2 GT inputs.
 * */

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/View.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDProducer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "L1Trigger/DemonstratorTools/interface/BoardDataWriter.h"
#include "L1Trigger/DemonstratorTools/interface/utilities.h"

#include "L1GTEvaluationInterface.h"

#include <vector>
#include <array>
#include <string>
#include <unordered_map>
#include <fstream>
#include <limits>

#include <optional>
#include <random>

using namespace l1t;

class L1GTEvaluationProducer : public edm::one::EDProducer<> {
public:
  explicit L1GTEvaluationProducer(const edm::ParameterSet &);
  ~L1GTEvaluationProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions &);

private:
  void produce(edm::Event &, const edm::EventSetup &) override;
  int nextValue(int mean = 1000, bool sign = false, int max = std::numeric_limits<int>::max());

  int nextPt() {
    return std::max<int>(0,
                         nextValue(300, false, (1 << P2GTCandidate::hwPT_t::width) - 1) +
                             std::normal_distribution<double>(0, 500)(randomGenerator_));
  }
  int nextEta() {
    return std::uniform_int_distribution<int>(-(1 << (P2GTCandidate::hwEta_t::width - 1)),
                                              (1 << (P2GTCandidate::hwEta_t::width - 1)) - 1)(randomGenerator_);
  }
  int nextPhi() {
    return std::uniform_int_distribution<int>(-(1 << (P2GTCandidate::hwPhi_t::width - 1)),
                                              (1 << (P2GTCandidate::hwPhi_t::width - 1)) - 1)(randomGenerator_);
  }

  void writeInputPatterns(
      const std::unordered_map<std::string, std::vector<std::unique_ptr<l1t::L1TGT_BaseInterface>>> &inputObjects);

  void writeOutputPatterns(
      const std::unordered_map<std::string, std::vector<std::unique_ptr<l1t::L1TGT_BaseInterface>>> &inputObjects);

  void endJob() override;

  std::mt19937 randomGenerator_;
  l1t::demo::BoardDataWriter inputBoardDataWriter_;
  std::unordered_map<std::string, std::size_t> numChannels_;
  l1t::demo::BoardDataWriter outputBoardDataWriter_;
};

static constexpr std::array<const char *, 29> AVAILABLE_COLLECTIONS{{"GTTPromptJets",
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
                                                                     "GTTPromptTracks",
                                                                     "GTTDisplacedTracks",
                                                                     "GTTPrimaryVert"}};

template <typename T1, typename T2>
static std::vector<T1> vconvert(std::vector<T2> ivec) {
  return std::vector<T1>(ivec.begin(), ivec.end());
}

L1GTEvaluationProducer::L1GTEvaluationProducer(const edm::ParameterSet &config)
    : randomGenerator_(config.exists("random_seed") ? config.getUntrackedParameter<unsigned int>("random_seed")
                                                    : std::random_device()()),
      inputBoardDataWriter_(
          l1t::demo::parseFileFormat(config.getUntrackedParameter<std::string>("patternFormat")),
          config.getUntrackedParameter<std::string>("inputFilename"),
          config.getUntrackedParameter<std::string>("inputFileExtension"),
          9,
          1,
          config.getUntrackedParameter<unsigned int>("maxFrames"),
          [&]() {
            const edm::ParameterSet &iChannels = config.getUntrackedParameterSet("InputChannels");
            demo::BoardDataWriter::ChannelMap_t channelMap;

            channelMap.insert(
                {{"GCT", 1},
                 {{6, 0}, vconvert<std::size_t>(iChannels.getUntrackedParameter<std::vector<unsigned int>>("GCT_1"))}});
            channelMap.insert(
                {{"GMT", 1},
                 {{18, 0},
                  vconvert<std::size_t>(iChannels.getUntrackedParameter<std::vector<unsigned int>>("GMT_1"))}});
            channelMap.insert(
                {{"GTT", 1},
                 {{6, 0}, vconvert<std::size_t>(iChannels.getUntrackedParameter<std::vector<unsigned int>>("GTT_1"))}});
            channelMap.insert(
                {{"GTT", 2},
                 {{6, 0}, vconvert<std::size_t>(iChannels.getUntrackedParameter<std::vector<unsigned int>>("GTT_2"))}});
            channelMap.insert(
                {{"GTT", 3},
                 {{6, 0}, vconvert<std::size_t>(iChannels.getUntrackedParameter<std::vector<unsigned int>>("GTT_3"))}});
            channelMap.insert(
                {{"GTT", 4},
                 {{6, 0}, vconvert<std::size_t>(iChannels.getUntrackedParameter<std::vector<unsigned int>>("GTT_4"))}});
            channelMap.insert(
                {{"CL2", 1},
                 {{6, 0}, vconvert<std::size_t>(iChannels.getUntrackedParameter<std::vector<unsigned int>>("CL2_1"))}});
            channelMap.insert(
                {{"CL2", 2},
                 {{6, 0}, vconvert<std::size_t>(iChannels.getUntrackedParameter<std::vector<unsigned int>>("CL2_2"))}});
            channelMap.insert(
                {{"CL2", 3},
                 {{6, 0}, vconvert<std::size_t>(iChannels.getUntrackedParameter<std::vector<unsigned int>>("CL2_3"))}});

            return channelMap;
          }()),
      numChannels_(),
      outputBoardDataWriter_(l1t::demo::parseFileFormat(config.getUntrackedParameter<std::string>("patternFormat")),
                             config.getUntrackedParameter<std::string>("outputFilename"),
                             config.getUntrackedParameter<std::string>("outputFileExtension"),
                             9,
                             1,
                             config.getUntrackedParameter<unsigned int>("maxFrames"),
                             [&]() {
                               const edm::ParameterSet &oChannels = config.getUntrackedParameterSet("OutputChannels");
                               demo::BoardDataWriter::ChannelMap_t channelMap;
                               for (const char *name : AVAILABLE_COLLECTIONS) {
                                 if (oChannels.exists(name)) {
                                   std::vector<unsigned int> channels =
                                       oChannels.getUntrackedParameter<std::vector<unsigned int>>(name);
                                   for (std::size_t i = 0; i < channels.size(); i++) {
                                     channelMap.insert({{name, i}, {{1, 0}, {channels.at(i)}}});
                                   }

                                   numChannels_.insert({name, channels.size()});
                                 } else {
                                   numChannels_.insert({name, 0});
                                 }
                               }
                               return channelMap;
                             }()) {
  for (const char *name : AVAILABLE_COLLECTIONS) {
    produces<P2GTCandidateCollection>(name);
  }
}

void L1GTEvaluationProducer::fillDescriptions(edm::ConfigurationDescriptions &description) {
  edm::ParameterSetDescription desc;
  desc.addOptionalUntracked<unsigned int>("random_seed");
  desc.addUntracked<unsigned int>("maxFrames", 1024);
  desc.addUntracked<std::string>("inputFilename");
  desc.addUntracked<std::string>("inputFileExtension", "txt");
  desc.addUntracked<std::string>("outputFilename");
  desc.addUntracked<std::string>("outputFileExtension", "txt");
  desc.addUntracked<std::string>("patternFormat", "EMPv2");

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

  desc.addUntracked<edm::ParameterSetDescription>("InputChannels", inputChannelDesc);

  edm::ParameterSetDescription outputChannelDesc;
  for (const char *name : AVAILABLE_COLLECTIONS) {
    outputChannelDesc.addOptionalUntracked<std::vector<unsigned int>>(name);
  }

  desc.addUntracked<edm::ParameterSetDescription>("OutputChannels", outputChannelDesc);

  description.addWithDefaultLabel(desc);
}

int L1GTEvaluationProducer::nextValue(int mean, bool sign, int max) {
  bool positive = sign ? std::bernoulli_distribution(0.5)(randomGenerator_) : true;

  int result;
  do {
    result = std::poisson_distribution<int>(mean)(randomGenerator_);
  } while (result > max);

  return positive ? result : -result;
}

template <typename... Args>
static std::vector<ap_uint<64>> vpack(const Args &...vobjects) {
  std::vector<ap_uint<64>> vpacked;

  (
      [&vpacked](const std::vector<std::unique_ptr<l1t::L1TGT_BaseInterface>> &objects) {
        std::optional<ap_uint<64>> next_packed;
        for (const auto &object : objects) {
          if (object->packed_width() == 64) {
            const l1t::L1TGT_Interface<64> &interface_obj = dynamic_cast<const l1t::L1TGT_Interface<64> &>(*object);
            vpacked.emplace_back(interface_obj.pack());
          } else if (object->packed_width() == 96) {
            const l1t::L1TGT_Interface<96> &interface_obj = dynamic_cast<const l1t::L1TGT_Interface<96> &>(*object);
            ap_uint<96> packed = interface_obj.pack();
            if (next_packed.has_value()) {
              vpacked.emplace_back(packed(95, 64) << 32 | next_packed.value());
              next_packed.reset();
            } else {
              next_packed = packed(95, 64);
            }

            vpacked.emplace_back(packed(63, 0));

          } else if (object->packed_width() == 128) {
            const l1t::L1TGT_Interface<128> &interface_obj = dynamic_cast<const l1t::L1TGT_Interface<128> &>(*object);
            ap_uint<128> packed = interface_obj.pack();
            vpacked.emplace_back(packed(63, 0));
            vpacked.emplace_back(packed(127, 64));
          }
        }
      }(vobjects),
      ...);

  return vpacked;
}

void L1GTEvaluationProducer::writeInputPatterns(
    const std::unordered_map<std::string, std::vector<std::unique_ptr<l1t::L1TGT_BaseInterface>>> &inputObjects) {
  inputBoardDataWriter_.addEvent(
      l1t::demo::EventData{{{{"GTT", 1},
                             vpack(inputObjects.at("GTTPromptJets"),
                                   inputObjects.at("GTTDisplacedJets"),
                                   inputObjects.at("GTTPromptHtSum"),
                                   inputObjects.at("GTTDisplacedHtSum"),
                                   inputObjects.at("GTTEtSum"))},
                            {{"GTT", 2}, vpack(inputObjects.at("GTTHadronicTaus"))},
                            {{"CL2", 1},
                             vpack(inputObjects.at("CL2JetsSC4"),
                                   inputObjects.at("CL2HtSum"),
                                   inputObjects.at("CL2EtSum"),
                                   inputObjects.at("CL2JetsSC8"))},
                            {{"CL2", 2}, vpack(inputObjects.at("CL2Taus"))},
                            {{"GCT", 1},
                             vpack(inputObjects.at("GCTNonIsoEg"),
                                   inputObjects.at("GCTIsoEg"),
                                   inputObjects.at("GCTJets"),
                                   inputObjects.at("GCTTaus"),
                                   inputObjects.at("GCTEtSum"),
                                   inputObjects.at("GCTHtSum"))},
                            {{"GMT", 1},
                             vpack(inputObjects.at("GMTSaPromptMuons"),
                                   inputObjects.at("GMTSaDisplacedMuons"),
                                   inputObjects.at("GMTTkMuons"),
                                   inputObjects.at("GMTTopo"))},
                            {{"CL2", 3}, vpack(inputObjects.at("CL2Electrons"), inputObjects.at("CL2Photons"))},
                            {{"GTT", 3},
                             vpack(inputObjects.at("GTTPhiCandidates"),
                                   inputObjects.at("GTTRhoCandidates"),
                                   inputObjects.at("GTTBsCandidates"))},
                            {{"GTT", 4},
                             vpack(inputObjects.at("GTTPromptTracks"),
                                   inputObjects.at("GTTDisplacedTracks"),
                                   inputObjects.at("GTTPrimaryVert"))}}});
}

void L1GTEvaluationProducer::writeOutputPatterns(
    const std::unordered_map<std::string, std::vector<std::unique_ptr<l1t::L1TGT_BaseInterface>>> &outputObjects) {
  std::map<demo::LinkId, std::vector<ap_uint<64>>> eventData;

  for (const char *name : AVAILABLE_COLLECTIONS) {
    std::vector<ap_uint<64>> data = vpack(outputObjects.at(name));

    for (std::size_t i = 0; i < numChannels_.at(name); i++) {
      for (std::size_t j = i; j < data.size(); j += numChannels_.at(name)) {
        eventData[{name, i}].push_back(data[j]);
      }

      while (eventData[{name, i}].size() < 9) {
        eventData[{name, i}].push_back(0);
      }
    }
  }

  outputBoardDataWriter_.addEvent(eventData);
}

void L1GTEvaluationProducer::produce(edm::Event &event, const edm::EventSetup &setup) {
  // Generate random input objects
  std::unordered_map<std::string, std::vector<std::unique_ptr<l1t::L1TGT_BaseInterface>>> inputObjects;
  for (std::size_t i = 0; i < 12; ++i) {
    // Global Muon Trigger
    inputObjects["GMTSaPromptMuons"].emplace_back(std::make_unique<l1t::L1TGT_GMT_PromptDisplacedMuon>(
        true, nextPt(), nextEta(), nextPhi(), nextValue(), nextValue(), nextValue(), nextValue()));

    inputObjects["GMTSaDisplacedMuons"].emplace_back(std::make_unique<l1t::L1TGT_GMT_PromptDisplacedMuon>(
        true, nextPt(), nextEta(), nextPhi(), nextValue(), nextValue(), nextValue(), nextValue()));
    inputObjects["GMTTkMuons"].emplace_back(std::make_unique<l1t::L1TGT_GMT_TrackMatchedmuon>(true,
                                                                                              nextPt(),
                                                                                              nextEta(),
                                                                                              nextPhi(),
                                                                                              nextValue(),
                                                                                              nextValue(),
                                                                                              nextValue(),
                                                                                              nextValue(),
                                                                                              nextValue(),
                                                                                              nextValue()));
    inputObjects["GMTTopo"].emplace_back(
        std::make_unique<l1t::L1TGT_GMT_TopoObject>(true, nextPt(), nextEta(), nextPhi(), nextValue(), nextValue()));

    // Global Calorimeter Trigger
    inputObjects["GCTNonIsoEg"].emplace_back(
        std::make_unique<l1t::L1TGT_GCT_EgammaNonIsolated6p6>(true, nextPt(), nextEta(), nextPhi()));
    inputObjects["GCTIsoEg"].emplace_back(
        std::make_unique<l1t::L1TGT_GCT_EgammaIsolated6p6>(true, nextPt(), nextEta(), nextPhi()));
    inputObjects["GCTJets"].emplace_back(std::make_unique<l1t::L1TGT_GCT_jet6p6>(true, nextPt(), nextEta(), nextPhi()));
    inputObjects["GCTTaus"].emplace_back(
        std::make_unique<l1t::L1TGT_GCT_tau6p6>(true, nextPt(), nextEta(), nextPhi(), nextValue()));

    // Global Track Trigger
    inputObjects["GTTPromptTracks"].emplace_back(std::make_unique<l1t::L1TGT_GTT_Track>());
    inputObjects["GTTDisplacedTracks"].emplace_back(std::make_unique<l1t::L1TGT_GTT_Track>());
    inputObjects["GTTPrimaryVert"].emplace_back(
        std::make_unique<l1t::L1TGT_GTT_PrimaryVert>(true, nextPt(), nextEta(), nextPhi(), nextValue(), nextValue()));
    inputObjects["GTTPromptJets"].emplace_back(
        std::make_unique<l1t::L1TGT_GTT_PromptJet>(true, nextPt(), nextEta(), nextPhi(), nextValue(), nextValue()));
    inputObjects["GTTDisplacedJets"].emplace_back(std::make_unique<l1t::L1TGT_GTT_DisplacedJet>(
        true, nextPt(), nextEta(), nextPhi(), nextValue(), nextValue(), nextValue()));
    inputObjects["GTTHadronicTaus"].emplace_back(std::make_unique<l1t::L1TGT_GTT_HadronicTau>(
        true, nextPt(), nextEta(), nextPhi(), nextValue(), nextValue(), nextValue(), nextValue()));
    inputObjects["GTTPhiCandidates"].emplace_back(
        std::make_unique<l1t::L1TGT_GTT_LightMeson>(true, nextPt(), nextEta(), nextPhi(), nextValue()));
    inputObjects["GTTRhoCandidates"].emplace_back(
        std::make_unique<l1t::L1TGT_GTT_LightMeson>(true, nextPt(), nextEta(), nextPhi(), nextValue()));
    inputObjects["GTTBsCandidates"].emplace_back(
        std::make_unique<l1t::L1TGT_GTT_LightMeson>(true, nextPt(), nextEta(), nextPhi(), nextValue()));

    // Correlator Layer-2
    inputObjects["CL2JetsSC4"].emplace_back(
        std::make_unique<l1t::L1TGT_CL2_Jet>(true, nextPt(), nextEta(), nextPhi(), nextValue()));
    inputObjects["CL2JetsSC8"].emplace_back(
        std::make_unique<l1t::L1TGT_CL2_Jet>(true, nextPt(), nextEta(), nextPhi(), nextValue()));
    inputObjects["CL2Electrons"].emplace_back(std::make_unique<l1t::L1TGT_CL2_Electron>(
        true, nextPt(), nextEta(), nextPhi(), nextValue(), nextValue(), nextValue(), nextValue()));
    inputObjects["CL2Photons"].emplace_back(
        std::make_unique<l1t::L1TGT_CL2_Photon>(true, nextPt(), nextEta(), nextPhi(), nextValue(), nextValue()));
    inputObjects["CL2Taus"].emplace_back(std::make_unique<l1t::L1TGT_CL2_Tau>(
        true, nextPt(), nextEta(), nextPhi(), nextValue(), nextValue(), nextValue(), nextValue()));
  }

  inputObjects["CL2HtSum"].emplace_back(
      std::make_unique<l1t::L1TGT_CL2_Sum>(true, nextValue(), nextValue(), nextValue()));
  inputObjects["CL2EtSum"].emplace_back(
      std::make_unique<l1t::L1TGT_CL2_Sum>(true, nextValue(), nextValue(), nextValue()));
  inputObjects["GCTHtSum"].emplace_back(
      std::make_unique<l1t::L1TGT_GCT_Sum2>(true, nextValue(), nextValue(), nextValue()));
  inputObjects["GCTEtSum"].emplace_back(
      std::make_unique<l1t::L1TGT_GCT_Sum2>(true, nextValue(), nextValue(), nextValue()));

  inputObjects["GTTPromptHtSum"].emplace_back(
      std::make_unique<l1t::L1TGT_GTT_Sum>(true, nextValue(), nextValue(), nextValue()));
  inputObjects["GTTDisplacedHtSum"].emplace_back(
      std::make_unique<l1t::L1TGT_GTT_Sum>(true, nextValue(), nextValue(), nextValue()));
  inputObjects["GTTEtSum"].emplace_back(
      std::make_unique<l1t::L1TGT_GTT_Sum>(true, nextValue(), nextValue(), nextValue()));

  // Write them to a pattern file
  writeInputPatterns(inputObjects);
  writeOutputPatterns(inputObjects);

  for (const auto &[key, inputCollection] : inputObjects) {
    std::unique_ptr<P2GTCandidateCollection> gtCollection = std::make_unique<P2GTCandidateCollection>();
    for (const auto &object : inputCollection) {
      gtCollection->emplace_back(object->to_GTObject());
    }

    event.put(std::move(gtCollection), key);
  }
}

void L1GTEvaluationProducer::endJob() {
  inputBoardDataWriter_.flush();
  outputBoardDataWriter_.flush();
}

DEFINE_FWK_MODULE(L1GTEvaluationProducer);
