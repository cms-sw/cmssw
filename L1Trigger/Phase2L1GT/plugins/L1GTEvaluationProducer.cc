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
#include "L1GTChannelMapping.h"

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
  const std::array<std::tuple<const char *, std::size_t, std::size_t>, 27> outputChannelDef_;
  l1t::demo::BoardDataWriter outputBoardDataWriter_;
};

L1GTEvaluationProducer::L1GTEvaluationProducer(const edm::ParameterSet &config)
    : randomGenerator_(config.exists("random_seed") ? config.getParameter<unsigned int>("random_seed")
                                                    : std::random_device()()),
      inputBoardDataWriter_(
          l1t::demo::parseFileFormat(config.getParameter<std::string>("patternFormat")),
          config.getParameter<std::string>("inputFilename"),
          config.getParameter<std::string>("inputFileExtension"),
          9,
          1,
          config.getParameter<unsigned int>("maxFrames"),
          config.getParameter<std::string>("platform") == "VU13P" ? INPUT_CHANNEL_MAP_VU13P : INPUT_CHANNEL_MAP_VU9P),
      outputChannelDef_(config.getParameter<std::string>("platform") == "VU13P" ? OUTPUT_CHANNELS_VU13P
                                                                                : OUTPUT_CHANNELS_VU9P),
      outputBoardDataWriter_(l1t::demo::parseFileFormat(config.getParameter<std::string>("patternFormat")),
                             config.getParameter<std::string>("outputFilename"),
                             config.getParameter<std::string>("outputFileExtension"),
                             9,
                             1,
                             config.getParameter<unsigned int>("maxFrames"),
                             [&]() {
                               demo::BoardDataWriter::ChannelMap_t channelMap;
                               for (const auto &[name, start, end] : outputChannelDef_) {
                                 for (std::size_t i = start; i < end; i++) {
                                   channelMap.insert({{name, i - start}, {{1, 0}, {i}}});
                                 }
                               }
                               return channelMap;
                             }()) {
  produces<P2GTCandidateCollection>("GCTNonIsoEg");
  produces<P2GTCandidateCollection>("GCTIsoEg");
  produces<P2GTCandidateCollection>("GCTJets");
  produces<P2GTCandidateCollection>("GCTTaus");
  produces<P2GTCandidateCollection>("GCTHtSum");
  produces<P2GTCandidateCollection>("GCTEtSum");
  produces<P2GTCandidateCollection>("GMTSaPromptMuons");
  produces<P2GTCandidateCollection>("GMTSaDisplacedMuons");
  produces<P2GTCandidateCollection>("GMTTkMuons");
  produces<P2GTCandidateCollection>("GMTTopo");
  produces<P2GTCandidateCollection>("GTTPromptJets");
  produces<P2GTCandidateCollection>("GTTDisplacedJets");
  produces<P2GTCandidateCollection>("GTTPhiCandidates");
  produces<P2GTCandidateCollection>("GTTRhoCandidates");
  produces<P2GTCandidateCollection>("GTTBsCandidates");
  produces<P2GTCandidateCollection>("GTTHadronicTaus");
  produces<P2GTCandidateCollection>("GTTPromptTracks");
  produces<P2GTCandidateCollection>("GTTDisplacedTracks");
  produces<P2GTCandidateCollection>("GTTPrimaryVert");
  produces<P2GTCandidateCollection>("GTTPromptHtSum");
  produces<P2GTCandidateCollection>("GTTDisplacedHtSum");
  produces<P2GTCandidateCollection>("GTTEtSum");
  produces<P2GTCandidateCollection>("CL2JetsSC4");
  produces<P2GTCandidateCollection>("CL2JetsSC8");
  produces<P2GTCandidateCollection>("CL2Taus");
  produces<P2GTCandidateCollection>("CL2Electrons");
  produces<P2GTCandidateCollection>("CL2Photons");
  produces<P2GTCandidateCollection>("CL2HtSum");
  produces<P2GTCandidateCollection>("CL2EtSum");
}

void L1GTEvaluationProducer::fillDescriptions(edm::ConfigurationDescriptions &description) {
  edm::ParameterSetDescription desc;
  desc.addOptional<unsigned int>("random_seed");
  desc.add<unsigned int>("maxFrames", 1024);
  desc.add<std::string>("inputFilename");
  desc.add<std::string>("inputFileExtension", "txt");
  desc.add<std::string>("outputFilename");
  desc.add<std::string>("outputFileExtension", "txt");
  desc.add<std::string>("patternFormat", "EMPv2");
  desc.ifValue(edm::ParameterDescription<std::string>("platform", "VU9P", true),
               edm::allowedValues<std::string>("VU9P", "VU13P"));
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

  for (const auto &[name, start, end] : outputChannelDef_) {
    std::vector<ap_uint<64>> data = vpack(outputObjects.at(name));
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
