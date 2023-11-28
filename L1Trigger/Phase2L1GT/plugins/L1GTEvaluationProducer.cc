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

  void endJob() override;

  std::mt19937 randomGenerator_;
  l1t::demo::BoardDataWriter boardDataWriter_;
};

template <typename T, std::size_t low, std::size_t high, std::size_t incr = 1>
static constexpr std::array<T, high - low> arange() {
  std::array<T, high - low> array;
  T value = low;
  for (T &el : array) {
    el = value;
    value += incr;
  }
  return array;
}

template <typename T, std::size_t low, std::size_t high, std::size_t incr = 1>
static std::vector<T> vrange() {
  std::array<T, high - low> arr(arange<T, low, high, incr>());
  return std::vector(std::begin(arr), std::end(arr));
}

static const l1t::demo::BoardDataWriter::ChannelMap_t CHANNEL_MAP_VU9P{
    {{"GTT", 0}, {{6, 0}, vrange<std::size_t, 0, 6>()}},
    {{"GTT", 1}, {{6, 0}, vrange<std::size_t, 6, 12>()}},
    {{"CL2", 0}, {{6, 0}, vrange<std::size_t, 28, 34>()}},
    {{"CL2", 1}, {{6, 0}, vrange<std::size_t, 34, 40>()}},
    {{"GCT", 0}, {{6, 0}, vrange<std::size_t, 54, 60>()}},
    {{"GMT", 0}, {{18, 0}, vrange<std::size_t, 60, 78>()}},
    {{"CL2", 2}, {{6, 0}, vrange<std::size_t, 80, 86>()}},
    {{"GTT", 2}, {{6, 0}, vrange<std::size_t, 104, 110>()}},
    {{"GTT", 3}, {{6, 0}, vrange<std::size_t, 110, 116>()}}};

static const l1t::demo::BoardDataWriter::ChannelMap_t CHANNEL_MAP_VU13P{
    {{"GTT", 0}, {{6, 0}, vrange<std::size_t, 0, 6>()}},
    {{"GTT", 1}, {{6, 0}, vrange<std::size_t, 6, 12>()}},
    {{"GCT", 0}, {{6, 0}, vrange<std::size_t, 24, 30>()}},
    {{"CL2", 0}, {{6, 0}, vrange<std::size_t, 32, 38>()}},
    {{"CL2", 1}, {{6, 0}, vrange<std::size_t, 38, 44>()}},
    {{"GMT", 0}, {{18, 0}, {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 68, 69, 70, 71, 72, 73}}},
    {{"CL2", 2}, {{6, 0}, vrange<std::size_t, 80, 86>()}},
    {{"GTT", 2}, {{6, 0}, vrange<std::size_t, 112, 118>()}},
    {{"GTT", 3}, {{6, 0}, vrange<std::size_t, 118, 124>()}}};

L1GTEvaluationProducer::L1GTEvaluationProducer(const edm::ParameterSet &config)
    : randomGenerator_(config.exists("random_seed") ? config.getParameter<unsigned int>("random_seed")
                                                    : std::random_device()()),
      boardDataWriter_(l1t::demo::parseFileFormat(config.getParameter<std::string>("patternFormat")),
                       config.getParameter<std::string>("outputFilename"),
                       config.getParameter<std::string>("outputFileExtension"),
                       9,
                       1,
                       config.getParameter<unsigned int>("maxLines"),
                       config.getParameter<std::string>("platform") == "VU13P" ? CHANNEL_MAP_VU13P : CHANNEL_MAP_VU9P) {
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
  produces<P2GTCandidateCollection>("GTTPrimaryVert");
  produces<P2GTCandidateCollection>("GTTPromptHtSum");
  produces<P2GTCandidateCollection>("GTTDisplacedHtSum");
  produces<P2GTCandidateCollection>("GTTEtSum");
  produces<P2GTCandidateCollection>("CL2Jets");
  produces<P2GTCandidateCollection>("CL2Taus");
  produces<P2GTCandidateCollection>("CL2Electrons");
  produces<P2GTCandidateCollection>("CL2Photons");
  produces<P2GTCandidateCollection>("CL2HtSum");
  produces<P2GTCandidateCollection>("CL2EtSum");
}

void L1GTEvaluationProducer::fillDescriptions(edm::ConfigurationDescriptions &description) {
  edm::ParameterSetDescription desc;
  desc.addOptional<unsigned int>("random_seed");
  desc.add<unsigned int>("maxLines", 1024);
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
  boardDataWriter_.addEvent(l1t::demo::EventData{
      {{{"GTT", 0},
        vpack(inputObjects.at("GTTPromptJets"),
              inputObjects.at("GTTDisplacedJets"),
              inputObjects.at("GTTPromptHtSum"),
              inputObjects.at("GTTDisplacedHtSum"),
              inputObjects.at("GTTEtSum"))},
       {{"GTT", 1}, vpack(inputObjects.at("GTTHadronicTaus"))},
       {{"CL2", 0}, vpack(inputObjects.at("CL2Jets"), inputObjects.at("CL2HtSum"), inputObjects.at("CL2EtSum"))},
       {{"CL2", 1}, vpack(inputObjects.at("CL2Taus"))},
       {{"GCT", 0},
        vpack(inputObjects.at("GCTNonIsoEg"),
              inputObjects.at("GCTIsoEg"),
              inputObjects.at("GCTJets"),
              inputObjects.at("GCTTaus"),
              inputObjects.at("GCTHtSum"),
              inputObjects.at("GCTEtSum"))},
       {{"GMT", 0},
        vpack(inputObjects.at("GMTSaPromptMuons"),
              inputObjects.at("GMTSaDisplacedMuons"),
              inputObjects.at("GMTTkMuons"),
              inputObjects.at("GMTTopo"))},
       {{"CL2", 2}, vpack(inputObjects.at("CL2Electrons"), inputObjects.at("CL2Photons"))},
       {{"GTT", 2},
        vpack(inputObjects.at("GTTPhiCandidates"),
              inputObjects.at("GTTRhoCandidates"),
              inputObjects.at("GTTBsCandidates"))},
       {{"GTT", 3}, vpack(inputObjects.at("GTTPrimaryVert"))}}});
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
    inputObjects["CL2Jets"].emplace_back(
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

  for (const auto &[key, inputCollection] : inputObjects) {
    std::unique_ptr<P2GTCandidateCollection> gtCollection = std::make_unique<P2GTCandidateCollection>();
    for (const auto &object : inputCollection) {
      gtCollection->emplace_back(object->to_GTObject());
    }

    event.put(std::move(gtCollection), key);
  }
}

void L1GTEvaluationProducer::endJob() { boardDataWriter_.flush(); }

DEFINE_FWK_MODULE(L1GTEvaluationProducer);
