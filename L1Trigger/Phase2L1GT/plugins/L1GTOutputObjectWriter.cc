#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "DataFormats/Common/interface/Handle.h"

#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/View.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"

#include "L1Trigger/DemonstratorTools/interface/BoardDataWriter.h"
#include "L1Trigger/DemonstratorTools/interface/utilities.h"

#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"

#include "L1GTEvaluationInterface.h"

#include <vector>
#include <array>
#include <tuple>
#include <string>
#include <fstream>
#include <limits>

#include <optional>

using namespace l1t;

static constexpr std::array<std::tuple<const char*, std::size_t, std::size_t>, 27> OUTPUT_CHANNELS_VU9P{
    {{"GTTPromptJets", 2, 6},
     {"GTTDisplacedJets", 6, 10},
     {"GTTPromptHtSum", 10, 11},
     {"GTTDisplacedHtSum", 11, 12},
     {"GTTEtSum", 12, 13},
     {"GTTHadronicTaus", 13, 16},
     {"CL2JetsSC4", 24, 28},
     {"CL2JetsSC8", 28, 32},
     {"CL2Taus", 34, 37},
     {"CL2HtSum", 37, 38},
     {"CL2EtSum", 38, 39},
     {"GCTNonIsoEg", 48, 50},
     {"GCTIsoEg", 50, 52},
     {"GCTJets", 52, 54},
     {"GCTTaus", 54, 56},
     {"GCTHtSum", 56, 57},
     {"GCTEtSum", 57, 58},
     {"GMTSaPromptMuons", 60, 62},
     {"GMTSaDisplacedMuons", 62, 64},
     {"GMTTkMuons", 64, 67},
     {"GMTTopo", 67, 69},
     {"CL2Electrons", 80, 83},
     {"CL2Photons", 83, 86},
     {"GTTPhiCandidates", 104, 107},
     {"GTTRhoCandidates", 107, 110},
     {"GTTBsCandidates", 110, 113},
     {"GTTPrimaryVert", 113, 115}}};

static constexpr std::array<std::tuple<const char*, std::size_t, std::size_t>, 27> OUTPUT_CHANNELS_VU13P{
    {{"GTTPromptJets", 2, 6},
     {"GTTDisplacedJets", 6, 10},
     {"GTTPromptHtSum", 10, 11},
     {"GTTDisplacedHtSum", 11, 12},
     {"GTTEtSum", 12, 13},
     {"GTTHadronicTaus", 13, 16},
     {"GCTNonIsoEg", 26, 28},
     {"GCTIsoEg", 28, 30},
     {"GCTJets", 30, 32},
     {"CL2JetsSC4", 32, 36},
     {"CL2JetsSC8", 36, 40},
     {"CL2Taus", 40, 43},
     {"CL2HtSum", 43, 44},
     {"CL2EtSum", 44, 45},
     {"GMTSaPromptMuons", 68, 70},
     {"GMTSaDisplacedMuons", 70, 72},
     {"GMTTkMuons", 72, 75},
     {"GMTTopo", 75, 77},
     {"CL2Electrons", 80, 83},
     {"CL2Photons", 83, 86},
     {"GCTTaus", 96, 98},
     {"GCTHtSum", 98, 99},
     {"GCTEtSum", 99, 100},
     {"GTTPhiCandidates", 112, 115},
     {"GTTRhoCandidates", 115, 118},
     {"GTTBsCandidates", 118, 121},
     {"GTTPrimaryVert", 121, 123}}};

class L1GTOutputObjectWriter : public edm::one::EDAnalyzer<> {
public:
  explicit L1GTOutputObjectWriter(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  template <typename T>
  std::vector<std::unique_ptr<L1TGT_BaseInterface>> fillCollection(
      const edm::Event& event, const edm::EDGetTokenT<P2GTCandidateCollection>& token) const;
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  const edm::EDGetTokenT<P2GTCandidateCollection> gctNonIsoEgToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gctIsoEgToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gctJetsToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gctTausToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gctHtSumToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gctEtSumToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gmtSaPromptMuonsToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gmtSaDisplacedMuonsToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gmtTkMuonsToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gmtTopoToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gttPromptJetsToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gttDisplacedJetsToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gttPhiCandidatesToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gttRhoCandidatesToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gttBsCandidatesToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gttHadronicTausToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gttPrimaryVertToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gttPromptHtSumToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gttDisplacedHtSumToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> gttEtSumToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> cl2JetsSc4Token_;
  const edm::EDGetTokenT<P2GTCandidateCollection> cl2JetsSc8Token_;
  const edm::EDGetTokenT<P2GTCandidateCollection> cl2TausToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> cl2ElectronsToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> cl2PhotonsToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> cl2HtSumToken_;
  const edm::EDGetTokenT<P2GTCandidateCollection> cl2EtSumToken_;
  const std::array<std::tuple<const char*, std::size_t, std::size_t>, 27> outputChannelDef_;
  l1t::demo::BoardDataWriter boardDataWriter_;
};

L1GTOutputObjectWriter::L1GTOutputObjectWriter(const edm::ParameterSet& config)
    : gctNonIsoEgToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GCTNonIsoEg"))),
      gctIsoEgToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GCTIsoEg"))),
      gctJetsToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GCTJets"))),
      gctTausToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GCTTaus"))),
      gctHtSumToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GCTHtSum"))),
      gctEtSumToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GCTEtSum"))),
      gmtSaPromptMuonsToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GMTSaPromptMuons"))),
      gmtSaDisplacedMuonsToken_(
          consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GMTSaDisplacedMuons"))),
      gmtTkMuonsToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GMTTkMuons"))),
      gmtTopoToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GMTTopo"))),
      gttPromptJetsToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GTTPromptJets"))),
      gttDisplacedJetsToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GTTDisplacedJets"))),
      gttPhiCandidatesToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GTTPhiCandidates"))),
      gttRhoCandidatesToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GTTRhoCandidates"))),
      gttBsCandidatesToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GTTBsCandidates"))),
      gttHadronicTausToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GTTHadronicTaus"))),
      gttPrimaryVertToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GTTPrimaryVert"))),
      gttPromptHtSumToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GTTPromptHtSum"))),
      gttDisplacedHtSumToken_(
          consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GTTDisplacedHtSum"))),
      gttEtSumToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("GTTEtSum"))),
      cl2JetsSc4Token_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("CL2JetsSC4"))),
      cl2JetsSc8Token_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("CL2JetsSC8"))),
      cl2TausToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("CL2Taus"))),
      cl2ElectronsToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("CL2Electrons"))),
      cl2PhotonsToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("CL2Photons"))),
      cl2HtSumToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("CL2HtSum"))),
      cl2EtSumToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("CL2EtSum"))),
      outputChannelDef_(config.getParameter<std::string>("platform") == "VU13P" ? OUTPUT_CHANNELS_VU13P
                                                                                : OUTPUT_CHANNELS_VU9P),
      boardDataWriter_(l1t::demo::parseFileFormat(config.getParameter<std::string>("patternFormat")),
                       config.getParameter<std::string>("outputFilename"),
                       config.getParameter<std::string>("outputFileExtension"),
                       9,
                       1,
                       config.getParameter<unsigned int>("maxLines"),
                       [&]() {
                         demo::BoardDataWriter::ChannelMap_t channelMap;
                         for (const auto& [name, start, end] : outputChannelDef_) {
                           for (std::size_t i = start; i < end; i++) {
                             channelMap.insert({{name, i - start}, {{1, 0}, {i}}});
                           }
                         }
                         return channelMap;
                       }()) {}

void L1GTOutputObjectWriter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("GCTNonIsoEg");
  desc.add<edm::InputTag>("GCTIsoEg");
  desc.add<edm::InputTag>("GCTJets");
  desc.add<edm::InputTag>("GCTTaus");
  desc.add<edm::InputTag>("GCTHtSum");
  desc.add<edm::InputTag>("GCTEtSum");
  desc.add<edm::InputTag>("GMTSaPromptMuons");
  desc.add<edm::InputTag>("GMTSaDisplacedMuons");
  desc.add<edm::InputTag>("GMTTkMuons");
  desc.add<edm::InputTag>("GMTTopo");
  desc.add<edm::InputTag>("GTTPromptJets");
  desc.add<edm::InputTag>("GTTDisplacedJets");
  desc.add<edm::InputTag>("GTTPhiCandidates");
  desc.add<edm::InputTag>("GTTRhoCandidates");
  desc.add<edm::InputTag>("GTTBsCandidates");
  desc.add<edm::InputTag>("GTTHadronicTaus");
  desc.add<edm::InputTag>("GTTPrimaryVert");
  desc.add<edm::InputTag>("GTTPromptHtSum");
  desc.add<edm::InputTag>("GTTDisplacedHtSum");
  desc.add<edm::InputTag>("GTTEtSum");
  desc.add<edm::InputTag>("CL2JetsSC4");
  desc.add<edm::InputTag>("CL2JetsSC8");
  desc.add<edm::InputTag>("CL2Taus");
  desc.add<edm::InputTag>("CL2Electrons");
  desc.add<edm::InputTag>("CL2Photons");
  desc.add<edm::InputTag>("CL2HtSum");
  desc.add<edm::InputTag>("CL2EtSum");

  desc.add<unsigned int>("maxLines", 1024);
  desc.add<std::string>("outputFilename");
  desc.add<std::string>("outputFileExtension", "txt");
  desc.add<std::string>("patternFormat", "EMPv2");
  desc.ifValue(edm::ParameterDescription<std::string>("platform", "VU9P", true),
               edm::allowedValues<std::string>("VU9P", "VU13P"));

  descriptions.addWithDefaultLabel(desc);
}

template <typename T>
std::vector<std::unique_ptr<L1TGT_BaseInterface>> L1GTOutputObjectWriter::fillCollection(
    const edm::Event& event, const edm::EDGetTokenT<P2GTCandidateCollection>& token) const {
  std::vector<std::unique_ptr<L1TGT_BaseInterface>> outputCollection;

  for (const P2GTCandidate& object : event.get(token)) {
    outputCollection.push_back(std::make_unique<T>(T::from_GTObject(object)));
  }

  return outputCollection;
}

template <typename... Args>
static std::vector<ap_uint<64>> vpack(const Args&... vobjects) {
  std::vector<ap_uint<64>> vpacked;

  (
      [&vpacked](const std::vector<std::unique_ptr<l1t::L1TGT_BaseInterface>>& objects) {
        std::optional<ap_uint<64>> next_packed;
        for (const auto& object : objects) {
          if (object->packed_width() == 64) {
            const l1t::L1TGT_Interface<64>& interface_obj = dynamic_cast<const l1t::L1TGT_Interface<64>&>(*object);
            vpacked.emplace_back(interface_obj.pack());
          } else if (object->packed_width() == 96) {
            const l1t::L1TGT_Interface<96>& interface_obj = dynamic_cast<const l1t::L1TGT_Interface<96>&>(*object);
            ap_uint<96> packed = interface_obj.pack();
            if (next_packed.has_value()) {
              vpacked.emplace_back(packed(95, 64) << 32 | next_packed.value());
              next_packed.reset();
            } else {
              next_packed = packed(95, 64);
            }

            vpacked.emplace_back(packed(63, 0));

          } else if (object->packed_width() == 128) {
            const l1t::L1TGT_Interface<128>& interface_obj = dynamic_cast<const l1t::L1TGT_Interface<128>&>(*object);
            ap_uint<128> packed = interface_obj.pack();
            vpacked.emplace_back(packed(63, 0));
            vpacked.emplace_back(packed(127, 64));
          }
        }
      }(vobjects),
      ...);

  return vpacked;
}

void L1GTOutputObjectWriter::analyze(const edm::Event& event, const edm::EventSetup&) {
  std::map<std::string, std::vector<std::unique_ptr<L1TGT_BaseInterface>>> outputObjects;

  outputObjects.emplace("GCTNonIsoEg", std::move(fillCollection<L1TGT_Common3Vector<64>>(event, gctNonIsoEgToken_)));
  outputObjects.emplace("GCTIsoEg", std::move(fillCollection<L1TGT_Common3Vector<64>>(event, gctIsoEgToken_)));
  outputObjects.emplace("GCTJets", std::move(fillCollection<L1TGT_Common3Vector<64>>(event, gctJetsToken_)));
  outputObjects.emplace("GCTTaus", std::move(fillCollection<L1TGT_GCT_tau6p6>(event, gctTausToken_)));
  outputObjects.emplace("GCTHtSum", std::move(fillCollection<L1TGT_CommonSum>(event, gctHtSumToken_)));
  outputObjects.emplace("GCTEtSum", std::move(fillCollection<L1TGT_CommonSum>(event, gctEtSumToken_)));
  outputObjects.emplace("GMTSaPromptMuons",
                        std::move(fillCollection<L1TGT_GMT_PromptDisplacedMuon>(event, gmtSaPromptMuonsToken_)));
  outputObjects.emplace("GMTSaDisplacedMuons",
                        std::move(fillCollection<L1TGT_GMT_PromptDisplacedMuon>(event, gmtSaDisplacedMuonsToken_)));
  outputObjects.emplace("GMTTkMuons", std::move(fillCollection<L1TGT_GMT_TrackMatchedmuon>(event, gmtTkMuonsToken_)));
  outputObjects.emplace("GMTTopo", std::move(fillCollection<L1TGT_GMT_TopoObject>(event, gmtTopoToken_)));
  outputObjects.emplace("GTTPromptJets", std::move(fillCollection<L1TGT_GTT_PromptJet>(event, gttPromptJetsToken_)));
  outputObjects.emplace("GTTDisplacedJets",
                        std::move(fillCollection<L1TGT_GTT_DisplacedJet>(event, gttDisplacedJetsToken_)));
  outputObjects.emplace("GTTPhiCandidates",
                        std::move(fillCollection<L1TGT_GTT_LightMeson>(event, gttPhiCandidatesToken_)));
  outputObjects.emplace("GTTRhoCandidates",
                        std::move(fillCollection<L1TGT_GTT_LightMeson>(event, gttRhoCandidatesToken_)));
  outputObjects.emplace("GTTBsCandidates",
                        std::move(fillCollection<L1TGT_GTT_LightMeson>(event, gttBsCandidatesToken_)));
  outputObjects.emplace("GTTHadronicTaus",
                        std::move(fillCollection<L1TGT_GTT_HadronicTau>(event, gttHadronicTausToken_)));
  outputObjects.emplace("GTTPrimaryVert",
                        std::move(fillCollection<L1TGT_GTT_PrimaryVert>(event, gttPrimaryVertToken_)));
  outputObjects.emplace("GTTPromptHtSum", std::move(fillCollection<L1TGT_CommonSum>(event, gttPromptHtSumToken_)));
  outputObjects.emplace("GTTDisplacedHtSum",
                        std::move(fillCollection<L1TGT_CommonSum>(event, gttDisplacedHtSumToken_)));
  outputObjects.emplace("GTTEtSum", std::move(fillCollection<L1TGT_CommonSum>(event, gttEtSumToken_)));
  outputObjects.emplace("CL2JetsSC4", std::move(fillCollection<L1TGT_CL2_Jet>(event, cl2JetsSc4Token_)));
  outputObjects.emplace("CL2JetsSC8", std::move(fillCollection<L1TGT_CL2_Jet>(event, cl2JetsSc8Token_)));
  outputObjects.emplace("CL2Taus", std::move(fillCollection<L1TGT_CL2_Tau>(event, cl2TausToken_)));
  outputObjects.emplace("CL2Electrons", std::move(fillCollection<L1TGT_CL2_Electron>(event, cl2ElectronsToken_)));
  outputObjects.emplace("CL2Photons", std::move(fillCollection<L1TGT_CL2_Photon>(event, cl2PhotonsToken_)));
  outputObjects.emplace("CL2HtSum", std::move(fillCollection<L1TGT_CommonSum>(event, cl2HtSumToken_)));
  outputObjects.emplace("CL2EtSum", std::move(fillCollection<L1TGT_CommonSum>(event, cl2EtSumToken_)));

  std::map<demo::LinkId, std::vector<ap_uint<64>>> eventData;

  for (const auto& [name, start, end] : outputChannelDef_) {
    std::vector<ap_uint<64>> data = vpack(outputObjects[name]);
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

void L1GTOutputObjectWriter::endJob() { boardDataWriter_.flush(); }

DEFINE_FWK_MODULE(L1GTOutputObjectWriter);
