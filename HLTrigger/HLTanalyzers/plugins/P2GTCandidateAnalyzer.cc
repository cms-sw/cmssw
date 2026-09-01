// P2GTCandidateAnalyzer.cc
//
// EDAnalyzer that replicates the L1 GT algo-block / object-type matching
// pattern used in HLT filters (e.g. HLTMuonTrkL1TkMuFilter), but with the
// algo names and the target l1t::P2GTCandidate::ObjectType configurable via
// the python cfg, and prints out the matched candidates' data members.
//
// NOTE: l1t::P2GTCandidate stores most hardware quantities as Optional<int>
// internally; the hw*() getters throw std::invalid_argument when a field
// isn't set for a given ObjectType (e.g. a jet has no hwD0, a track has no
// hwQualityScore). printCandidate() below therefore tries each field and
// silently skips the ones that aren't populated for this candidate.

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1Trigger/interface/P2GTAlgoBlock.h"
#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"

#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

class P2GTCandidateAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit P2GTCandidateAnalyzer(const edm::ParameterSet&);
  ~P2GTCandidateAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  static l1t::P2GTCandidate::ObjectType objectTypeFromString(const std::string& name);
  static std::string printCandidate(const l1t::P2GTCandidateRef& cand);

  const edm::EDGetTokenT<l1t::P2GTAlgoBlockMap> m_algoBlockToken;
  const std::vector<std::string> m_l1GTAlgoNames;
  const l1t::P2GTCandidate::ObjectType m_objectType;
  const std::string m_objectTypeName;  // kept around for logging
};

P2GTCandidateAnalyzer::P2GTCandidateAnalyzer(const edm::ParameterSet& iConfig)
    : m_algoBlockToken(consumes<l1t::P2GTAlgoBlockMap>(iConfig.getParameter<edm::InputTag>("l1GTAlgoBlockTag"))),
      m_l1GTAlgoNames(iConfig.getParameter<std::vector<std::string>>("l1GTAlgoNames")),
      m_objectType(objectTypeFromString(iConfig.getParameter<std::string>("objectType"))),
      m_objectTypeName(iConfig.getParameter<std::string>("objectType")) {}

l1t::P2GTCandidate::ObjectType P2GTCandidateAnalyzer::objectTypeFromString(const std::string& name) {
  static const std::unordered_map<std::string, l1t::P2GTCandidate::ObjectType> kObjectTypeMap = {
      {"Undefined", l1t::P2GTCandidate::ObjectType::Undefined},
      {"GCTNonIsoEg", l1t::P2GTCandidate::ObjectType::GCTNonIsoEg},
      {"GCTIsoEg", l1t::P2GTCandidate::ObjectType::GCTIsoEg},
      {"GCTJets", l1t::P2GTCandidate::ObjectType::GCTJets},
      {"GCTTaus", l1t::P2GTCandidate::ObjectType::GCTTaus},
      {"GCTHtSum", l1t::P2GTCandidate::ObjectType::GCTHtSum},
      {"GCTEtSum", l1t::P2GTCandidate::ObjectType::GCTEtSum},
      {"GMTSaPromptMuons", l1t::P2GTCandidate::ObjectType::GMTSaPromptMuons},
      {"GMTSaDisplacedMuons", l1t::P2GTCandidate::ObjectType::GMTSaDisplacedMuons},
      {"GMTTkMuons", l1t::P2GTCandidate::ObjectType::GMTTkMuons},
      {"GMTTopo", l1t::P2GTCandidate::ObjectType::GMTTopo},
      {"GTTPromptJets", l1t::P2GTCandidate::ObjectType::GTTPromptJets},
      {"GTTDisplacedJets", l1t::P2GTCandidate::ObjectType::GTTDisplacedJets},
      {"GTTPhiCandidates", l1t::P2GTCandidate::ObjectType::GTTPhiCandidates},
      {"GTTRhoCandidates", l1t::P2GTCandidate::ObjectType::GTTRhoCandidates},
      {"GTTBsCandidates", l1t::P2GTCandidate::ObjectType::GTTBsCandidates},
      {"GTTHadronicTaus", l1t::P2GTCandidate::ObjectType::GTTHadronicTaus},
      {"GTTPromptTracks", l1t::P2GTCandidate::ObjectType::GTTPromptTracks},
      {"GTTDisplacedTracks", l1t::P2GTCandidate::ObjectType::GTTDisplacedTracks},
      {"GTTPrimaryVert", l1t::P2GTCandidate::ObjectType::GTTPrimaryVert},
      {"GTTPromptHtSum", l1t::P2GTCandidate::ObjectType::GTTPromptHtSum},
      {"GTTDisplacedHtSum", l1t::P2GTCandidate::ObjectType::GTTDisplacedHtSum},
      {"GTTEtSum", l1t::P2GTCandidate::ObjectType::GTTEtSum},
      {"CL2JetsSC4", l1t::P2GTCandidate::ObjectType::CL2JetsSC4},
      {"CL2JetsSC8", l1t::P2GTCandidate::ObjectType::CL2JetsSC8},
      {"CL2Taus", l1t::P2GTCandidate::ObjectType::CL2Taus},
      {"CL2Electrons", l1t::P2GTCandidate::ObjectType::CL2Electrons},
      {"CL2Photons", l1t::P2GTCandidate::ObjectType::CL2Photons},
      {"CL2HtSum", l1t::P2GTCandidate::ObjectType::CL2HtSum},
      {"CL2EtSum", l1t::P2GTCandidate::ObjectType::CL2EtSum},
  };

  auto it = kObjectTypeMap.find(name);
  if (it == kObjectTypeMap.end()) {
    throw cms::Exception("Configuration")
        << "P2GTCandidateAnalyzer: unknown objectType '" << name
        << "'. See l1t::P2GTCandidate::ObjectType in DataFormats/L1Trigger/interface/P2GTCandidate.h.";
  }
  return it->second;
}

// Try a single hw*() getter, appending "name=value" to oss on success and
// silently doing nothing if the field isn't set for this candidate.
namespace {
  template <typename Getter>
  void tryPrintField(std::ostringstream& oss, const l1t::P2GTCandidateRef& cand, const char* label, Getter getter) {
    try {
      oss << " " << label << "=" << getter(*cand);
    } catch (const std::invalid_argument&) {
      // field not set for this ObjectType; skip it
    }
  }
}  // namespace

std::string P2GTCandidateAnalyzer::printCandidate(const l1t::P2GTCandidateRef& cand) {
  std::ostringstream oss;
  oss << "pt=" << cand->pt() << " eta=" << cand->eta() << " phi=" << cand->phi()
      << " objectType=" << static_cast<int>(cand->objectType());

  tryPrintField(oss, cand, "hwPT", [](const l1t::P2GTCandidate& c) { return c.hwPT(); });
  tryPrintField(oss, cand, "hwPhi", [](const l1t::P2GTCandidate& c) { return c.hwPhi(); });
  tryPrintField(oss, cand, "hwEta", [](const l1t::P2GTCandidate& c) { return c.hwEta(); });
  tryPrintField(oss, cand, "hwZ0", [](const l1t::P2GTCandidate& c) { return c.hwZ0(); });
  tryPrintField(oss, cand, "hwIsolationPT", [](const l1t::P2GTCandidate& c) { return c.hwIsolationPT(); });
  tryPrintField(oss, cand, "hwQualityFlags", [](const l1t::P2GTCandidate& c) { return c.hwQualityFlags(); });
  tryPrintField(oss, cand, "hwQualityScore", [](const l1t::P2GTCandidate& c) { return c.hwQualityScore(); });
  tryPrintField(oss, cand, "hwCharge", [](const l1t::P2GTCandidate& c) { return c.hwCharge(); });
  tryPrintField(oss, cand, "hwD0", [](const l1t::P2GTCandidate& c) { return c.hwD0(); });
  tryPrintField(oss, cand, "hwBeta", [](const l1t::P2GTCandidate& c) { return c.hwBeta(); });
  tryPrintField(oss, cand, "hwMass", [](const l1t::P2GTCandidate& c) { return c.hwMass(); });
  tryPrintField(oss, cand, "hwIndex", [](const l1t::P2GTCandidate& c) { return c.hwIndex(); });
  tryPrintField(oss, cand, "hwSeed_pT", [](const l1t::P2GTCandidate& c) { return c.hwSeed_pT(); });
  tryPrintField(oss, cand, "hwSeed_z0", [](const l1t::P2GTCandidate& c) { return c.hwSeed_z0(); });
  tryPrintField(oss, cand, "hwScalarSumPT", [](const l1t::P2GTCandidate& c) { return c.hwScalarSumPT(); });
  tryPrintField(oss, cand, "hwNumber_of_tracks", [](const l1t::P2GTCandidate& c) { return c.hwNumber_of_tracks(); });
  tryPrintField(oss, cand, "hwNumber_of_displaced_tracks", [](const l1t::P2GTCandidate& c) {
    return c.hwNumber_of_displaced_tracks();
  });
  tryPrintField(oss, cand, "hwSum_pT_pv", [](const l1t::P2GTCandidate& c) { return c.hwSum_pT_pv(); });
  tryPrintField(oss, cand, "hwType", [](const l1t::P2GTCandidate& c) { return c.hwType(); });
  tryPrintField(oss, cand, "hwNumber_of_tracks_in_pv", [](const l1t::P2GTCandidate& c) {
    return c.hwNumber_of_tracks_in_pv();
  });
  tryPrintField(oss, cand, "hwNumber_of_tracks_not_in_pv", [](const l1t::P2GTCandidate& c) {
    return c.hwNumber_of_tracks_not_in_pv();
  });

  return oss.str();
}

void P2GTCandidateAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  std::vector<l1t::P2GTCandidateRef> vl1cands;

  const l1t::P2GTAlgoBlockMap& algos = iEvent.get(m_algoBlockToken);
  for (const auto& algoName : m_l1GTAlgoNames) {
    if (algos.count(algoName) > 0 && algos.at(algoName).decisionBeforeBxMaskAndPrescale()) {
      const l1t::P2GTCandidateVectorRef& objects = algos.at(algoName).trigObjects();
      for (const l1t::P2GTCandidateRef& obj : objects) {
        if (obj->objectType() == m_objectType) {
          vl1cands.push_back(obj);
	  edm::LogPrint("P2GTCandidateAnalyzer") << "Found P2GTCandidate ObjectType::" << m_objectTypeName
                                             << " from algo " << algoName;
        }
      }
    }
  }

  edm::LogInfo("P2GTCandidateAnalyzer")
      << "Matched " << vl1cands.size() << " P2GTCandidate(s) of type " << m_objectTypeName;

  for (const l1t::P2GTCandidateRef& cand : vl1cands) {
    edm::LogPrint("P2GTCandidateAnalyzer") << "  " << printCandidate(cand);
  }
}

void P2GTCandidateAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("l1GTAlgoBlockTag", edm::InputTag("l1tGTProducer", "AlgoBlocks"));
  desc.add<std::vector<std::string>>("l1GTAlgoNames", std::vector<std::string>{});
  desc.add<std::string>("objectType", "GMTTkMuons");
  descriptions.addWithDefaultLabel(desc);
}

DEFINE_FWK_MODULE(P2GTCandidateAnalyzer);
