// EDAnalyzer used by the HLTP2GTFilter unit test.
// For each event it compares the decision of an L1 PathStatusFilter path
// (the reference) against the decision of an HLTP2GT*ObjectFilter path
// (the new implementation) and throws in endJob if any event disagreed.
//
// Uses one::EDAnalyzer so the mutable mismatch counter needs no locking.

#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <string>

class HLTP2GTFilterTestAnalyzer : public edm::one::EDAnalyzer<> {
public:
  explicit HLTP2GTFilterTestAnalyzer(const edm::ParameterSet&);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void endJob() override;

  const edm::EDGetTokenT<edm::TriggerResults> m_trigResToken;
  const std::string m_referencePath;
  const std::string m_underTestPath;
  unsigned int m_nMismatches{0};
  unsigned int m_nEvents{0};
};

HLTP2GTFilterTestAnalyzer::HLTP2GTFilterTestAnalyzer(const edm::ParameterSet& iConfig)
    : m_trigResToken(consumes<edm::TriggerResults>(
          iConfig.getParameter<edm::InputTag>("triggerResults"))),
      m_referencePath(iConfig.getParameter<std::string>("referencePath")),
      m_underTestPath(iConfig.getParameter<std::string>("underTestPath")) {}

void HLTP2GTFilterTestAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("triggerResults", edm::InputTag("TriggerResults", "", "TEST"));
  desc.add<std::string>("referencePath", "");
  desc.add<std::string>("underTestPath", "");
  descriptions.addWithDefaultLabel(desc);
}

void HLTP2GTFilterTestAnalyzer::analyze(const edm::Event& iEvent, const edm::EventSetup&) {
  ++m_nEvents;

  const auto& trigRes = iEvent.get(m_trigResToken);
  const auto& names   = iEvent.triggerNames(trigRes);

  auto getIndex = [&](const std::string& pathName) -> unsigned int {
    const unsigned int idx = names.triggerIndex(pathName);
    if (idx >= trigRes.size())
      throw cms::Exception("Configuration")
          << "HLTP2GTFilterTestAnalyzer: path \"" << pathName
          << "\" not found in TriggerResults.\nAvailable paths:\n"
          << [&]() {
               std::string s;
               for (const auto& n : names.triggerNames())
                 s += "  " + n + "\n";
               return s;
             }();
    return idx;
  };

  const bool refAccept  = trigRes.accept(getIndex(m_referencePath));
  const bool testAccept = trigRes.accept(getIndex(m_underTestPath));

  if (refAccept != testAccept) {
    ++m_nMismatches;
    edm::LogPrint("HLTP2GTFilterTestAnalyzer")
        << "MISMATCH run=" << iEvent.run()
        << " lumi=" << iEvent.luminosityBlock()
        << " event=" << iEvent.id().event()
        << "  reference(" << m_referencePath << ")=" << refAccept
        << "  underTest(" << m_underTestPath << ")=" << testAccept;
  }
}

void HLTP2GTFilterTestAnalyzer::endJob() {
  if (m_nMismatches > 0)
    throw cms::Exception("TestFailure")
        << "HLTP2GTFilterTestAnalyzer [" << m_underTestPath << "]:"
        << " found " << m_nMismatches << " mismatched decision(s)"
        << " out of " << m_nEvents << " events.";
  edm::LogPrint("HLTP2GTFilterTestAnalyzer")
      << "[" << m_underTestPath << "] OK — "
      << m_nEvents << " events, 0 mismatches.";
}

DEFINE_FWK_MODULE(HLTP2GTFilterTestAnalyzer);
