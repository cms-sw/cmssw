#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/L1Trigger/interface/EtSum.h"

#include <unordered_set>
#include <vector>

class L1TEtSumsPrinter : public edm::global::EDAnalyzer<> {
public:
  explicit L1TEtSumsPrinter(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;

  edm::EDGetTokenT<l1t::EtSumBxCollection> const srcToken_;
  std::unordered_set<int> const etSumTypes_;
};

L1TEtSumsPrinter::L1TEtSumsPrinter(const edm::ParameterSet& iConfig)
    : srcToken_{consumes(iConfig.getParameter<edm::InputTag>("src"))}, etSumTypes_{[](std::vector<int> const& vInts) {
        return std::unordered_set<int>{vInts.begin(), vInts.end()};
      }(iConfig.getParameter<std::vector<int>>("etSumTypes"))} {}

void L1TEtSumsPrinter::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
  auto const& etSums = iEvent.get(srcToken_);
  auto const& moduleLabel = moduleDescription().moduleLabel();
  for (int ibx = etSums.getFirstBX(); ibx <= etSums.getLastBX(); ++ibx) {
    auto const size = etSums.size(ibx);
    for (uint idx = 0; idx < size; ++idx) {
      auto const& etSum = etSums.at(ibx, idx);
      if ((not etSumTypes_.empty()) and etSumTypes_.find(etSum.getType()) == etSumTypes_.end()) {
        continue;
      }

      edm::LogPrint("L1TEtSumsPrinter") << "[" << moduleLabel << "] etSums[" << ibx << "][" << idx
                                        << "] (type, hwPt) = (" << etSum.getType() << ", " << etSum.hwPt() << ")";
    }
  }
}

void L1TEtSumsPrinter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag("gtStage2Digis:EtSum"))
      ->setComment("Input collection of type l1t::EtSumBxCollection");
  desc.add<std::vector<int>>("etSumTypes", {})
      ->setComment("If not empty, only the specified l1t::EtSumType values are considered");
  descriptions.add("l1tEtSumsPrinter", desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TEtSumsPrinter);
