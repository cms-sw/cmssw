#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "DataFormats/Luminosity/interface/PixelClusterCounts.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Utilities/interface/EDGetToken.h"

namespace {
  struct Empty {};
}  // namespace
class PixelClusterCountsAuditor : public edm::global::EDAnalyzer<edm::LuminosityBlockCache<Empty>> {
public:
  PixelClusterCountsAuditor(edm::ParameterSet const& iPSet);

  void analyze(edm::StreamID id, edm::Event const&, edm::EventSetup const&) const final {}
  std::shared_ptr<Empty> globalBeginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const final;

  void globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const final;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  std::vector<edm::EDGetTokenT<reco::PixelClusterCounts>> tokens_;
};

PixelClusterCountsAuditor::PixelClusterCountsAuditor(edm::ParameterSet const& iPSet) {
  auto tags = iPSet.getUntrackedParameter<std::vector<edm::InputTag>>("counts");
  tokens_.reserve(tags.size());
  for (auto const& t : tags) {
    tokens_.emplace_back(consumes<reco::PixelClusterCounts, edm::InLumi>(t));
  }
}
void PixelClusterCountsAuditor::fillDescriptions(edm::ConfigurationDescriptions& iConfig) {
  edm::ParameterSetDescription desc;
  desc.addUntracked<std::vector<edm::InputTag>>("counts")->setComment(
      "Which PixelClusterCounts to retrieve from the LuminosityBlock");
  iConfig.addDefault(desc);
}

std::shared_ptr<Empty> PixelClusterCountsAuditor::globalBeginLuminosityBlock(edm::LuminosityBlock const& iLumi,
                                                                             edm::EventSetup const&) const {
  for (auto t : tokens_) {
    auto h = iLumi.getHandle(t);
    auto prov = h.provenance();
    auto const& c = *h;
    edm::LogSystem("PixelClusterCountsAudit")
        .format("Branch: {}\n readCounts: {}\n readRocCounts: {}\n readEvents: {}\n readModID: {}\n readRocID: {}",
                prov->branchName(),
                c.readCounts().size(),
                c.readRocCounts().size(),
                c.readEvents().size(),
                c.readModID().size(),
                c.readRocID().size());
  }
  return {};
}

void PixelClusterCountsAuditor::globalEndLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) const {}

DEFINE_FWK_MODULE(PixelClusterCountsAuditor);
