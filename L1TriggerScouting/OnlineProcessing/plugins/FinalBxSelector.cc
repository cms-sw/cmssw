#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include <vector>
#include <set>

/*
 * Filter orbits that don't contain at least one selected BX
 * from a BxSelector module and produce a vector of selected BXs
 */
class FinalBxSelector : public edm::stream::EDFilter<> {
public:
  explicit FinalBxSelector(const edm::ParameterSet&);
  ~FinalBxSelector() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  bool filter(edm::Event&, const edm::EventSetup&) override;

  // tokens for BX selected by each analysis
  std::vector<edm::EDGetTokenT<std::vector<unsigned>>> selectedBxsToken_;
};

FinalBxSelector::FinalBxSelector(const edm::ParameterSet& iPSet) {
  // get the list of selected BXs
  std::vector<edm::InputTag> bxLabels = iPSet.getParameter<std::vector<edm::InputTag>>("analysisLabels");
  for (const auto& bxLabel : bxLabels) {
    selectedBxsToken_.push_back(consumes<std::vector<unsigned>>(bxLabel));
  }

  produces<std::vector<unsigned>>("SelBx").setBranchAlias("SelectedBxs");
}

// ------------ method called for each ORBIT  ------------
bool FinalBxSelector::filter(edm::Event& iEvent, const edm::EventSetup&) {
  bool noBxSelected = true;
  std::set<unsigned> uniqueBxs;

  for (const auto& token : selectedBxsToken_) {
    edm::Handle<std::vector<unsigned>> bxList;
    iEvent.getByToken(token, bxList);

    for (const unsigned& bx : *bxList) {
      uniqueBxs.insert(bx);
      noBxSelected = false;
    }
  }

  auto selectedBxs = std::make_unique<std::vector<unsigned>>(uniqueBxs.begin(), uniqueBxs.end());
  iEvent.put(std::move(selectedBxs), "SelBx");

  return !noBxSelected;
}

void FinalBxSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(FinalBxSelector);
