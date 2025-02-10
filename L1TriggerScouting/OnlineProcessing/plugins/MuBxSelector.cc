#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"
#include "FWCore/Utilities/interface/Span.h"

// L1 scouting
#include "DataFormats/L1Scouting/interface/L1ScoutingMuon.h"
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "L1TriggerScouting/Utilities/interface/conversion.h"

#include <memory>
#include <utility>
#include <vector>

using namespace l1ScoutingRun3;

class MuBxSelector : public edm::stream::EDProducer<> {
public:
  explicit MuBxSelector(const edm::ParameterSet&);
  ~MuBxSelector() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // tokens for scouting data
  edm::EDGetTokenT<OrbitCollection<l1ScoutingRun3::Muon>> muonsTokenData_;

  // SELECTION THRESHOLDS
  int minNMu_;
  std::vector<double> minMuPt_;
  std::vector<double> maxMuEta_;
  std::vector<int> minMuTfIndex_;
  std::vector<int> maxMuTfIndex_;
  std::vector<int> minMuHwQual_;
};

MuBxSelector::MuBxSelector(const edm::ParameterSet& iPSet)
    : muonsTokenData_(consumes(iPSet.getParameter<edm::InputTag>("muonsTag"))),
      minNMu_(iPSet.getParameter<int>("minNMu")),
      minMuPt_(iPSet.getParameter<std::vector<double>>("minMuPt")),
      maxMuEta_(iPSet.getParameter<std::vector<double>>("maxMuEta")),
      minMuTfIndex_(iPSet.getParameter<std::vector<int>>("minMuTfIndex")),
      maxMuTfIndex_(iPSet.getParameter<std::vector<int>>("maxMuTfIndex")),
      minMuHwQual_(iPSet.getParameter<std::vector<int>>("minMuHwQual"))

{
  if ((minMuPt_.size() != (size_t)(size_t)minNMu_) || (maxMuEta_.size() != (size_t)minNMu_) ||
      (minMuTfIndex_.size() != (size_t)minNMu_) || (maxMuTfIndex_.size() != (size_t)minNMu_) ||
      (minMuHwQual_.size() != (size_t)minNMu_))
    throw cms::Exception("MuBxSelector::MuBxSelector")
        << "size mismatch: size of minMuPt or maxMuEta or minMuTfIndex or maxMuTfIndex or minMuHwQual  != minNMu.";

  produces<std::vector<unsigned>>("SelBx").setBranchAlias("MuSelectedBx");
}

// ------------ method called for each ORBIT  ------------
void MuBxSelector::produce(edm::Event& iEvent, const edm::EventSetup&) {
  edm::Handle<OrbitCollection<l1ScoutingRun3::Muon>> muonsCollection;

  iEvent.getByToken(muonsTokenData_, muonsCollection);

  std::unique_ptr<std::vector<unsigned>> muBx(new std::vector<unsigned>);

  // loop over valid bunch crossings
  for (const unsigned& bx : muonsCollection->getFilledBxs()) {
    const auto& muons = muonsCollection->bxIterator(bx);

    // we have at least a muon
    if (muons.size() < minNMu_)
      continue;

    // it must be in a certain eta region with an pT and quality threshold
    bool muCond = false;
    int nAccMus = 0;
    for (const auto& muon : muons) {
      muCond = (std::abs(ugmt::fEta(muon.hwEta())) < maxMuEta_[nAccMus]) &&
               (muon.tfMuonIndex() <= maxMuTfIndex_[nAccMus]) && (muon.tfMuonIndex() >= minMuTfIndex_[nAccMus]) &&
               (ugmt::fPt(muon.hwPt()) >= minMuPt_[nAccMus]) && (muon.hwQual() >= minMuHwQual_[nAccMus]);
      if (muCond)
        nAccMus++;  // found muon meeting requirements
      if (nAccMus == minNMu_)
        break;  // found all requested muons
    }

    if (nAccMus < minNMu_)
      continue;

    muBx->push_back(bx);

  }  // end orbit loop

  iEvent.put(std::move(muBx), "SelBx");
}

void MuBxSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(MuBxSelector);
