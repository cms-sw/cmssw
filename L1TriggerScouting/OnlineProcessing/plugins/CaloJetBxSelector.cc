#include <cmath>
#include <memory>
#include <utility>
#include <vector>

#include "DataFormats/L1Scouting/interface/L1ScoutingCaloJet.h"
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"

class CaloJetBxSelector : public edm::global::EDProducer<> {
public:
  explicit CaloJetBxSelector(const edm::ParameterSet&);

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  edm::EDGetTokenT<l1ScoutingRun3::CaloJetOrbitCollection> const jetsToken_;
  unsigned int const minNJet_;
  std::vector<double> const minJetPt_;
  std::vector<double> const maxJetAbsEta_;
  std::vector<int> const minJetNConst_;
};

CaloJetBxSelector::CaloJetBxSelector(const edm::ParameterSet& iPSet)
    : jetsToken_(consumes(iPSet.getParameter<edm::InputTag>("jetsTag"))),
      minNJet_(iPSet.getParameter<unsigned int>("minNJet")),
      minJetPt_(iPSet.getParameter<std::vector<double>>("minJetPt")),
      maxJetAbsEta_(iPSet.getParameter<std::vector<double>>("maxJetAbsEta")),
      minJetNConst_(iPSet.getParameter<std::vector<int>>("minJetNConst")) {
  if (minNJet_ != minJetPt_.size()) {
    throw cms::Exception("InvalidConfiguration")
        << "invalid parameter values: \"minNJet\" (" << minNJet_ << ") differs from the size of \"minJetPt\" ("
        << minJetPt_.size() << ")";
  }

  if (minNJet_ != maxJetAbsEta_.size()) {
    throw cms::Exception("InvalidConfiguration")
        << "invalid parameter values: \"minNJet\" (" << minNJet_ << ") differs from the size of \"maxJetAbsEta\" ("
        << maxJetAbsEta_.size() << ")";
  }

  if (minNJet_ != minJetNConst_.size()) {
    throw cms::Exception("InvalidConfiguration")
        << "invalid parameter values: \"minNJet\" (" << minNJet_ << ") differs from the size of \"minJetNConst\" ("
        << minJetNConst_.size() << ")";
  }

  produces<std::vector<unsigned int>>("SelBx").setBranchAlias("CaloJetSelectedBx");
}

// ------------ method called for each ORBIT  ------------
void CaloJetBxSelector::produce(edm::StreamID, edm::Event& iEvent, const edm::EventSetup&) const {
  auto const& caloJetsInOrbit = iEvent.get(jetsToken_);

  auto jetBx = std::make_unique<std::vector<unsigned int>>();

  // loop over valid BXs with CaloJets
  for (auto const bx : caloJetsInOrbit.getFilledBxs()) {
    if (minNJet_ > 0) {
      auto const& jets = caloJetsInOrbit.bxIterator(bx);

      // skip BX if the number of jets is below the minimum
      if (jets.size() < minNJet_) {
        continue;
      }

      // check if there are enough jets passing selection requirements (pT, |eta|, #constituents)
      unsigned int nAccJets{0};
      for (auto const& jet : jets) {
        // increment nAccJets if the jet passes all requirements
        if (jet.pt() >= minJetPt_[nAccJets] and
            (maxJetAbsEta_[nAccJets] < 0 or std::abs(jet.eta()) < maxJetAbsEta_[nAccJets]) and
            jet.nConst() >= minJetNConst_[nAccJets]) {
          ++nAccJets;
        }
        // found enough jets, so exit early
        if (nAccJets == minNJet_) {
          break;
        }
      }

      // skip BX if the number of selected jets is below the minimum
      if (nAccJets < minNJet_) {
        continue;
      }
    }

    jetBx->emplace_back(bx);
  }  // end orbit loop

  iEvent.put(std::move(jetBx), "SelBx");
}

void CaloJetBxSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("jetsTag")->setComment(
      "Input collection of CaloJets (type: l1ScoutingRun3::CaloJetOrbitCollection)");
  desc.add<unsigned int>("minNJet", 0)
      ->setComment("Min number of jets passing min-pT, max-|eta| and min-#constituents requirements (inclusive)");
  desc.add<std::vector<double>>("minJetPt", {})
      ->setComment("Min jet-pT thresholds (size must be equal to the value of \"minNJet\")");
  desc.add<std::vector<double>>("maxJetAbsEta", {})
      ->setComment(
          "Max jet-|eta| thresholds (size must be equal to the value of \"minNJet\"; a negative value corresponds to "
          "not applying the |eta| cut)");
  desc.add<std::vector<int>>("minJetNConst", {})
      ->setComment("Thresholds on the min number of jet constituents (size must be equal to the value of \"minNJet\")");
  descriptions.addDefault(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(CaloJetBxSelector);
