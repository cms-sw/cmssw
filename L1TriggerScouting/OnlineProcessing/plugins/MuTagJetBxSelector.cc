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
#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "L1TriggerScouting/Utilities/interface/conversion.h"

// root libraries
#include "TLorentzVector.h"
#include "Math/VectorUtil.h"

#include <memory>
#include <utility>
#include <vector>

using namespace l1ScoutingRun3;

class MuTagJetBxSelector : public edm::stream::EDProducer<> {
public:
  explicit MuTagJetBxSelector(const edm::ParameterSet&);
  ~MuTagJetBxSelector() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // tokens for scouting data
  edm::EDGetTokenT<OrbitCollection<l1ScoutingRun3::Muon>> muonsTokenData_;
  edm::EDGetTokenT<OrbitCollection<l1ScoutingRun3::Jet>> jetsTokenData_;

  // SELECTION THRESHOLDS
  int minNJet_;
  std::vector<double> minJetEt_;
  std::vector<double> maxJetEta_;
  std::vector<double> minMuPt_;
  std::vector<double> maxMuEta_;
  std::vector<int> minMuTfIndex_;
  std::vector<int> maxMuTfIndex_;
  std::vector<int> minMuHwQual_;
  std::vector<double> maxDR_;
};

MuTagJetBxSelector::MuTagJetBxSelector(const edm::ParameterSet& iPSet)
    : muonsTokenData_(consumes(iPSet.getParameter<edm::InputTag>("muonsTag"))),
      jetsTokenData_(consumes(iPSet.getParameter<edm::InputTag>("jetsTag"))),
      minNJet_(iPSet.getParameter<int>("minNJet")),
      minJetEt_(iPSet.getParameter<std::vector<double>>("minJetEt")),
      maxJetEta_(iPSet.getParameter<std::vector<double>>("maxJetEta")),
      minMuPt_(iPSet.getParameter<std::vector<double>>("minMuPt")),
      maxMuEta_(iPSet.getParameter<std::vector<double>>("maxMuEta")),
      minMuTfIndex_(iPSet.getParameter<std::vector<int>>("minMuTfIndex")),
      maxMuTfIndex_(iPSet.getParameter<std::vector<int>>("maxMuTfIndex")),
      minMuHwQual_(iPSet.getParameter<std::vector<int>>("minMuHwQual")),
      maxDR_(iPSet.getParameter<std::vector<double>>("maxDR")) {
  if ((minJetEt_.size() != (size_t)minNJet_) || (maxJetEta_.size() != (size_t)minNJet_) ||
      (minMuPt_.size() != (size_t)minNJet_) || (maxMuEta_.size() != (size_t)minNJet_) ||
      (minMuTfIndex_.size() != (size_t)minNJet_) || (maxMuTfIndex_.size() != (size_t)minNJet_) ||
      (minMuHwQual_.size() != (size_t)minNJet_) || (maxDR_.size() != (size_t)minNJet_))
    throw cms::Exception("MuTagJetBxSelector::MuTagJetBxSelector")
        << "size mismatch: size of minJetEt or maxJetEta or  minMuPt or maxMuEta or minMuTfIndex or maxMuTfIndex or "
           "minMuHwQual or maxDR != minNMu.";

  produces<std::vector<unsigned>>("SelBx").setBranchAlias("MuTagJetSelectedBx");
}

// ------------ method called for each ORBIT  ------------
void MuTagJetBxSelector::produce(edm::Event& iEvent, const edm::EventSetup&) {
  edm::Handle<OrbitCollection<l1ScoutingRun3::Muon>> muonsCollection;
  edm::Handle<OrbitCollection<l1ScoutingRun3::Jet>> jetsCollection;

  iEvent.getByToken(muonsTokenData_, muonsCollection);
  iEvent.getByToken(jetsTokenData_, jetsCollection);

  std::unique_ptr<std::vector<unsigned>> muTagJetBx(new std::vector<unsigned>);

  // loop over valid bunch crossings
  for (const unsigned& bx : jetsCollection->getFilledBxs()) {
    const auto& jets = jetsCollection->bxIterator(bx);
    const auto& muons = muonsCollection->bxIterator(bx);

    // we have at least N jets and N muons
    if (jets.size() < minNJet_ && muons.size() < minNJet_)
      continue;

    // it must satisfy certain requirements
    bool jetCond = false;
    bool muCond = false;
    int nAccJets = 0;
    for (const auto& jet : jets) {
      jetCond = (std::abs(demux::fEta(jet.hwEta())) < maxJetEta_[nAccJets]) &&
                (demux::fEt(jet.hwEt()) >= minJetEt_[nAccJets]);
      if (!jetCond)
        continue;  // jet does not satisfy requirements, next one
      ROOT::Math::PtEtaPhiMVector jetLV(demux::fEt(jet.hwEt()), demux::fEta(jet.hwEta()), demux::fPhi(jet.hwPhi()), 0);

      for (const auto& muon : muons) {
        muCond = (std::abs(ugmt::fEta(muon.hwEta())) < maxMuEta_[nAccJets]) &&
                 (muon.tfMuonIndex() <= maxMuTfIndex_[nAccJets]) && (muon.tfMuonIndex() >= minMuTfIndex_[nAccJets]) &&
                 (ugmt::fPt(muon.hwPt()) >= minMuPt_[nAccJets]) && (muon.hwQual() >= minMuHwQual_[nAccJets]);
        if (!muCond)
          continue;  // muon does not satisfy requirements, next one
        ROOT::Math::PtEtaPhiMVector muLV(
            ugmt::fPt(muon.hwPt()), ugmt::fEta(muon.hwEta()), ugmt::fPhi(muon.hwPhi()), 0.1057);

        float dr = ROOT::Math::VectorUtil::DeltaR(jetLV, muLV);
        if (dr < maxDR_[nAccJets]) {
          nAccJets++;  // found mu-tag for current jet, end muon loop
          break;
        }
      }

      if (nAccJets == minNJet_)
        break;  // found all requested mu-tagged jets
    }

    if (nAccJets < minNJet_)
      continue;

    muTagJetBx->push_back(bx);

  }  // end orbit loop

  iEvent.put(std::move(muTagJetBx), "SelBx");
}

void MuTagJetBxSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(MuTagJetBxSelector);
