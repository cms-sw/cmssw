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
#include "DataFormats/L1Scouting/interface/L1ScoutingCalo.h"
#include "DataFormats/L1Scouting/interface/OrbitCollection.h"
#include "L1TriggerScouting/Utilities/interface/conversion.h"

#include <memory>
#include <utility>
#include <vector>

using namespace l1ScoutingRun3;

class JetBxSelector : public edm::stream::EDProducer<> {
public:
  explicit JetBxSelector(const edm::ParameterSet&);
  ~JetBxSelector() override {}
  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;

  // tokens for scouting data
  edm::EDGetTokenT<OrbitCollection<l1ScoutingRun3::Jet>> jetsTokenData_;

  // SELECTION THRESHOLDS
  int minNJet_;
  std::vector<double> minJetEt_;
  std::vector<double> maxJetEta_;
};

JetBxSelector::JetBxSelector(const edm::ParameterSet& iPSet)
    : jetsTokenData_(consumes(iPSet.getParameter<edm::InputTag>("jetsTag"))),
      minNJet_(iPSet.getParameter<int>("minNJet")),
      minJetEt_(iPSet.getParameter<std::vector<double>>("minJetEt")),
      maxJetEta_(iPSet.getParameter<std::vector<double>>("maxJetEta"))

{
  if ((minJetEt_.size() != (size_t)minNJet_) || (maxJetEta_.size() != (size_t)minNJet_))
    throw cms::Exception("JetBxSelector::JetBxSelector") << "size mismatch: size of minJetEt or maxJetEta != minNJet.";

  produces<std::vector<unsigned>>("SelBx").setBranchAlias("JetSelectedBx");
}

// ------------ method called for each ORBIT  ------------
void JetBxSelector::produce(edm::Event& iEvent, const edm::EventSetup&) {
  edm::Handle<OrbitCollection<l1ScoutingRun3::Jet>> jetsCollection;

  iEvent.getByToken(jetsTokenData_, jetsCollection);

  std::unique_ptr<std::vector<unsigned>> jetBx(new std::vector<unsigned>);

  // loop over valid bunch crossings
  for (const unsigned& bx : jetsCollection->getFilledBxs()) {
    const auto& jets = jetsCollection->bxIterator(bx);

    // we have at least N jets
    if (jets.size() < minNJet_)
      continue;

    // it must be in a certain eta region with an pT and quality threshold
    bool jetCond = false;
    int nAccJets = 0;
    for (const auto& jet : jets) {
      jetCond = (std::abs(demux::fEta(jet.hwEta())) < maxJetEta_[nAccJets]) &&
                (demux::fEt(jet.hwEt()) >= minJetEt_[nAccJets]);
      if (jetCond)
        nAccJets++;  // found jet meeting requirements
      if (nAccJets == minNJet_)
        break;  // found all requested jets
    }

    if (nAccJets < minNJet_)
      continue;

    jetBx->push_back(bx);

  }  // end orbit loop

  iEvent.put(std::move(jetBx), "SelBx");
}

void JetBxSelector::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.setUnknown();
  descriptions.addDefault(desc);
}

DEFINE_FWK_MODULE(JetBxSelector);
