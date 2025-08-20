#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/Utilities/interface/EDPutToken.h"

#include "DataFormats/TestObjects/interface/ToyProducts.h"

class BranchIDListsModifierProducer : public edm::global::EDProducer<> {
public:
  BranchIDListsModifierProducer(edm::ParameterSet const& iPSet);

  void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  static void fillDescriptions(edm::ConfigurationDescriptions& iDesc);

private:
  edm::EDPutTokenT<int> const token_;
  edm::EDPutTokenT<edmtest::ATransientIntProduct> extraToken_;
  bool const extraProduct_;
};

BranchIDListsModifierProducer::BranchIDListsModifierProducer(edm::ParameterSet const& iPSet)
    : token_(produces()), extraProduct_(iPSet.getUntrackedParameter<bool>("makeExtraProduct")) {
  if (extraProduct_) {
    extraToken_ = produces("extra");
  }
}

void BranchIDListsModifierProducer::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
  iEvent.emplace(token_, 1);
  if (extraProduct_) {
    iEvent.emplace(extraToken_, 2);
  }
}

void BranchIDListsModifierProducer::fillDescriptions(edm::ConfigurationDescriptions& iDesc) {
  edm::ParameterSetDescription ps;
  ps.setComment(
      "Module which can cause the BranchIDLists to change even when the top level PSet remains the same.\n"
      "Used for multi-file merge tests.");

  ps.addUntracked<bool>("makeExtraProduct", false)->setComment("If set to true will produce an extra product");

  iDesc.addDefault(ps);
}

DEFINE_FWK_MODULE(BranchIDListsModifierProducer);
