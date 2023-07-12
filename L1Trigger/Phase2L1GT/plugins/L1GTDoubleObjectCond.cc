#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/Ref.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "L1Trigger/Phase2L1GT/interface/L1GTInvariantMassError.h"

#include "L1Trigger/Phase2L1GT/interface/L1GTScales.h"

#include "L1GTOptionalParam.h"
#include "L1GTSingleCollectionCut.h"
#include "L1GTDeltaCut.h"
#include "L1GTSingleInOutLUT.h"

#include <cinttypes>
#include <memory>
#include <vector>
#include <set>

#include <ap_int.h>

using namespace l1t;

class L1GTDoubleObjectCond : public edm::global::EDFilter<> {
public:
  explicit L1GTDoubleObjectCond(const edm::ParameterSet&);
  ~L1GTDoubleObjectCond() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  bool filter(edm::StreamID, edm::Event&, const edm::EventSetup&) const override;

  const L1GTScales scales_;

  const L1GTSingleCollectionCut collection1Cuts_;
  const L1GTSingleCollectionCut collection2Cuts_;

  const bool enable_sanity_checks_;
  const bool inv_mass_checks_;

  const L1GTDeltaCut deltaCuts_;

  const edm::EDGetTokenT<P2GTCandidateCollection> token1_;
  const edm::EDGetTokenT<P2GTCandidateCollection> token2_;
};

L1GTDoubleObjectCond::L1GTDoubleObjectCond(const edm::ParameterSet& config)
    : scales_(config.getParameter<edm::ParameterSet>("scales")),
      collection1Cuts_(config.getParameterSet("collection1"), config, scales_),
      collection2Cuts_(config.getParameterSet("collection2"), config, scales_),
      enable_sanity_checks_(config.getUntrackedParameter<bool>("sanity_checks")),
      inv_mass_checks_(config.getUntrackedParameter<bool>("inv_mass_checks")),
      deltaCuts_(config, config, scales_, enable_sanity_checks_, inv_mass_checks_),
      token1_(consumes<P2GTCandidateCollection>(collection1Cuts_.tag())),
      token2_(collection1Cuts_.tag() == collection2Cuts_.tag()
                  ? token1_
                  : consumes<P2GTCandidateCollection>(collection2Cuts_.tag())) {
  produces<P2GTCandidateVectorRef>(collection1Cuts_.tag().instance());

  if (!(collection1Cuts_.tag() == collection2Cuts_.tag())) {
    produces<P2GTCandidateVectorRef>(collection2Cuts_.tag().instance());
  }

  if (inv_mass_checks_) {
    produces<InvariantMassErrorCollection>();
  }
}

void L1GTDoubleObjectCond::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  edm::ParameterSetDescription collection1Desc;
  L1GTSingleCollectionCut::fillPSetDescription(collection1Desc);
  desc.add<edm::ParameterSetDescription>("collection1", collection1Desc);

  edm::ParameterSetDescription collection2Desc;
  L1GTSingleCollectionCut::fillPSetDescription(collection2Desc);
  desc.add<edm::ParameterSetDescription>("collection2", collection2Desc);

  desc.addUntracked<bool>("sanity_checks", false);
  desc.addUntracked<bool>("inv_mass_checks", false);

  L1GTDeltaCut::fillPSetDescription(desc);
  L1GTDeltaCut::fillLUTDescriptions(desc);

  edm::ParameterSetDescription scalesDesc;
  L1GTScales::fillPSetDescription(scalesDesc);
  desc.add<edm::ParameterSetDescription>("scales", scalesDesc);

  descriptions.addWithDefaultLabel(desc);
}

bool L1GTDoubleObjectCond::filter(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
  edm::Handle<P2GTCandidateCollection> col1 = event.getHandle(token1_);
  edm::Handle<P2GTCandidateCollection> col2 = event.getHandle(token2_);

  bool condition_result = false;

  std::set<std::size_t> triggeredIdcs1;
  std::set<std::size_t> triggeredIdcs2;

  InvariantMassErrorCollection massErrors;

  for (std::size_t idx1 = 0; idx1 < col1->size(); ++idx1) {
    for (std::size_t idx2 = 0; idx2 < col2->size(); ++idx2) {
      // If we're looking at the same collection then we shouldn't use the same object in one comparison.
      if (col1.product() == col2.product() && idx1 == idx2) {
        continue;
      }

      bool pass = true;
      pass &= collection1Cuts_.checkObject(col1->at(idx1));
      pass &= collection2Cuts_.checkObject(col2->at(idx2));
      pass &= deltaCuts_.checkObjects(col1->at(idx1), col2->at(idx2), massErrors);

      condition_result |= pass;

      if (pass) {
        triggeredIdcs1.emplace(idx1);
        if (col1.product() != col2.product()) {
          triggeredIdcs2.emplace(idx2);
        } else {
          triggeredIdcs1.emplace(idx2);
        }
      }
    }
  }

  if (condition_result) {
    std::unique_ptr<P2GTCandidateVectorRef> triggerCol1 = std::make_unique<P2GTCandidateVectorRef>();

    for (std::size_t idx : triggeredIdcs1) {
      triggerCol1->push_back(P2GTCandidateRef(col1, idx));
    }
    event.put(std::move(triggerCol1), collection1Cuts_.tag().instance());

    if (col1.product() != col2.product()) {
      std::unique_ptr<P2GTCandidateVectorRef> triggerCol2 = std::make_unique<P2GTCandidateVectorRef>();

      for (std::size_t idx : triggeredIdcs2) {
        triggerCol2->push_back(P2GTCandidateRef(col2, idx));
      }
      event.put(std::move(triggerCol2), collection2Cuts_.tag().instance());
    }
  }

  if (inv_mass_checks_) {
    event.put(std::make_unique<InvariantMassErrorCollection>(std::move(massErrors)), "");
  }

  return condition_result;
}

DEFINE_FWK_MODULE(L1GTDoubleObjectCond);
