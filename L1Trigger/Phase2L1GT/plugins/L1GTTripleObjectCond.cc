#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/Ref.h"
#include "FWCore/Utilities/interface/EDGetToken.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "L1Trigger/Phase2L1GT/interface/L1GTScales.h"
#include "L1GTSingleCollectionCut.h"
#include "L1GTDeltaCut.h"
#include "L1GTSingleInOutLUT.h"

#include <set>

#include <ap_int.h>

using namespace l1t;

class L1GTTripleObjectCond : public edm::global::EDFilter<> {
public:
  explicit L1GTTripleObjectCond(const edm::ParameterSet&);
  ~L1GTTripleObjectCond() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  const L1GTScales scales_;

  const L1GTSingleCollectionCut collection1Cuts_;
  const L1GTSingleCollectionCut collection2Cuts_;
  const L1GTSingleCollectionCut collection3Cuts_;

  const bool enable_sanity_checks_;
  const bool inv_mass_checks_;

  const L1GTDeltaCut delta12Cuts_;
  const L1GTDeltaCut delta13Cuts_;
  const L1GTDeltaCut delta23Cuts_;

  const edm::EDGetTokenT<P2GTCandidateCollection> token1_;
  const edm::EDGetTokenT<P2GTCandidateCollection> token2_;
  const edm::EDGetTokenT<P2GTCandidateCollection> token3_;
};

L1GTTripleObjectCond::L1GTTripleObjectCond(const edm::ParameterSet& config)
    : scales_(config.getParameter<edm::ParameterSet>("scales")),
      collection1Cuts_(config.getParameter<edm::ParameterSet>("collection1"), config, scales_),
      collection2Cuts_(config.getParameter<edm::ParameterSet>("collection2"), config, scales_),
      collection3Cuts_(config.getParameter<edm::ParameterSet>("collection3"), config, scales_),
      enable_sanity_checks_(config.getUntrackedParameter<bool>("sanity_checks")),
      inv_mass_checks_(config.getUntrackedParameter<bool>("inv_mass_checks")),
      delta12Cuts_(
          config.getParameter<edm::ParameterSet>("delta12"), config, scales_, enable_sanity_checks_, inv_mass_checks_),
      delta13Cuts_(
          config.getParameter<edm::ParameterSet>("delta13"), config, scales_, enable_sanity_checks_, inv_mass_checks_),
      delta23Cuts_(
          config.getParameter<edm::ParameterSet>("delta23"), config, scales_, enable_sanity_checks_, inv_mass_checks_),
      token1_(consumes<P2GTCandidateCollection>(collection1Cuts_.tag())),
      token2_(collection1Cuts_.tag() == collection2Cuts_.tag()
                  ? token1_
                  : consumes<P2GTCandidateCollection>(collection2Cuts_.tag())),
      token3_(collection1Cuts_.tag() == collection3Cuts_.tag()
                  ? token1_
                  : (collection2Cuts_.tag() == collection3Cuts_.tag()
                         ? token2_
                         : consumes<P2GTCandidateCollection>(collection3Cuts_.tag()))) {
  produces<P2GTCandidateVectorRef>(collection1Cuts_.tag().instance());

  if (!(collection1Cuts_.tag() == collection2Cuts_.tag())) {
    produces<P2GTCandidateVectorRef>(collection2Cuts_.tag().instance());
  }

  if (!(collection1Cuts_.tag() == collection3Cuts_.tag()) && !(collection2Cuts_.tag() == collection3Cuts_.tag())) {
    produces<P2GTCandidateVectorRef>(collection3Cuts_.tag().instance());
  }

  if (inv_mass_checks_) {
    produces<InvariantMassErrorCollection>();
  }
}

void L1GTTripleObjectCond::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;

  edm::ParameterSetDescription collection1Desc;
  L1GTSingleCollectionCut::fillPSetDescription(collection1Desc);
  desc.add<edm::ParameterSetDescription>("collection1", collection1Desc);

  edm::ParameterSetDescription collection2Desc;
  L1GTSingleCollectionCut::fillPSetDescription(collection2Desc);
  desc.add<edm::ParameterSetDescription>("collection2", collection2Desc);

  edm::ParameterSetDescription collection3Desc;
  L1GTSingleCollectionCut::fillPSetDescription(collection3Desc);
  desc.add<edm::ParameterSetDescription>("collection3", collection3Desc);

  edm::ParameterSetDescription scalesDesc;
  L1GTScales::fillPSetDescription(scalesDesc);
  desc.add<edm::ParameterSetDescription>("scales", scalesDesc);

  desc.addUntracked<bool>("sanity_checks", false);
  desc.addUntracked<bool>("inv_mass_checks", false);

  edm::ParameterSetDescription delta12Desc;
  L1GTDeltaCut::fillPSetDescription(delta12Desc);
  desc.add<edm::ParameterSetDescription>("delta12", delta12Desc);

  edm::ParameterSetDescription delta13Desc;
  L1GTDeltaCut::fillPSetDescription(delta13Desc);
  desc.add<edm::ParameterSetDescription>("delta13", delta13Desc);

  edm::ParameterSetDescription delta23Desc;
  L1GTDeltaCut::fillPSetDescription(delta23Desc);
  desc.add<edm::ParameterSetDescription>("delta23", delta23Desc);

  L1GTDeltaCut::fillLUTDescriptions(desc);

  descriptions.addWithDefaultLabel(desc);
}

bool L1GTTripleObjectCond::filter(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
  edm::Handle<P2GTCandidateCollection> col1 = event.getHandle(token1_);
  edm::Handle<P2GTCandidateCollection> col2 = event.getHandle(token2_);
  edm::Handle<P2GTCandidateCollection> col3 = event.getHandle(token3_);

  bool condition_result = false;

  std::set<std::size_t> triggeredIdcs1;
  std::set<std::size_t> triggeredIdcs2;
  std::set<std::size_t> triggeredIdcs3;

  InvariantMassErrorCollection massErrors;

  for (std::size_t idx1 = 0; idx1 < col1->size(); ++idx1) {
    for (std::size_t idx2 = 0; idx2 < col2->size(); ++idx2) {
      for (std::size_t idx3 = 0; idx3 < col3->size(); ++idx3) {
        // If we're looking at the same collection then we shouldn't use the same object in one comparison.
        if (col1.product() == col2.product() && idx1 == idx2) {
          continue;
        }

        if (col1.product() == col3.product() && idx1 == idx3) {
          continue;
        }

        if (col2.product() == col3.product() && idx2 == idx3) {
          continue;
        }

        bool pass = true;
        pass &= collection1Cuts_.checkObject(col1->at(idx1));
        pass &= collection2Cuts_.checkObject(col2->at(idx2));
        pass &= collection3Cuts_.checkObject(col3->at(idx3));
        pass &= delta12Cuts_.checkObjects(col1->at(idx1), col2->at(idx2), massErrors);
        pass &= delta13Cuts_.checkObjects(col1->at(idx1), col3->at(idx3), massErrors);
        pass &= delta23Cuts_.checkObjects(col2->at(idx2), col3->at(idx3), massErrors);

        condition_result |= pass;

        if (pass) {
          triggeredIdcs1.emplace(idx1);

          if (col1.product() != col2.product()) {
            triggeredIdcs2.emplace(idx2);
          } else {
            triggeredIdcs1.emplace(idx2);
          }

          if (col1.product() != col3.product() && col2.product() != col3.product()) {
            triggeredIdcs3.emplace(idx3);
          } else if (col1.product() == col3.product()) {
            triggeredIdcs1.emplace(idx3);
          } else {
            triggeredIdcs2.emplace(idx3);
          }
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

    if (col1.product() != col3.product() && col2.product() != col3.product()) {
      std::unique_ptr<P2GTCandidateVectorRef> triggerCol3 = std::make_unique<P2GTCandidateVectorRef>();

      for (std::size_t idx : triggeredIdcs3) {
        triggerCol3->push_back(P2GTCandidateRef(col3, idx));
      }
      event.put(std::move(triggerCol3), collection3Cuts_.tag().instance());
    }
  }

  if (inv_mass_checks_) {
    event.put(std::make_unique<InvariantMassErrorCollection>(std::move(massErrors)), "");
  }

  return condition_result;
}

DEFINE_FWK_MODULE(L1GTTripleObjectCond);
