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
#include "L1GTCorrelationalCut.h"
#include "L1GTSingleInOutLUT.h"
#include "L1GTOptionalParam.h"

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

  const L1GTCorrelationalCut deltaCuts_;

  const std::optional<unsigned int> minQualityScoreSum_;
  const std::optional<unsigned int> maxQualityScoreSum_;

  const edm::EDGetTokenT<P2GTCandidateCollection> token1_;
  const edm::EDGetTokenT<P2GTCandidateCollection> token2_;
  const edm::EDGetTokenT<P2GTCandidateCollection> primVertToken_;
};

L1GTDoubleObjectCond::L1GTDoubleObjectCond(const edm::ParameterSet& config)
    : scales_(config.getParameter<edm::ParameterSet>("scales")),
      collection1Cuts_(config.getParameterSet("collection1"), config, scales_),
      collection2Cuts_(config.getParameterSet("collection2"), config, scales_),
      enable_sanity_checks_(config.getUntrackedParameter<bool>("sanity_checks")),
      inv_mass_checks_(config.getUntrackedParameter<bool>("inv_mass_checks")),
      deltaCuts_(config, config, scales_, enable_sanity_checks_, inv_mass_checks_),
      minQualityScoreSum_(getOptionalParam<unsigned int>("minQualityScoreSum", config)),
      maxQualityScoreSum_(getOptionalParam<unsigned int>("maxQualityScoreSum", config)),
      token1_(consumes<P2GTCandidateCollection>(collection1Cuts_.tag())),
      token2_(collection1Cuts_.tag() == collection2Cuts_.tag()
                  ? token1_
                  : consumes<P2GTCandidateCollection>(collection2Cuts_.tag())),
      primVertToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("primVertTag"))) {
  produces<P2GTCandidateVectorRef>(collection1Cuts_.tag().instance());

  if (!(collection1Cuts_.tag() == collection2Cuts_.tag())) {
    produces<P2GTCandidateVectorRef>(collection2Cuts_.tag().instance());
  }

  if (inv_mass_checks_) {
    produces<InvariantMassErrorCollection>();
  }

  if ((minQualityScoreSum_ || maxQualityScoreSum_) && !(collection1Cuts_.tag() == collection2Cuts_.tag())) {
    throw cms::Exception("Configuration") << "A qualityScore sum can only be calculated within one collection.";
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

  desc.add<edm::InputTag>("primVertTag");

  desc.addUntracked<bool>("sanity_checks", false);
  desc.addUntracked<bool>("inv_mass_checks", false);

  L1GTCorrelationalCut::fillPSetDescription(desc);
  L1GTCorrelationalCut::fillLUTDescriptions(desc);

  desc.addOptional<unsigned int>("minQualityScoreSum");
  desc.addOptional<unsigned int>("maxQualityScoreSum");

  edm::ParameterSetDescription scalesDesc;
  L1GTScales::fillPSetDescription(scalesDesc);
  desc.add<edm::ParameterSetDescription>("scales", scalesDesc);

  descriptions.addWithDefaultLabel(desc);
}

bool L1GTDoubleObjectCond::filter(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
  edm::Handle<P2GTCandidateCollection> col1 = event.getHandle(token1_);
  edm::Handle<P2GTCandidateCollection> col2 = event.getHandle(token2_);
  edm::Handle<P2GTCandidateCollection> primVertCol = event.getHandle(primVertToken_);

  bool condition_result = false;

  std::set<std::size_t> triggeredIdcs1;
  std::set<std::size_t> triggeredIdcs2;

  InvariantMassErrorCollection massErrors;

  for (std::size_t idx1 = 0; idx1 < col1->size(); ++idx1) {
    bool single1Pass = collection1Cuts_.checkObject(col1->at(idx1));
    single1Pass &= collection1Cuts_.checkPrimaryVertices(col1->at(idx1), *primVertCol);

    for (std::size_t idx2 = 0; idx2 < col2->size(); ++idx2) {
      // If we're looking at the same collection then we shouldn't use the same object in one comparison.
      if (col1.product() == col2.product() && idx1 == idx2) {
        continue;
      }

      bool pass = single1Pass;
      pass &= collection2Cuts_.checkObject(col2->at(idx2));
      pass &= collection2Cuts_.checkPrimaryVertices(col2->at(idx2), *primVertCol);
      pass &= deltaCuts_.checkObjects(col1->at(idx1), col2->at(idx2), massErrors);

      if (minQualityScoreSum_ || maxQualityScoreSum_) {
        unsigned int qualityScoreSum =
            col1->at(idx1).hwQualityScore().to_uint() + col2->at(idx2).hwQualityScore().to_uint();

        pass &= minQualityScoreSum_ ? qualityScoreSum > minQualityScoreSum_ : true;
        pass &= maxQualityScoreSum_ ? qualityScoreSum < maxQualityScoreSum_ : true;
      }

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

  condition_result &= collection1Cuts_.checkCollection(*col1);
  condition_result &= collection2Cuts_.checkCollection(*col2);

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
