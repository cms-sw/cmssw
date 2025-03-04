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
#include "L1GTCorrelationalCut.h"
#include "L1GT3BodyCut.h"
#include "L1GTSingleInOutLUT.h"
#include "L1GTOptionalParam.h"

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

  const L1GTCorrelationalCut correl12Cuts_;
  const L1GTCorrelationalCut correl13Cuts_;
  const L1GTCorrelationalCut correl23Cuts_;

  const L1GT3BodyCut correl123Cuts_;

  const std::optional<unsigned int> minQualityScoreSum_;
  const std::optional<unsigned int> maxQualityScoreSum_;

  const edm::EDGetTokenT<P2GTCandidateCollection> token1_;
  const edm::EDGetTokenT<P2GTCandidateCollection> token2_;
  const edm::EDGetTokenT<P2GTCandidateCollection> token3_;
  const edm::EDGetTokenT<P2GTCandidateCollection> primVertToken_;
};

L1GTTripleObjectCond::L1GTTripleObjectCond(const edm::ParameterSet& config)
    : scales_(config.getParameter<edm::ParameterSet>("scales")),
      collection1Cuts_(config.getParameter<edm::ParameterSet>("collection1"), config, scales_),
      collection2Cuts_(config.getParameter<edm::ParameterSet>("collection2"), config, scales_),
      collection3Cuts_(config.getParameter<edm::ParameterSet>("collection3"), config, scales_),
      enable_sanity_checks_(config.getUntrackedParameter<bool>("sanity_checks")),
      inv_mass_checks_(config.getUntrackedParameter<bool>("inv_mass_checks")),
      correl12Cuts_(
          config.getParameter<edm::ParameterSet>("correl12"), config, scales_, enable_sanity_checks_, inv_mass_checks_),
      correl13Cuts_(
          config.getParameter<edm::ParameterSet>("correl13"), config, scales_, enable_sanity_checks_, inv_mass_checks_),
      correl23Cuts_(
          config.getParameter<edm::ParameterSet>("correl23"), config, scales_, enable_sanity_checks_, inv_mass_checks_),
      correl123Cuts_(config, config, scales_, inv_mass_checks_),
      minQualityScoreSum_(getOptionalParam<unsigned int>("minQualityScoreSum", config)),
      maxQualityScoreSum_(getOptionalParam<unsigned int>("maxQualityScoreSum", config)),
      token1_(consumes<P2GTCandidateCollection>(collection1Cuts_.tag())),
      token2_(collection1Cuts_.tag() == collection2Cuts_.tag()
                  ? token1_
                  : consumes<P2GTCandidateCollection>(collection2Cuts_.tag())),
      token3_(collection1Cuts_.tag() == collection3Cuts_.tag()
                  ? token1_
                  : (collection2Cuts_.tag() == collection3Cuts_.tag()
                         ? token2_
                         : consumes<P2GTCandidateCollection>(collection3Cuts_.tag()))),
      primVertToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("primVertTag"))) {
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

  if ((minQualityScoreSum_ || maxQualityScoreSum_) &&
      !(collection1Cuts_.tag() == collection2Cuts_.tag() && collection2Cuts_.tag() == collection3Cuts_.tag())) {
    throw cms::Exception("Configuration") << "A qualityScore sum can only be calculated within one collection.";
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

  desc.add<edm::InputTag>("primVertTag");

  desc.addUntracked<bool>("sanity_checks", false);
  desc.addUntracked<bool>("inv_mass_checks", false);

  edm::ParameterSetDescription correl12Desc;
  L1GTCorrelationalCut::fillPSetDescription(correl12Desc);
  desc.add<edm::ParameterSetDescription>("correl12", correl12Desc);

  edm::ParameterSetDescription correl13Desc;
  L1GTCorrelationalCut::fillPSetDescription(correl13Desc);
  desc.add<edm::ParameterSetDescription>("correl13", correl13Desc);

  edm::ParameterSetDescription correl23Desc;
  L1GTCorrelationalCut::fillPSetDescription(correl23Desc);
  desc.add<edm::ParameterSetDescription>("correl23", correl23Desc);

  L1GT3BodyCut::fillPSetDescription(desc);

  desc.addOptional<unsigned int>("minQualityScoreSum");
  desc.addOptional<unsigned int>("maxQualityScoreSum");

  L1GTCorrelationalCut::fillLUTDescriptions(desc);

  descriptions.addWithDefaultLabel(desc);
}

bool L1GTTripleObjectCond::filter(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
  edm::Handle<P2GTCandidateCollection> col1 = event.getHandle(token1_);
  edm::Handle<P2GTCandidateCollection> col2 = event.getHandle(token2_);
  edm::Handle<P2GTCandidateCollection> col3 = event.getHandle(token3_);
  edm::Handle<P2GTCandidateCollection> primVertCol = event.getHandle(primVertToken_);

  bool condition_result = false;

  std::set<std::size_t> triggeredIdcs1;
  std::set<std::size_t> triggeredIdcs2;
  std::set<std::size_t> triggeredIdcs3;

  InvariantMassErrorCollection massErrors;

  for (std::size_t idx1 = 0; idx1 < col1->size(); ++idx1) {
    bool single1Pass = collection1Cuts_.checkObject(col1->at(idx1));
    single1Pass &= collection1Cuts_.checkPrimaryVertices(col1->at(idx1), *primVertCol);

    for (std::size_t idx2 = 0; idx2 < col2->size(); ++idx2) {
      bool single2Pass = collection2Cuts_.checkObject(col2->at(idx2));
      single2Pass &= collection2Cuts_.checkPrimaryVertices(col2->at(idx2), *primVertCol);

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

        bool pass = single1Pass & single2Pass;

        pass &= collection3Cuts_.checkObject(col3->at(idx3));
        pass &= collection3Cuts_.checkPrimaryVertices(col3->at(idx3), *primVertCol);
        pass &= correl12Cuts_.checkObjects(col1->at(idx1), col2->at(idx2), massErrors);
        pass &= correl13Cuts_.checkObjects(col1->at(idx1), col3->at(idx3), massErrors);
        pass &= correl23Cuts_.checkObjects(col2->at(idx2), col3->at(idx3), massErrors);
        pass &= correl123Cuts_.checkObjects(col1->at(idx1), col2->at(idx2), col3->at(idx3), massErrors);

        if (minQualityScoreSum_ || maxQualityScoreSum_) {
          unsigned int qualityScoreSum = col1->at(idx1).hwQualityScore().to_uint() +
                                         col2->at(idx2).hwQualityScore().to_uint() +
                                         col3->at(idx3).hwQualityScore().to_uint();

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

  condition_result &= collection1Cuts_.checkCollection(*col1);
  condition_result &= collection2Cuts_.checkCollection(*col2);
  condition_result &= collection3Cuts_.checkCollection(*col3);

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
