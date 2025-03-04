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

#include "L1Trigger/Phase2L1GT/interface/L1GTScales.h"
#include "L1GTSingleCollectionCut.h"
#include "L1GTCorrelationalCut.h"
#include "L1GT3BodyCut.h"
#include "L1GTSingleInOutLUT.h"
#include "L1GTOptionalParam.h"

#include <cinttypes>
#include <memory>
#include <vector>
#include <set>

#include <ap_int.h>

using namespace l1t;

class L1GTQuadObjectCond : public edm::global::EDFilter<> {
public:
  explicit L1GTQuadObjectCond(const edm::ParameterSet&);
  ~L1GTQuadObjectCond() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  const L1GTScales scales_;

  const L1GTSingleCollectionCut collection1Cuts_;
  const L1GTSingleCollectionCut collection2Cuts_;
  const L1GTSingleCollectionCut collection3Cuts_;
  const L1GTSingleCollectionCut collection4Cuts_;

  const bool enable_sanity_checks_;
  const bool inv_mass_checks_;

  const L1GTCorrelationalCut correl12Cuts_;
  const L1GTCorrelationalCut correl13Cuts_;
  const L1GTCorrelationalCut correl23Cuts_;
  const L1GTCorrelationalCut correl14Cuts_;
  const L1GTCorrelationalCut correl24Cuts_;
  const L1GTCorrelationalCut correl34Cuts_;

  const L1GT3BodyCut correl123Cuts_;
  const L1GT3BodyCut correl124Cuts_;
  const L1GT3BodyCut correl134Cuts_;
  const L1GT3BodyCut correl234Cuts_;

  const std::optional<unsigned int> minQualityScoreSum_;
  const std::optional<unsigned int> maxQualityScoreSum_;

  const edm::EDGetTokenT<P2GTCandidateCollection> token1_;
  const edm::EDGetTokenT<P2GTCandidateCollection> token2_;
  const edm::EDGetTokenT<P2GTCandidateCollection> token3_;
  const edm::EDGetTokenT<P2GTCandidateCollection> token4_;
  const edm::EDGetTokenT<P2GTCandidateCollection> primVertToken_;
};

L1GTQuadObjectCond::L1GTQuadObjectCond(const edm::ParameterSet& config)
    : scales_(config.getParameter<edm::ParameterSet>("scales")),
      collection1Cuts_(config.getParameter<edm::ParameterSet>("collection1"), config, scales_),
      collection2Cuts_(config.getParameter<edm::ParameterSet>("collection2"), config, scales_),
      collection3Cuts_(config.getParameter<edm::ParameterSet>("collection3"), config, scales_),
      collection4Cuts_(config.getParameter<edm::ParameterSet>("collection4"), config, scales_),
      enable_sanity_checks_(config.getUntrackedParameter<bool>("sanity_checks")),
      inv_mass_checks_(config.getUntrackedParameter<bool>("inv_mass_checks")),
      correl12Cuts_(
          config.getParameter<edm::ParameterSet>("correl12"), config, scales_, enable_sanity_checks_, inv_mass_checks_),
      correl13Cuts_(
          config.getParameter<edm::ParameterSet>("correl13"), config, scales_, enable_sanity_checks_, inv_mass_checks_),
      correl23Cuts_(
          config.getParameter<edm::ParameterSet>("correl23"), config, scales_, enable_sanity_checks_, inv_mass_checks_),
      correl14Cuts_(
          config.getParameter<edm::ParameterSet>("correl14"), config, scales_, enable_sanity_checks_, inv_mass_checks_),
      correl24Cuts_(
          config.getParameter<edm::ParameterSet>("correl24"), config, scales_, enable_sanity_checks_, inv_mass_checks_),
      correl34Cuts_(
          config.getParameter<edm::ParameterSet>("correl34"), config, scales_, enable_sanity_checks_, inv_mass_checks_),
      correl123Cuts_(config.getParameter<edm::ParameterSet>("correl123"), config, scales_, inv_mass_checks_),
      correl124Cuts_(config.getParameter<edm::ParameterSet>("correl124"), config, scales_, inv_mass_checks_),
      correl134Cuts_(config.getParameter<edm::ParameterSet>("correl134"), config, scales_, inv_mass_checks_),
      correl234Cuts_(config.getParameter<edm::ParameterSet>("correl234"), config, scales_, inv_mass_checks_),
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
      token4_(collection1Cuts_.tag() == collection4Cuts_.tag()
                  ? token1_
                  : (collection2Cuts_.tag() == collection4Cuts_.tag()
                         ? token2_
                         : (collection3Cuts_.tag() == collection4Cuts_.tag()
                                ? token3_
                                : consumes<P2GTCandidateCollection>(collection4Cuts_.tag())))),
      primVertToken_(consumes<P2GTCandidateCollection>(config.getParameter<edm::InputTag>("primVertTag"))) {
  produces<P2GTCandidateVectorRef>(collection1Cuts_.tag().instance());

  if (!(collection1Cuts_.tag() == collection2Cuts_.tag())) {
    produces<P2GTCandidateVectorRef>(collection2Cuts_.tag().instance());
  }

  if (!(collection1Cuts_.tag() == collection3Cuts_.tag()) && !(collection2Cuts_.tag() == collection3Cuts_.tag())) {
    produces<P2GTCandidateVectorRef>(collection3Cuts_.tag().instance());
  }

  if (!(collection1Cuts_.tag() == collection4Cuts_.tag()) && !(collection2Cuts_.tag() == collection4Cuts_.tag()) &&
      !(collection3Cuts_.tag() == collection4Cuts_.tag())) {
    produces<P2GTCandidateVectorRef>(collection4Cuts_.tag().instance());
  }

  if (inv_mass_checks_) {
    produces<InvariantMassErrorCollection>();
  }

  if ((minQualityScoreSum_ || maxQualityScoreSum_) &&
      !(collection1Cuts_.tag() == collection2Cuts_.tag() && collection2Cuts_.tag() == collection3Cuts_.tag() &&
        collection3Cuts_.tag() == collection4Cuts_.tag())) {
    throw cms::Exception("Configuration") << "A qualityScore sum can only be calculated within one collection.";
  }
}

void L1GTQuadObjectCond::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
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

  edm::ParameterSetDescription collection4Desc;
  L1GTSingleCollectionCut::fillPSetDescription(collection4Desc);
  desc.add<edm::ParameterSetDescription>("collection4", collection4Desc);

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

  edm::ParameterSetDescription correl14Desc;
  L1GTCorrelationalCut::fillPSetDescription(correl14Desc);
  desc.add<edm::ParameterSetDescription>("correl14", correl14Desc);

  edm::ParameterSetDescription correl24Desc;
  L1GTCorrelationalCut::fillPSetDescription(correl24Desc);
  desc.add<edm::ParameterSetDescription>("correl24", correl24Desc);

  edm::ParameterSetDescription correl34Desc;
  L1GTCorrelationalCut::fillPSetDescription(correl34Desc);
  desc.add<edm::ParameterSetDescription>("correl34", correl34Desc);

  edm::ParameterSetDescription correl123Desc;
  L1GT3BodyCut::fillPSetDescription(correl123Desc);
  desc.add<edm::ParameterSetDescription>("correl123", correl123Desc);

  edm::ParameterSetDescription correl124Desc;
  L1GT3BodyCut::fillPSetDescription(correl124Desc);
  desc.add<edm::ParameterSetDescription>("correl124", correl124Desc);

  edm::ParameterSetDescription correl134Desc;
  L1GT3BodyCut::fillPSetDescription(correl134Desc);
  desc.add<edm::ParameterSetDescription>("correl134", correl134Desc);

  edm::ParameterSetDescription correl234Desc;
  L1GT3BodyCut::fillPSetDescription(correl234Desc);
  desc.add<edm::ParameterSetDescription>("correl234", correl234Desc);

  desc.addOptional<unsigned int>("minQualityScoreSum");
  desc.addOptional<unsigned int>("maxQualityScoreSum");

  L1GTCorrelationalCut::fillLUTDescriptions(desc);

  descriptions.addWithDefaultLabel(desc);
}

bool L1GTQuadObjectCond::filter(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
  edm::Handle<P2GTCandidateCollection> col1 = event.getHandle(token1_);
  edm::Handle<P2GTCandidateCollection> col2 = event.getHandle(token2_);
  edm::Handle<P2GTCandidateCollection> col3 = event.getHandle(token3_);
  edm::Handle<P2GTCandidateCollection> col4 = event.getHandle(token4_);
  edm::Handle<P2GTCandidateCollection> primVertCol = event.getHandle(primVertToken_);

  bool condition_result = false;

  std::set<std::size_t> triggeredIdcs1;
  std::set<std::size_t> triggeredIdcs2;
  std::set<std::size_t> triggeredIdcs3;
  std::set<std::size_t> triggeredIdcs4;

  InvariantMassErrorCollection massErrors;

  for (std::size_t idx1 = 0; idx1 < col1->size(); ++idx1) {
    bool single1Pass = collection1Cuts_.checkObject(col1->at(idx1));
    single1Pass &= collection1Cuts_.checkPrimaryVertices(col1->at(idx1), *primVertCol);

    for (std::size_t idx2 = 0; idx2 < col2->size(); ++idx2) {
      bool single2Pass = collection2Cuts_.checkObject(col2->at(idx2));
      single2Pass &= collection2Cuts_.checkPrimaryVertices(col2->at(idx2), *primVertCol);

      for (std::size_t idx3 = 0; idx3 < col3->size(); ++idx3) {
        bool single3Pass = collection3Cuts_.checkObject(col3->at(idx3));
        single3Pass &= collection3Cuts_.checkPrimaryVertices(col3->at(idx3), *primVertCol);

        for (std::size_t idx4 = 0; idx4 < col4->size(); ++idx4) {
          // If we're looking at the same collection then we shouldn't use the same object in one comparison.
          if (col1.product() == col2.product() && idx1 == idx2) {
            continue;
          }

          if (col2.product() == col3.product() && idx2 == idx3) {
            continue;
          }

          if (col1.product() == col3.product() && idx1 == idx3) {
            continue;
          }

          if (col1.product() == col4.product() && idx1 == idx4) {
            continue;
          }

          if (col2.product() == col4.product() && idx2 == idx4) {
            continue;
          }

          if (col3.product() == col4.product() && idx3 == idx4) {
            continue;
          }

          bool pass = single1Pass & single2Pass & single3Pass;

          pass &= collection4Cuts_.checkObject(col4->at(idx4));
          pass &= collection4Cuts_.checkPrimaryVertices(col4->at(idx4), *primVertCol);
          pass &= correl12Cuts_.checkObjects(col1->at(idx1), col2->at(idx2), massErrors);
          pass &= correl13Cuts_.checkObjects(col1->at(idx1), col3->at(idx3), massErrors);
          pass &= correl23Cuts_.checkObjects(col2->at(idx2), col3->at(idx3), massErrors);
          pass &= correl14Cuts_.checkObjects(col1->at(idx1), col4->at(idx4), massErrors);
          pass &= correl24Cuts_.checkObjects(col2->at(idx2), col4->at(idx4), massErrors);
          pass &= correl34Cuts_.checkObjects(col3->at(idx3), col4->at(idx4), massErrors);
          pass &= correl123Cuts_.checkObjects(col1->at(idx1), col2->at(idx2), col3->at(idx3), massErrors);
          pass &= correl124Cuts_.checkObjects(col1->at(idx1), col2->at(idx2), col4->at(idx4), massErrors);
          pass &= correl134Cuts_.checkObjects(col1->at(idx1), col3->at(idx3), col4->at(idx4), massErrors);
          pass &= correl234Cuts_.checkObjects(col2->at(idx2), col3->at(idx3), col4->at(idx4), massErrors);

          if (minQualityScoreSum_ || maxQualityScoreSum_) {
            unsigned int qualityScoreSum =
                col1->at(idx1).hwQualityScore().to_uint() + col2->at(idx2).hwQualityScore().to_uint() +
                col3->at(idx3).hwQualityScore().to_uint() + col4->at(idx4).hwQualityScore().to_uint();

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

            if (col1.product() != col4.product() && col2.product() != col4.product() &&
                col3.product() != col4.product()) {
              triggeredIdcs4.emplace(idx4);
            } else if (col1.product() == col4.product()) {
              triggeredIdcs1.emplace(idx4);
            } else if (col2.product() == col4.product()) {
              triggeredIdcs2.emplace(idx4);
            } else {
              triggeredIdcs3.emplace(idx4);
            }
          }
        }
      }
    }
  }

  condition_result &= collection1Cuts_.checkCollection(*col1);
  condition_result &= collection2Cuts_.checkCollection(*col2);
  condition_result &= collection3Cuts_.checkCollection(*col3);
  condition_result &= collection4Cuts_.checkCollection(*col4);

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

    if (col1.product() != col4.product() && col2.product() != col4.product() && col3.product() != col4.product()) {
      std::unique_ptr<P2GTCandidateVectorRef> triggerCol4 = std::make_unique<P2GTCandidateVectorRef>();

      for (std::size_t idx : triggeredIdcs4) {
        triggerCol4->push_back(P2GTCandidateRef(col4, idx));
      }
      event.put(std::move(triggerCol4), collection4Cuts_.tag().instance());
    }
  }

  if (inv_mass_checks_) {
    event.put(std::make_unique<InvariantMassErrorCollection>(std::move(massErrors)), "");
  }

  return condition_result;
}

DEFINE_FWK_MODULE(L1GTQuadObjectCond);
