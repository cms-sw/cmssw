// -*- C++ -*-
//
// Package:    PhysicsTools/NanoAOD
// Class:      L1TriggerResultsConverter
//
/**\class L1TriggerResultsConverter L1TriggerResultsConverter.cc PhysicsTools/L1TriggerResultsConverter/plugins/L1TriggerResultsConverter.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Andrea Rizzi
//         Created:  Mon, 11 Aug 2017 11:20:30 GMT
//
//

// system include files
#include <memory>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMenuRcd.h"
#include "CondFormats/DataRecord/interface/L1GtTriggerMaskAlgoTrigRcd.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMenu.h"
#include "CondFormats/L1TObjects/interface/L1GtTriggerMask.h"
#include "CondFormats/L1TObjects/interface/L1TUtmTriggerMenu.h"
#include "CondFormats/DataRecord/interface/L1TUtmTriggerMenuRcd.h"

#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
#include "DataFormats/L1TGlobal/interface/GlobalExtBlk.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "DataFormats/Common/interface/TriggerResults.h"

#include "L1Trigger/GlobalTriggerAnalyzer/interface/L1GtUtils.h"
//
// class declaration
//

class L1TriggerResultsConverter : public edm::stream::EDProducer<> {
public:
  explicit L1TriggerResultsConverter(const edm::ParameterSet&);
  ~L1TriggerResultsConverter() override;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void produce(edm::Event&, const edm::EventSetup&) override;
  void beginRun(edm::Run const&, edm::EventSetup const&) override;

  // ----------member data ---------------------------
  const bool legacyL1_;
  const bool store_unprefireable_bits_;
  const edm::EDGetTokenT<L1GlobalTriggerReadoutRecord> tokenLegacy_;
  const edm::EDGetTokenT<GlobalAlgBlkBxCollection> token_;
  const edm::EDGetTokenT<GlobalExtBlkBxCollection> token_ext_;
  edm::ESGetToken<L1GtTriggerMenu, L1GtTriggerMenuRcd> l1gtmenuToken_;
  edm::ESGetToken<L1GtTriggerMask, L1GtTriggerMaskAlgoTrigRcd> l1gtalgoMaskToken_;
  edm::ESGetToken<L1TUtmTriggerMenu, L1TUtmTriggerMenuRcd> l1utmTrigToken_;
  std::vector<std::string> names_;
  std::vector<unsigned int> mask_;
  std::vector<unsigned int> indices_;
};

//
// constructors and destructor
//
L1TriggerResultsConverter::L1TriggerResultsConverter(const edm::ParameterSet& params)
    : legacyL1_(params.getParameter<bool>("legacyL1")),
      store_unprefireable_bits_(!legacyL1_ ? params.getParameter<bool>("storeUnprefireableBits") : false),
      tokenLegacy_(legacyL1_ ? consumes<L1GlobalTriggerReadoutRecord>(params.getParameter<edm::InputTag>("src"))
                             : edm::EDGetTokenT<L1GlobalTriggerReadoutRecord>()),
      token_(!legacyL1_ ? consumes<GlobalAlgBlkBxCollection>(params.getParameter<edm::InputTag>("src"))
                        : edm::EDGetTokenT<GlobalAlgBlkBxCollection>()),
      token_ext_(store_unprefireable_bits_
                     ? consumes<GlobalExtBlkBxCollection>(params.getParameter<edm::InputTag>("src_ext"))
                     : edm::EDGetTokenT<GlobalExtBlkBxCollection>()),
      l1gtmenuToken_(esConsumes<edm::Transition::BeginRun>()),
      l1gtalgoMaskToken_(esConsumes<edm::Transition::BeginRun>()),
      l1utmTrigToken_(esConsumes<edm::Transition::BeginRun>()) {
  produces<edm::TriggerResults>();
}

L1TriggerResultsConverter::~L1TriggerResultsConverter() {
  // do anything here that needs to be done at destruction time
  // (e.g. close files, deallocate resources etc.)
}

//
// member functions
//

void L1TriggerResultsConverter::beginRun(edm::Run const&, edm::EventSetup const& setup) {
  mask_.clear();
  names_.clear();
  indices_.clear();
  if (legacyL1_) {
    auto const& mapping = setup.getHandle(l1gtmenuToken_)->gtAlgorithmAliasMap();
    for (auto const& keyval : mapping) {
      names_.push_back(keyval.first);
      indices_.push_back(keyval.second.algoBitNumber());
    }
    mask_ = setup.getHandle(l1gtalgoMaskToken_)->gtTriggerMask();
  } else {
    auto const& mapping = setup.getHandle(l1utmTrigToken_)->getAlgorithmMap();
    for (auto const& keyval : mapping) {
      names_.push_back(keyval.first);
      indices_.push_back(keyval.second.getIndex());
    }
    if (store_unprefireable_bits_) {
      names_.push_back("L1_UnprefireableEvent_TriggerRules");
      names_.push_back("L1_UnprefireableEvent_FirstBxInTrain");
      names_.push_back("L1_FinalOR_BXmin1");
      names_.push_back("L1_FinalOR_BXmin2");
    }
  }
}

// ------------ method called to produce the data  ------------

void L1TriggerResultsConverter::produce(edm::Event& iEvent, const edm::EventSetup& iSetup) {
  const std::vector<bool>* wordp = nullptr;
  const std::vector<bool>* wordp_bxmin1 = nullptr;
  const std::vector<bool>* wordp_bxmin2 = nullptr;
  bool unprefireable_bit_triggerrules = false;
  bool unprefireable_bit_firstbxintrain = false;
  bool l1FinalOR_bxmin1 = false;
  bool l1FinalOR_bxmin2 = false;

  if (!legacyL1_) {
    const auto& resultsProd = iEvent.get(token_);
    if (not resultsProd.isEmpty(0)) {
      wordp = &resultsProd.at(0, 0).getAlgoDecisionFinal();
    }
    if (not resultsProd.isEmpty(-1)) {
      wordp_bxmin1 = &resultsProd.at(-1, 0).getAlgoDecisionFinal();
    }
    if (not resultsProd.isEmpty(-2)) {
      wordp_bxmin2 = &resultsProd.at(-2, 0).getAlgoDecisionFinal();
    }
    if (store_unprefireable_bits_) {
      auto handleExtResults = iEvent.getHandle(token_ext_);
      if (handleExtResults.isValid()) {
        if (not handleExtResults->isEmpty(0)) {
          //Stores the unprefirable event decision corresponding to trigger rules (e.g.: BX0 is unprefirable because BX-3 fired and therefore BX-2/-1 are masked).
          unprefireable_bit_triggerrules =
              handleExtResults->at(0, 0).getExternalDecision(GlobalExtBlk::maxExternalConditions - 1);
        }
      } else {
        LogDebug("Unprefirable bit (trigger rules) not found, always set to false");
      }
    }
  } else {
    // Legacy access
    const auto& resultsProd = iEvent.get(tokenLegacy_);
    wordp = &resultsProd.decisionWord();
  }
  edm::HLTGlobalStatus l1bitsAsHLTStatus(names_.size());
  unsigned indices_size = indices_.size();
  for (size_t nidx = 0; nidx < indices_size; nidx++) {
    unsigned int const index = indices_[nidx];
    bool result = wordp ? wordp->at(index) : false;
    bool result_bxmin1 = wordp_bxmin1 ? wordp_bxmin1->at(index) : false;
    bool result_bxmin2 = wordp_bxmin2 ? wordp_bxmin2->at(index) : false;
    if (not mask_.empty())
      result &= (mask_.at(index) != 0);
    l1bitsAsHLTStatus[nidx] = edm::HLTPathStatus(result ? edm::hlt::Pass : edm::hlt::Fail);
    //Stores the unprefirable event decision corresponding to events in the first bx of a train.
    //In 2022/2023 the bx before that was manually masked.
    //Technically this was done by enabling the L1_FirstBunchBeforeTrain bit in the L1 menu, and vetoing that bit after L1 Final OR is evaluated.
    if (!legacyL1_) {
      if (names_[nidx] == "L1_FirstBunchBeforeTrain")
        unprefireable_bit_firstbxintrain = result_bxmin1;
      //Checks if any other seed was fired in BX-1 or -2
      else if (result_bxmin1) {
        l1FinalOR_bxmin1 = true;
      } else if (result_bxmin2) {
        l1FinalOR_bxmin2 = true;
      }
    }
  }
  if (store_unprefireable_bits_) {
    l1bitsAsHLTStatus[indices_size] =
        edm::HLTPathStatus(unprefireable_bit_triggerrules ? edm::hlt::Pass : edm::hlt::Fail);
    l1bitsAsHLTStatus[indices_size + 1] =
        edm::HLTPathStatus(unprefireable_bit_firstbxintrain ? edm::hlt::Pass : edm::hlt::Fail);
    l1bitsAsHLTStatus[indices_size + 2] = edm::HLTPathStatus(l1FinalOR_bxmin1 ? edm::hlt::Pass : edm::hlt::Fail);
    l1bitsAsHLTStatus[indices_size + 3] = edm::HLTPathStatus(l1FinalOR_bxmin2 ? edm::hlt::Pass : edm::hlt::Fail);
  }
  //mimic HLT trigger bits for L1
  auto out = std::make_unique<edm::TriggerResults>(l1bitsAsHLTStatus, names_);
  iEvent.put(std::move(out));
}

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
void L1TriggerResultsConverter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<bool>("legacyL1")->setComment("is legacy L1");
  desc.add<edm::InputTag>("src")->setComment(
      "L1 input (L1GlobalTriggerReadoutRecord if legacy, GlobalAlgBlkBxCollection otherwise)");
  desc.add<bool>("storeUnprefireableBits", false)
      ->setComment("Activate storage of L1 unprefireable bits (needs L1 external decision input)");
  desc.add<edm::InputTag>("src_ext", edm::InputTag(""))
      ->setComment("L1 external decision input (GlobalExtBlkBxCollection, only supported if not legacy");
  descriptions.add("L1TriggerResultsConverter", desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(L1TriggerResultsConverter);
