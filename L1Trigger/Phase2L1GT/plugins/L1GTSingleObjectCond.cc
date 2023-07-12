#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1Trigger/interface/P2GTCandidate.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "DataFormats/Common/interface/Ref.h"

#include "L1Trigger/Phase2L1GT/interface/L1GTScales.h"
#include "L1GTSingleCollectionCut.h"

#include <cmath>
#include <cinttypes>

#include <ap_int.h>

using namespace l1t;

class L1GTSingleObjectCond : public edm::global::EDFilter<> {
public:
  explicit L1GTSingleObjectCond(const edm::ParameterSet&);
  ~L1GTSingleObjectCond() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  bool filter(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;

  const L1GTScales scales_;
  const L1GTSingleCollectionCut collection;

  const edm::EDGetTokenT<P2GTCandidateCollection> token_;
};

L1GTSingleObjectCond::L1GTSingleObjectCond(const edm::ParameterSet& config)
    : scales_(config.getParameter<edm::ParameterSet>("scales")),
      collection(config, config, scales_),
      token_(consumes<P2GTCandidateCollection>(collection.tag())) {
  produces<P2GTCandidateVectorRef>(collection.tag().instance());
}

void L1GTSingleObjectCond::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  L1GTSingleCollectionCut::fillPSetDescription(desc);

  edm::ParameterSetDescription scalesDesc;
  L1GTScales::fillPSetDescription(scalesDesc);
  desc.add<edm::ParameterSetDescription>("scales", scalesDesc);

  descriptions.addWithDefaultLabel(desc);
}

bool L1GTSingleObjectCond::filter(edm::StreamID, edm::Event& event, const edm::EventSetup& setup) const {
  edm::Handle<P2GTCandidateCollection> col = event.getHandle(token_);

  bool condition_result = false;

  std::unique_ptr<P2GTCandidateVectorRef> triggerCol = std::make_unique<P2GTCandidateVectorRef>();

  for (std::size_t idx = 0; idx < col->size(); ++idx) {
    bool pass{collection.checkObject(col->at(idx))};
    condition_result |= pass;

    if (pass) {
      triggerCol->push_back(P2GTCandidateRef(col, idx));
    }
  }

  if (condition_result) {
    event.put(std::move(triggerCol), collection.tag().instance());
  }

  return condition_result;
}

DEFINE_FWK_MODULE(L1GTSingleObjectCond);
