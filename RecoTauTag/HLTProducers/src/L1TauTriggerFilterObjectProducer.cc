#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "HLTrigger/HLTcore/interface/HLTFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"

class L1TauTriggerFilterObjectProducer : public HLTFilter {
public:
  explicit L1TauTriggerFilterObjectProducer(const edm::ParameterSet& cfg)
      : HLTFilter(cfg),
        tausToken_(consumes<l1t::TauBxCollection>(cfg.getParameter<edm::InputTag>("taus"))),
        selectedBx_(cfg.getParameter<std::vector<int>>("selectedBx")),
        minPt_(cfg.getParameter<double>("minPt")),
        nExpected_(cfg.getParameter<int>("nExpected")) {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    makeHLTFilterDescription(desc);
    desc.add<edm::InputTag>("taus", edm::InputTag("hltGtStage2Digis", "Tau"))->setComment("Input GT stage 2 L1 taus");
    desc.add<std::vector<int>>("selectedBx", std::vector<int>())
        ->setComment("bunch crossings to select, empty means all");
    desc.add<double>("minPt", 0)->setComment("select taus with pt > minPt. minPt=0 means no pt cut");
    desc.add<int>("nExpected", 0)->setComment("minimal number of taus per event to pass the filter");
    descriptions.addWithDefaultLabel(desc);
  }

  bool hltFilter(edm::Event& event,
                 const edm::EventSetup& eventsetup,
                 trigger::TriggerFilterObjectWithRefs& filterproduct) const override {
    const auto& taus = event.getHandle(tausToken_);
    int nPassed = 0;
    for (int bx_index = taus->getFirstBX(); bx_index <= taus->getLastBX(); ++bx_index) {
      if (!selectedBx_.empty() && std::find(selectedBx_.begin(), selectedBx_.end(), bx_index) == selectedBx_.end())
        continue;
      const unsigned bx_index_shift = taus->begin(bx_index) - taus->begin();
      unsigned index_in_bx = 0;
      for (auto it = taus->begin(bx_index); it != taus->end(bx_index); ++it, ++index_in_bx) {
        if (it->pt() <= minPt_)
          continue;
        const l1t::TauRef tauRef(taus, bx_index_shift + index_in_bx);
        filterproduct.addObject(trigger::TriggerL1Tau, tauRef);
        ++nPassed;
      }
    }
    return nPassed >= nExpected_;
  }

private:
  const edm::EDGetTokenT<l1t::TauBxCollection> tausToken_;
  const std::vector<int> selectedBx_;
  const double minPt_;
  const int nExpected_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(L1TauTriggerFilterObjectProducer);
