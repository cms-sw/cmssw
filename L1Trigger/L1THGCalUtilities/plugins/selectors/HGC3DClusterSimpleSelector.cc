#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "CommonTools/Utils/interface/StringCutObjectSelector.h"

namespace l1t {
  class HGC3DClusterSimpleSelector : public edm::global::EDProducer<> {
  public:
    explicit HGC3DClusterSimpleSelector(const edm::ParameterSet &);
    ~HGC3DClusterSimpleSelector() override {}
    void produce(edm::StreamID, edm::Event &, const edm::EventSetup &) const override;

  private:
    const edm::EDGetTokenT<l1t::HGCalMulticlusterBxCollection> src_;
    const StringCutObjectSelector<l1t::HGCalMulticluster> cut_;

  };  // class
}  // namespace l1t

l1t::HGC3DClusterSimpleSelector::HGC3DClusterSimpleSelector(const edm::ParameterSet &iConfig)
    : src_(consumes<l1t::HGCalMulticlusterBxCollection>(iConfig.getParameter<edm::InputTag>("src"))),
      cut_(iConfig.getParameter<std::string>("cut")) {
  produces<l1t::HGCalMulticlusterBxCollection>();
}

void l1t::HGC3DClusterSimpleSelector::produce(edm::StreamID, edm::Event &iEvent, const edm::EventSetup &) const {
  std::unique_ptr<l1t::HGCalMulticlusterBxCollection> out = std::make_unique<l1t::HGCalMulticlusterBxCollection>();

  edm::Handle<l1t::HGCalMulticlusterBxCollection> multiclusters;
  iEvent.getByToken(src_, multiclusters);

  for (int bx = multiclusters->getFirstBX(); bx <= multiclusters->getLastBX(); ++bx) {
    for (auto it = multiclusters->begin(bx), ed = multiclusters->end(bx); it != ed; ++it) {
      const auto &c = *it;
      if (cut_(c)) {
        out->push_back(bx, c);
      }
    }
  }

  iEvent.put(std::move(out));
}
using l1t::HGC3DClusterSimpleSelector;
DEFINE_FWK_MODULE(HGC3DClusterSimpleSelector);
