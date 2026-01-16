#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDPutToken.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"

#include <memory>
#include <utility>
#include <vector>

namespace edmtest {

  class TestWriteHostHitSoA : public edm::global::EDProducer<> {
  public:
    TestWriteHostHitSoA(edm::ParameterSet const&);
    void produce(edm::StreamID, edm::Event&, edm::EventSetup const&) const override;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

    using HitsOnHost = ::reco::TrackingRecHitHost;

  private:
    const unsigned int hitSize_;
    const unsigned int offsetBPIX2_;
    edm::EDPutTokenT<HitsOnHost> putToken_;
  };

  TestWriteHostHitSoA::TestWriteHostHitSoA(edm::ParameterSet const& iPSet)
      : hitSize_(iPSet.getParameter<unsigned int>("hitSize")),
        offsetBPIX2_(iPSet.getParameter<unsigned int>("offsetBPIX2")),
        putToken_(produces()) {}

  void TestWriteHostHitSoA::produce(edm::StreamID, edm::Event& iEvent, edm::EventSetup const&) const {
    HitsOnHost hits(cms::alpakatools::host(), hitSize_, 100);
    auto hitsView = hits.view();
    for (unsigned int i = 0; i < hitSize_; ++i) {
      hitsView[i].xGlobal() = float(i);
    }
    hitsView.offsetBPIX2() = offsetBPIX2_;
    iEvent.emplace(putToken_, std::move(hits));
  }

  void TestWriteHostHitSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<unsigned int>("hitSize", 1000);
    desc.add<unsigned int>("offsetBPIX2", 50);
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestWriteHostHitSoA;
DEFINE_FWK_MODULE(TestWriteHostHitSoA);
