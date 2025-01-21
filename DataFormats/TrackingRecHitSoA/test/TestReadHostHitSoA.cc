#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"

#include <vector>

namespace edmtest {

  class TestReadHostHitSoA : public edm::global::EDAnalyzer<> {
  public:
    TestReadHostHitSoA(edm::ParameterSet const&);
    void analyze(edm::StreamID, edm::Event const&, edm::EventSetup const&) const override;
    void throwWithMessage(const char*) const;
    static void fillDescriptions(edm::ConfigurationDescriptions&);

    using HitsOnHost = ::reco::TrackingRecHitHost;

  private:

    edm::EDGetTokenT<HitsOnHost> getToken_;
  };

  TestReadHostHitSoA::TestReadHostHitSoA(edm::ParameterSet const& iPSet)
      : getToken_(consumes(iPSet.getParameter<edm::InputTag>("input"))) {}

  void TestReadHostHitSoA::analyze(edm::StreamID, edm::Event const& iEvent, edm::EventSetup const&) const {
    auto const& hits = iEvent.get(getToken_);
    auto hitsView = hits.view(); 

    for (int i = 0; i < hitsView.metadata().size(); ++i) {
      if(hitsView[i].xGlobal() != float(i))
      {
        throw cms::Exception("TestWriteHostHitSoA Failure") << "TestReadHostHitSoA::analyze, entry. i = " << i;
      }
    }
  }


  void TestReadHostHitSoA::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("input");
    descriptions.addDefault(desc);
  }
}  // namespace edmtest

using edmtest::TestReadHostHitSoA;
DEFINE_FWK_MODULE(TestReadHostHitSoA);