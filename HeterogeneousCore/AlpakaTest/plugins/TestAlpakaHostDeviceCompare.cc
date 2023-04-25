#include "DataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include <fmt/format.h>

class TestAlpakaHostDeviceCompare : public edm::global::EDAnalyzer<> {
public:
  TestAlpakaHostDeviceCompare(edm::ParameterSet const& iConfig)
      : hostToken_{consumes(iConfig.getUntrackedParameter<edm::InputTag>("srcHost"))},
        deviceToken_{consumes(iConfig.getUntrackedParameter<edm::InputTag>("srcDevice"))},
        expectedXdiff_{iConfig.getUntrackedParameter<double>("expectedXdiff")} {}

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.addUntracked<edm::InputTag>("srcHost");
    desc.addUntracked<edm::InputTag>("srcDevice");
    desc.addUntracked<double>("expectedXdiff", 0.);
    descriptions.addWithDefaultLabel(desc);
  }

  template <typename T>
  static void require(T const& host, T const& device, T const& expectedDiff, std::string_view name) {
    T const diff = host - device;
    if (diff != expectedDiff) {
      throw cms::Exception("Assert") << "Comparison of " << name << " failed, expected difference " << expectedDiff
                                     << " but got " << diff << ", host value " << host << " device value " << device;
    }
  }

  void analyze(edm::StreamID iStream, edm::Event const& iEvent, edm::EventSetup const& iSetup) const override {
    auto const& hostData = iEvent.get(hostToken_);
    auto const& deviceData = iEvent.get(deviceToken_);

    require(hostData->metadata().size(), deviceData->metadata().size(), 0, "metadata().size()");
    auto const hostView = hostData.view();
    auto const deviceView = deviceData.view();
    for (int i = 0; i < hostData->metadata().size(); ++i) {
      require(hostView[i].x(), deviceView[i].x(), expectedXdiff_, fmt::format("view[{}].x()", i));
    }
  }

private:
  edm::EDGetTokenT<portabletest::TestHostCollection> const hostToken_;
  edm::EDGetTokenT<portabletest::TestHostCollection> const deviceToken_;
  double const expectedXdiff_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TestAlpakaHostDeviceCompare);
