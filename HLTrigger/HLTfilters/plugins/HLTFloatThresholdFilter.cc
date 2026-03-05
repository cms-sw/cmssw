#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class HLTFloatThresholdFilter : public edm::global::EDFilter<> {
public:
  explicit HLTFloatThresholdFilter(const edm::ParameterSet& config)
      : src_(config.getParameter<edm::InputTag>("src")),
        threshold_(config.getParameter<double>("threshold")),
        token_(consumes<float>(src_)) {}

  bool filter(edm::StreamID, edm::Event& event, edm::EventSetup const&) const override {
    const auto& handle = event.getHandle(token_);
    return (handle.isValid() && *handle > threshold_);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  const edm::InputTag src_;
  const double threshold_;
  const edm::EDGetTokenT<float> token_;
};

void HLTFloatThresholdFilter::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<edm::InputTag>("src", edm::InputTag(""));
  desc.add<double>("threshold", -99.);
  descriptions.addWithDefaultLabel(desc);
}

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTFloatThresholdFilter);
