#include "DataFormats/Common/interface/Handle.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/stream/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

class HLTFloatThresholdFilter : public edm::stream::EDFilter<> {
 public:
  explicit HLTFloatThresholdFilter(const edm::ParameterSet& config)
      : src_(config.getParameter<edm::InputTag>("src")),
        threshold_(config.getParameter<double>("threshold")) {
    token_ = consumes<float>(src_);
  }

  bool filter(edm::Event& event, const edm::EventSetup&) override {
    edm::Handle<float> handle;
    event.getByToken(token_, handle);
    return (handle.isValid() && *handle > threshold_);
  }

 private:
  edm::InputTag src_;
  edm::EDGetTokenT<float> token_;
  double threshold_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(HLTFloatThresholdFilter);
