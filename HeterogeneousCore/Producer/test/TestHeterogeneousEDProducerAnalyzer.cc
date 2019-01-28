#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/transform.h"

#include "HeterogeneousCore/Product/interface/HeterogeneousProduct.h"

#include <vector>

class TestHeterogeneousEDProducerAnalyzer: public edm::global::EDAnalyzer<> {
public:
  explicit TestHeterogeneousEDProducerAnalyzer(edm::ParameterSet const& iConfig);
  ~TestHeterogeneousEDProducerAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

private:
  void analyze(edm::StreamID streamID, const edm::Event& iEvent, const edm::EventSetup& iSetup) const override;

  using InputType = HeterogeneousProductImpl<heterogeneous::CPUProduct<unsigned int>,
                                             heterogeneous::GPUCudaProduct<std::pair<float *, float *>>>;
  std::string label_;
  std::vector<edm::EDGetTokenT<HeterogeneousProduct>> srcTokens_;
};

TestHeterogeneousEDProducerAnalyzer::TestHeterogeneousEDProducerAnalyzer(const edm::ParameterSet& iConfig):
  label_(iConfig.getParameter<std::string>("@module_label")),
  srcTokens_(edm::vector_transform(iConfig.getParameter<std::vector<edm::InputTag> >("src"),
                                   [this](const edm::InputTag& tag) {
                                     return consumes<HeterogeneousProduct>(tag);
                                   }))
{}

void TestHeterogeneousEDProducerAnalyzer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  edm::ParameterSetDescription desc;
  desc.add<std::vector<edm::InputTag>>("src", std::vector<edm::InputTag>{});
  descriptions.add("testHeterogeneousEDProducerAnalyzer", desc);
}

void TestHeterogeneousEDProducerAnalyzer::analyze(edm::StreamID streamID, const edm::Event& iEvent, const edm::EventSetup& iSetup) const {
  edm::Handle<HeterogeneousProduct> hinput;
  int inp=0;
  for(const auto& token: srcTokens_) {
    iEvent.getByToken(token, hinput);
    edm::LogPrint("TestHeterogeneousEDProducerAnalyzer") << "Analyzer event " << iEvent.id().event()
                                                    << " stream " << streamID
                                                    << " label " << label_
                                                    << " coll " << inp
                                                    << " result " << hinput->get<InputType>().getProduct<HeterogeneousDevice::kCPU>();
    ++inp;
  }
}

DEFINE_FWK_MODULE(TestHeterogeneousEDProducerAnalyzer);
