#include "CUDADataFormats/PortableTestObjects/interface/TestHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/StreamID.h"

#include "TestAlgo.h"

class TestPortableProducerCPU : public edm::stream::EDProducer<> {
public:
  TestPortableProducerCPU(edm::ParameterSet const& config)
      : hostToken_{produces()}, size_{config.getParameter<int32_t>("size")} {}

  void produce(edm::Event& event, edm::EventSetup const&) override {
    // run the algorithm
    cudatest::TestHostCollection hostProduct{size_};
    algo_.fill(hostProduct);

    // put the product into the event
    event.emplace(hostToken_, std::move(hostProduct));
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<int32_t>("size");
    descriptions.addWithDefaultLabel(desc);
  }

private:
  const edm::EDPutTokenT<cudatest::TestHostCollection> hostToken_;
  const int32_t size_;

  // implementation of the algorithm
  cudatest::TestAlgo algo_;
};

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(TestPortableProducerCPU);
