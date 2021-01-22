#include "HeterogeneousCore/SonicCore/interface/SonicEDProducer.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonClient.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <random>

class TritonGraphProducer : public SonicEDProducer<TritonClient> {
public:
  explicit TritonGraphProducer(edm::ParameterSet const& cfg) : SonicEDProducer<TritonClient>(cfg) {
    //for debugging
    setDebugName("TritonGraphProducer");
  }
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    //get event-based seed for RNG
    unsigned int runNum_uint = static_cast<unsigned int>(iEvent.id().run());
    unsigned int lumiNum_uint = static_cast<unsigned int>(iEvent.id().luminosityBlock());
    unsigned int evNum_uint = static_cast<unsigned int>(iEvent.id().event());
    std::uint32_t seed = (lumiNum_uint << 10) + (runNum_uint << 20) + evNum_uint;
    std::mt19937 rng(seed);

    std::uniform_int_distribution<int> randint1(100, 4000);
    int nnodes = randint1(rng);
    std::uniform_int_distribution<int> randint2(8000, 15000);
    int nedges = randint2(rng);

    //set shapes
    auto& input1 = iInput.at("x__0");
    input1.setShape(0, nnodes);
    auto data1 = std::make_shared<TritonInput<float>>(1);
    auto& vdata1 = (*data1)[0];
    vdata1.reserve(input1.sizeShape());

    auto& input2 = iInput.at("edgeindex__1");
    input2.setShape(1, nedges);
    auto data2 = std::make_shared<TritonInput<int64_t>>(1);
    auto& vdata2 = (*data2)[0];
    vdata2.reserve(input2.sizeShape());

    //fill
    std::normal_distribution<float> randx(-10, 4);
    for (unsigned i = 0; i < input1.sizeShape(); ++i) {
      vdata1.push_back(randx(rng));
    }

    std::uniform_int_distribution<int> randedge(0, nnodes - 1);
    for (unsigned i = 0; i < input2.sizeShape(); ++i) {
      vdata2.push_back(randedge(rng));
    }

    // convert to server format
    input1.toServer(data1);
    input2.toServer(data2);
  }
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
    //check the results
    const auto& output1 = iOutput.begin()->second;
    // convert from server format
    const auto& tmp = output1.fromServer<float>();
    std::stringstream msg;
    for (int i = 0; i < output1.shape()[0]; ++i) {
      msg << "output " << i << ": ";
      for (int j = 0; j < output1.shape()[1]; ++j) {
        msg << tmp[0][output1.shape()[1] * i + j] << " ";
      }
      msg << "\n";
    }
    edm::LogInfo(client_.debugName()) << msg.str();
  }
  ~TritonGraphProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    TritonClient::fillPSetDescription(desc);
    //to ensure distinct cfi names
    descriptions.addWithDefaultLabel(desc);
  }
};

DEFINE_FWK_MODULE(TritonGraphProducer);
