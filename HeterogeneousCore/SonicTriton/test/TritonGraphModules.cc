#include "HeterogeneousCore/SonicTriton/interface/TritonEDProducer.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <sstream>
#include <string>
#include <vector>
#include <random>

class TritonGraphHelper {
public:
  TritonGraphHelper(edm::ParameterSet const& cfg)
      : nodeMin_(cfg.getParameter<unsigned>("nodeMin")),
        nodeMax_(cfg.getParameter<unsigned>("nodeMax")),
        edgeMin_(cfg.getParameter<unsigned>("edgeMin")),
        edgeMax_(cfg.getParameter<unsigned>("edgeMax")),
        brief_(cfg.getParameter<bool>("brief")) {}
  void makeInput(edm::Event const& iEvent, TritonInputMap& iInput) const {
    //get event-based seed for RNG
    unsigned int runNum_uint = static_cast<unsigned int>(iEvent.id().run());
    unsigned int lumiNum_uint = static_cast<unsigned int>(iEvent.id().luminosityBlock());
    unsigned int evNum_uint = static_cast<unsigned int>(iEvent.id().event());
    std::uint32_t seed = (lumiNum_uint << 10) + (runNum_uint << 20) + evNum_uint;
    std::mt19937 rng(seed);

    std::uniform_int_distribution<int> randint1(nodeMin_, nodeMax_);
    int nnodes = randint1(rng);
    std::uniform_int_distribution<int> randint2(edgeMin_, edgeMax_);
    int nedges = randint2(rng);

    //set shapes
    auto& input1 = iInput.at("x__0");
    input1.setShape(0, nnodes);
    auto data1 = input1.allocate<float>();
    auto& vdata1 = (*data1)[0];

    auto& input2 = iInput.at("edgeindex__1");
    input2.setShape(1, nedges);
    auto data2 = input2.allocate<int64_t>();
    auto& vdata2 = (*data2)[0];

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
  void makeOutput(const TritonOutputMap& iOutput, const std::string& debugName) const {
    //check the results
    const auto& output1 = iOutput.begin()->second;
    // convert from server format
    const auto& tmp = output1.fromServer<float>();
    if (brief_)
      edm::LogInfo(debugName) << "output shape: " << output1.shape()[0] << ", " << output1.shape()[1];
    else {
      edm::LogInfo msg(debugName);
      for (int i = 0; i < output1.shape()[0]; ++i) {
        msg << "output " << i << ": ";
        for (int j = 0; j < output1.shape()[1]; ++j) {
          msg << tmp[0][output1.shape()[1] * i + j] << " ";
        }
        msg << "\n";
      }
    }
  }
  static void fillPSetDescription(edm::ParameterSetDescription& desc) {
    desc.add<unsigned>("nodeMin", 100);
    desc.add<unsigned>("nodeMax", 4000);
    desc.add<unsigned>("edgeMin", 8000);
    desc.add<unsigned>("edgeMax", 15000);
    desc.add<bool>("brief", false);
  }

private:
  //members
  unsigned nodeMin_, nodeMax_;
  unsigned edgeMin_, edgeMax_;
  bool brief_;
};

class TritonGraphProducer : public TritonEDProducer<> {
public:
  explicit TritonGraphProducer(edm::ParameterSet const& cfg)
      : TritonEDProducer<>(cfg, "TritonGraphProducer"), helper_(cfg) {}
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    helper_.makeInput(iEvent, iInput);
  }
  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
    helper_.makeOutput(iOutput, debugName_);
  }
  ~TritonGraphProducer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    TritonClient::fillPSetDescription(desc);
    TritonGraphHelper::fillPSetDescription(desc);
    //to ensure distinct cfi names
    descriptions.addWithDefaultLabel(desc);
  }

private:
  //member
  TritonGraphHelper helper_;
};

DEFINE_FWK_MODULE(TritonGraphProducer);

#include "HeterogeneousCore/SonicTriton/interface/TritonEDFilter.h"

class TritonGraphFilter : public TritonEDFilter<> {
public:
  explicit TritonGraphFilter(edm::ParameterSet const& cfg) : TritonEDFilter<>(cfg, "TritonGraphFilter"), helper_(cfg) {}
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    helper_.makeInput(iEvent, iInput);
  }
  bool filter(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
    helper_.makeOutput(iOutput, debugName_);
    return true;
  }
  ~TritonGraphFilter() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    TritonClient::fillPSetDescription(desc);
    TritonGraphHelper::fillPSetDescription(desc);
    //to ensure distinct cfi names
    descriptions.addWithDefaultLabel(desc);
  }

private:
  //member
  TritonGraphHelper helper_;
};

DEFINE_FWK_MODULE(TritonGraphFilter);

#include "HeterogeneousCore/SonicTriton/interface/TritonOneEDAnalyzer.h"

class TritonGraphAnalyzer : public TritonOneEDAnalyzer<> {
public:
  explicit TritonGraphAnalyzer(edm::ParameterSet const& cfg)
      : TritonOneEDAnalyzer<>(cfg, "TritonGraphAnalyzer"), helper_(cfg) {}
  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    helper_.makeInput(iEvent, iInput);
  }
  void analyze(edm::Event const& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
    helper_.makeOutput(iOutput, debugName_);
  }
  ~TritonGraphAnalyzer() override = default;

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    TritonClient::fillPSetDescription(desc);
    TritonGraphHelper::fillPSetDescription(desc);
    //to ensure distinct cfi names
    descriptions.addWithDefaultLabel(desc);
  }

private:
  //member
  TritonGraphHelper helper_;
};

DEFINE_FWK_MODULE(TritonGraphAnalyzer);
