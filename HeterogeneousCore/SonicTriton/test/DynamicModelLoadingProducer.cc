#include "HeterogeneousCore/SonicTriton/interface/TritonEDProducer.h"
#include "HeterogeneousCore/SonicTriton/interface/TritonService.h"
#include "DataFormats/TestObjects/interface/ToyProducts.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

#include <memory>
#include <atomic>
#include <chrono>
#include <thread>

// Test module that explicitly exercises dynamic model loading
// This tests the reference counting and thread safety of loadModel/unloadModel
class DynamicModelLoadingProducer : public TritonEDProducer<> {
public:
  explicit DynamicModelLoadingProducer(edm::ParameterSet const& cfg)
      : TritonEDProducer<>(cfg),
        loadUnloadCycles_(cfg.getParameter<int>("loadUnloadCycles")),
        testConcurrency_(cfg.getParameter<bool>("testConcurrency")) {
    putToken_ = produces<edmtest::IntProduct>();
  }

  void acquire(edm::Event const& iEvent, edm::EventSetup const& iSetup, Input& iInput) override {
    edm::Service<TritonService> ts;
    const std::string& modelName = client_->modelName();

    // Test dynamic loading and unloading
    if (testConcurrency_) {
      // Stress test with multiple rapid load/unload cycles
      for (int i = 0; i < loadUnloadCycles_; ++i) {
        bool loadResult = ts->loadModel(modelName);
        edm::LogInfo("DynamicModelLoadingProducer")
            << "Load attempt " << i << ": " << (loadResult ? "success" : "failed");

        // Small delay to allow other threads to interleave
        if (i % 5 == 0) {
          std::this_thread::yield();
        }

        bool unloadResult = ts->unloadModel(modelName);
        edm::LogInfo("DynamicModelLoadingProducer")
            << "Unload attempt " << i << ": " << (unloadResult ? "success" : "failed");
      }
    } else {
      // Simple test: load once, unload once
      bool loadResult = ts->loadModel(modelName);
      edm::LogInfo("DynamicModelLoadingProducer") << "Single load: " << (loadResult ? "success" : "failed");

      bool unloadResult = ts->unloadModel(modelName);
      edm::LogInfo("DynamicModelLoadingProducer") << "Single unload: " << (unloadResult ? "success" : "failed");
    }

    // Fill dummy input - use actual input from the model (gat_test expects "x" input)
    // This is just to satisfy the base class requirements, not for actual inference
    auto& input_x = iInput.at("x");
    auto data_x = input_x.allocate<float>();
    // Minimal dummy data
    (*data_x)[0] = std::vector<float>{1.0f};
    input_x.setShape(0, 1, 0);
    input_x.toServer(data_x);
  }

  void produce(edm::Event& iEvent, edm::EventSetup const& iSetup, Output const& iOutput) override {
    // Produce dummy output
    iEvent.emplace(putToken_, loadUnloadCycles_);
  }

  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    TritonClient::fillPSetDescription(desc);
    desc.add<int>("loadUnloadCycles", 1);
    desc.add<bool>("testConcurrency", false);
    descriptions.addWithDefaultLabel(desc);
  }

private:
  int loadUnloadCycles_;
  bool testConcurrency_;
  edm::EDPutTokenT<edmtest::IntProduct> putToken_;
};

DEFINE_FWK_MODULE(DynamicModelLoadingProducer);
