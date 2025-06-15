#include <alpaka/alpaka.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include "DataFormats/PyTorchTest/interface/alpaka/Collections.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/AlpakaConfig.h"
#include "PhysicsTools/PyTorch/interface/Model.h"
#include "PhysicsTools/PyTorch/interface/SoAMetadata.h"
#include "PhysicsTools/PyTorch/interface/Nvtx.h"
#include "PhysicsTools/PyTorch/plugins/alpaka/Kernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using AotModel = cms::torch::alpaka::Model<cms::torch::alpaka::CompilationType::kAheadOfTime>;

  /**
   * @class AotClassificationProducer
   * @brief EDProducer that runs a classification model on input particles with Alpaka backend.
   *
   * Producer reads a collection of particles, runs inference on a just-in-time compiled model,
   * and writes the classification outputs to the event. 
   * 
   * Demonstrates the interoperability of Alpaka with PyTorch and the use of SoA data layout with JIT models.
   */
  class AotClassificationProducer : public stream::EDProducer<> {
  public:
    AotClassificationProducer(const edm::ParameterSet &params);

    void produce(device::Event &event, const device::EventSetup &event_setup) override;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  private:
    const device::EDGetToken<torchportable::ParticleCollection> inputs_token_;        /**< Token to get input data. */
    const device::EDPutToken<torchportable::ClassificationCollection> outputs_token_; /**< Token to store output data. */
    std::unique_ptr<Kernels> kernels_ = nullptr; /**< Kernel utilities for post-inference validation. */
    std::unique_ptr<AotModel> model_;            /**< Cache for the JIT model. */
  };

  AotClassificationProducer::AotClassificationProducer(edm::ParameterSet const &params)
      : EDProducer<>(params),
        inputs_token_{consumes(params.getParameter<edm::InputTag>("inputs"))},
        outputs_token_{produces()},
        kernels_{std::make_unique<Kernels>()} {
    cms::torch::alpaka::set_threading_guard();
    model_ = std::make_unique<AotModel>(params.getParameter<edm::FileInPath>("modelPath").fullPath());
  }

  /**
   * @brief Executes model inference on the particle collection.
   * @param event The current Alpaka event.
   * @param event_setup Event setup context (unused).
   */
  void AotClassificationProducer::produce(device::Event &event, const device::EventSetup &event_setup) {
    auto t1 = std::chrono::high_resolution_clock::now();

    // debug stream usage in concurrently scheduled modules
    std::stringstream msg_stream;
    msg_stream << "ClassifierAot::produce [E: " << event.id().event() << "]";
    auto msg = msg_stream.str();
    NvtxScopedRange produce_range(msg.c_str());

    // guard torch internal operations to not conflict with fw execution scheme
    cms::torch::alpaka::Guard<Queue> guard(event.queue());
    // sanity check
    assert(cms::torch::alpaka::queue_hash(event.queue()) == cms::torch::alpaka::current_stream_hash(event.queue()));

    // get data
    // TODO: const_cast should not be done by user
    // in principle should not be done by anyone
    // @see: torch::from_blob(void*)
    auto &inputs = const_cast<torchportable::ParticleCollection &>(event.get(inputs_token_));
    const size_t batch_size = inputs.const_view().metadata().size();
    auto outputs = torchportable::ClassificationCollection(batch_size, event.queue());

    // metadata for automatic tensor conversion
    auto input_records = inputs.view().records();
    auto output_records = outputs.view().records();
    cms::torch::alpaka::SoAMetadata<torchportable::ParticleSoA> inputs_metadata(batch_size);
    inputs_metadata.append_block("features", input_records.pt(), input_records.eta(), input_records.phi());
    cms::torch::alpaka::SoAMetadata<torchportable::ClassificationSoA> outputs_metadata(batch_size);
    outputs_metadata.append_block("preds", output_records.c1(), output_records.c2());
    cms::torch::alpaka::ModelMetadata<torchportable::ParticleSoA, torchportable::ClassificationSoA> metadata(
        inputs_metadata, outputs_metadata);

    // inference
    NvtxScopedRange move_to_device("Classifier::move_to_device");
    if (cms::torch::alpaka::device(event.queue()) != model_->device()) {
      std::cout << "(ClassifierAot) E: " << event.id().event() << " Model: " << model_->device() << " -> "
                << cms::torch::alpaka::device(event.queue()) << std::endl;
      model_->to(event.queue());
    }
    assert(cms::torch::alpaka::device(event.queue()) == model_->device());
    move_to_device.end();
    NvtxScopedRange infer_range("Classifier::inference");
    model_->forward(metadata);
    infer_range.end();

    // assert output match expected
    kernels_->AssertClassification(event.queue(), outputs);
    event.emplace(outputs_token_, std::move(outputs));
    alpaka::wait(event.queue());
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "(ClassifierAot) E: " << event.id().event() << " OK - "
              << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " us" << std::endl;
    produce_range.end();
  }

  /**
   * @brief Describes module configuration parameters.
   * @param descriptions Configuration description object.
   */
  void AotClassificationProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("inputs");
    desc.add<edm::FileInPath>("modelPath");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(AotClassificationProducer);