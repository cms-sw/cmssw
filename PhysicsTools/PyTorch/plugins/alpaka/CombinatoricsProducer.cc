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
#include "PhysicsTools/PyTorch/interface/Nvtx.h"
#include "PhysicsTools/PyTorch/plugins/alpaka/Kernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  /**
   * @class CombinatoricsProducer
   * @brief A dummy Alpaka EDProducer that fills a particle collection.
   *
   * This producer simulates combinatorics logic by filling a new ParticleCollection with
   * placeholder values via a kernel call. Intended primarily for testing and validation.
   */
  class CombinatoricsProducer : public stream::EDProducer<> {
  public:
    CombinatoricsProducer(const edm::ParameterSet &params);

    void produce(device::Event &event, const device::EventSetup &event_setup) override;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  private:
    const device::EDGetToken<torchportable::ParticleCollection> inputs_token_;  /**< Token to get input data. */
    const device::EDPutToken<torchportable::ParticleCollection> outputs_token_; /**< Token to store output data. */
    std::unique_ptr<Kernels> kernels_ = nullptr;                                /**< Kernel helper object. */
  };

  CombinatoricsProducer::CombinatoricsProducer(edm::ParameterSet const &params)
      : EDProducer<>(params),
        inputs_token_{consumes(params.getParameter<edm::InputTag>("inputs"))},
        outputs_token_{produces()},
        kernels_(std::make_unique<Kernels>()) {}

  /**
   * @brief Processes the event and fills output with mock data using a kernel.
   * @param event The current event.
   * @param event_setup Setup information for the event.
   */
  void CombinatoricsProducer::produce(device::Event &event, const device::EventSetup &event_setup) {
    auto t1 = std::chrono::high_resolution_clock::now();

    // debug stream usage in concurrently scheduled modules
    std::stringstream msg_stream;
    msg_stream << "Combinatorics::produce [E: " << event.id().event() << "]";
    auto msg = msg_stream.str();
    NvtxScopedRange produce_range(msg.c_str());

    // get data
    const auto &inputs = event.get(inputs_token_);
    const size_t batch_size = inputs.const_view().metadata().size();
    auto outputs = torchportable::ParticleCollection(batch_size, event.queue());

    // dummy kernel emulation
    NvtxScopedRange kernel_range("Combinatorics::kernel");
    kernels_->FillParticleCollection(event.queue(), outputs, 0.32f);
    kernel_range.end();

    // assert output match expected
    kernels_->AssertCombinatorics(event.queue(), outputs, 0.32f);
    event.emplace(outputs_token_, std::move(outputs));
    alpaka::wait(event.queue());
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "(Combinatorics) E: " << event.id().event() << " OK - "
              << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " us" << std::endl;
    produce_range.end();
  }

  /**
   * @brief Defines configuration parameters for the module.
   * @param descriptions Object to be populated with parameter descriptions.
   */
  void CombinatoricsProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<edm::InputTag>("inputs");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(CombinatoricsProducer);
