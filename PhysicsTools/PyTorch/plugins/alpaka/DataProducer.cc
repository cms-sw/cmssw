#include "DataFormats/PyTorchTest/interface/alpaka/Collections.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "PhysicsTools/PyTorch/interface/Nvtx.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  /**
   * @class DataProducer
   * @brief A minimal Alpaka EDProducer that generates a dummy ParticleCollection.
   *
   * This module produces a particle collection of configurable batch size,
   * zero-initialized, for testing or as placeholder data in processing chains.
   */
  class DataProducer : public stream::EDProducer<> {
  public:
    DataProducer(const edm::ParameterSet &params);

    void produce(device::Event &event, const device::EventSetup &event_setup) override;
    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions);

  private:
    const device::EDPutToken<torchportable::ParticleCollection> sic_put_token_; /**< Token to store output data. */
    const uint32_t batch_size_; /**< Size of the batch to be produced. */
  };

  DataProducer::DataProducer(edm::ParameterSet const &params)
      : EDProducer<>(params), sic_put_token_{produces()}, batch_size_(params.getParameter<uint32_t>("batchSize")) {}

  /**
   * @brief Produces a ParticleCollection of fixed size with zero-initialized data.
   * @param event The current Alpaka event.
   * @param event_setup Event setup context (not used).
   */
  void DataProducer::produce(device::Event &event, const device::EventSetup &event_setup) {
    auto t1 = std::chrono::high_resolution_clock::now();

    // debug stream usage in concurrently scheduled modules
    std::stringstream msg_stream;
    msg_stream << "Data::produce [E: " << event.id().event() << "]";
    auto msg = msg_stream.str();
    NvtxScopedRange produce_range(msg.c_str());

    // create dummy data
    auto collection = torchportable::ParticleCollection(batch_size_, event.queue());
    collection.zeroInitialise(event.queue());
    event.emplace(sic_put_token_, std::move(collection));
    alpaka::wait(event.queue());
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "(Data) E: " << event.id().event() << " OK - "
              << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " us" << std::endl;
    produce_range.end();
  }

  /**
   * @brief Describes the allowed configuration parameters for this module.
   * @param descriptions Configuration description object to populate.
   */
  void DataProducer::fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<uint32_t>("batchSize");
    descriptions.addWithDefaultLabel(desc);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(DataProducer);
