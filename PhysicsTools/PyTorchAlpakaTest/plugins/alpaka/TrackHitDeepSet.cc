#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "DataFormats/PortableTestObjects/interface/TestSoA.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/HitDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/ParticleDeviceCollection.h"
#include "DataFormats/PortableTestObjects/interface/alpaka/SimpleNetDeviceCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/BatchedTensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/Environment.h"
#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/CommonKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest {

  class TrackHitDeepSet : public stream::EDProducer<> {
  public:
    TrackHitDeepSet(const edm::ParameterSet &params)
        : EDProducer<>(params),
          particles_token_(consumes(params.getParameter<edm::InputTag>("particles"))),
          hits_token_(consumes(params.getParameter<edm::InputTag>("hits"))),
          hit_to_track_token_(consumes(params.getParameter<edm::InputTag>("hit_to_track"))),
          deepSet_token_{produces()},
          model_(params.getParameter<edm::FileInPath>("model").fullPath()),
          batch_size_(params.getParameter<uint32_t>("batchSize")),
          environment_{static_cast<::torchtest::Environment>(params.getUntrackedParameter<int>("environment"))} {
      assert(batch_size_ != 0 && "TrackHitDeepSet: batch_size should be greater than 0");
    }

    static void fillDescriptions(edm::ConfigurationDescriptions &descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("model");
      desc.add<uint32_t>("batchSize");
      desc.add<edm::InputTag>("particles");
      desc.add<edm::InputTag>("hits");
      desc.add<edm::InputTag>("hit_to_track");
      desc.addUntracked<int>("environment", static_cast<int>(::torchtest::Environment::kProduction));
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event &event, const device::EventSetup &event_setup) override {
      auto &queue = event.queue();
      // in/out collections
      const auto &particles = event.get(particles_token_);
      const auto &hits = event.get(hits_token_);
      const auto &hit_to_track = event.get(hit_to_track_token_);

      const auto total_size = particles.const_view().metadata().size();

      auto regression_collection = portabletest::SimpleNetDeviceCollection(queue, total_size);
      if (total_size == 0) {
        event.emplace(deepSet_token_, std::move(regression_collection));
        return;
      }

      uint32_t n_batches;
      n_batches = (total_size + batch_size_ - 1) / batch_size_;

      auto track_begin = portabletest::TrackBeginDeviceCollection(queue, n_batches);
      kernels::fillTrackBegin(queue, track_begin, batch_size_);

      // records
      auto input_records = particles.const_view().records();
      auto hit_records = hits.const_view().records();
      auto hit_to_track_records = hit_to_track.const_view().records();
      auto track_begin_records = track_begin.const_view().records();

      auto output_records = regression_collection.view().records();

      // input and output tensor definitions
      auto inputs = cms::torch::alpakatools::BatchedTensorCollection<Queue>();
      auto outputs = cms::torch::alpakatools::BatchedTensorCollection<Queue>();

      inputs.addBatched<portabletest::ParticleSoA>(
          "track_features", batch_size_, input_records.pt(), input_records.eta(), input_records.phi());
      inputs.addBatched<portabletest::HitSoA>("hit_features", hit_records.x(), hit_records.y(), hit_records.z());
      inputs.addBatched<portabletest::HitToTrackSoA>("hit_to_track", hit_to_track_records.trackIndex());
      inputs.addBatched<portabletest::TrackBeginSoA>("track_begin", 1, track_begin_records.trackBegin());

      outputs.addBatched<portabletest::SimpleNetSoA>("regression_head", batch_size_, output_records.reco_pt());

      model_.forward(queue, inputs, outputs);

      // put device-side product into event
      event.emplace(deepSet_token_, std::move(regression_collection));
    }

  private:
    // event query tokens
    const device::EDGetToken<portabletest::ParticleDeviceCollection> particles_token_;
    const device::EDGetToken<portabletest::HitDeviceCollection> hits_token_;
    const device::EDGetToken<portabletest::HitToTrackDeviceCollection> hit_to_track_token_;
    const device::EDPutToken<portabletest::SimpleNetDeviceCollection> deepSet_token_;
    // model
    torch::AlpakaModel model_;
    const uint32_t batch_size_;
    // debug mode flag
    const ::torchtest::Environment environment_;
  };
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest

DEFINE_FWK_ALPAKA_MODULE(torchtest::TrackHitDeepSet);
