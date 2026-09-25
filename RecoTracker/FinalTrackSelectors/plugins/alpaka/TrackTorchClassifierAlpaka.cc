#include "RecoTracker/FinalTrackSelectors/interface/alpaka/TrackFeaturesDeviceCollection.h"
#include "RecoTracker/FinalTrackSelectors/interface/alpaka/TrackScoresDeviceCollection.h"
#include "RecoTracker/FinalTrackSelectors/interface/TrackTorchClassifierFeaturesSoA.h"

#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/stream/FixedQueueEDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

#include "PhysicsTools/PyTorchAlpaka/interface/TensorCollection.h"
#include "PhysicsTools/PyTorchAlpaka/interface/alpaka/AlpakaModel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class TrackTorchClassifierAlpaka : public stream::FixedQueueEDProducer<> {
  public:
    TrackTorchClassifierAlpaka(const edm::ParameterSet& iConfig)
        : FixedQueueEDProducer<>(iConfig),
          featuresInput_token_(consumes(iConfig.getParameter<edm::InputTag>("features"))),
          scoresPut_token_{produces()},
          model_(iConfig.getParameter<edm::FileInPath>("modelPath").fullPath()) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::FileInPath>("modelPath",
                                edm::FileInPath("RecoTracker/FinalTrackSelectors/data/TrackTorchClassifier/model.pt"));
      desc.add<edm::InputTag>("features", edm::InputTag("hltInitialStepTrackFeatureExtractor"));
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(device::Event& iEvent, const device::EventSetup& iSetup) override {
      const auto& features = iEvent.get(featuresInput_token_);
      const auto batch_size = features.const_view().metadata().size();

      auto scores_device = TrackScoresDeviceCollection(iEvent.queue(), batch_size);

      auto input_records = features.const_view().records();
      auto output_records = scores_device.view().records();

      cms::torch::alpakatools::TensorCollection<Queue> inputs(batch_size);
      inputs.add<TrackTorchClassifierFeaturesSoA>("features",
                                                  input_records.dxyBeamSpot(),
                                                  input_records.dzBeamSpot(),
                                                  input_records.dxyError(),
                                                  input_records.dzError(),
                                                  input_records.normalizedChi2(),
                                                  input_records.eta(),
                                                  input_records.phi(),
                                                  input_records.etaError(),
                                                  input_records.phiError(),
                                                  input_records.ndof(),
                                                  input_records.lostInnerHits(),
                                                  input_records.lostOuterHits(),
                                                  input_records.layersWithoutMeas(),
                                                  input_records.validPixelHits(),
                                                  input_records.validStripHits());

      cms::torch::alpakatools::TensorCollection<Queue> outputs(batch_size);
      outputs.add<TrackTorchClassifierScoresSoA>("scores", output_records.score());

      model_.forward(iEvent.queue(), inputs, outputs);

      iEvent.emplace(scoresPut_token_, std::move(scores_device));
    }

    void beginStream(edm::StreamID sid, Queue queue) override {
      // Warmup the model with dummy data
      const int warmupBatchSize = 4992;
      // Allocate dummy input and output tensors on the device
      auto features = TrackFeaturesDeviceCollection(queue, warmupBatchSize);
      auto scores_device = TrackScoresDeviceCollection(queue, warmupBatchSize);

      auto input_records = features.view().records();
      auto output_records = scores_device.view().records();

      for (auto it = 0; it < warmupIterations_; ++it) {
        cms::torch::alpakatools::TensorCollection<Queue> dummy_inputs(warmupBatchSize);
        cms::torch::alpakatools::TensorCollection<Queue> dummy_outputs(warmupBatchSize);

        dummy_inputs.add<TrackTorchClassifierFeaturesSoA>("features",
                                                          input_records.dxyBeamSpot(),
                                                          input_records.dzBeamSpot(),
                                                          input_records.dxyError(),
                                                          input_records.dzError(),
                                                          input_records.normalizedChi2(),
                                                          input_records.eta(),
                                                          input_records.phi(),
                                                          input_records.etaError(),
                                                          input_records.phiError(),
                                                          input_records.ndof(),
                                                          input_records.lostInnerHits(),
                                                          input_records.lostOuterHits(),
                                                          input_records.layersWithoutMeas(),
                                                          input_records.validPixelHits(),
                                                          input_records.validStripHits());

        dummy_outputs.add<TrackTorchClassifierScoresSoA>("scores", output_records.score());

        model_.forward(queue, dummy_inputs, dummy_outputs);
      }
    }

  private:
    const device::EDGetToken<TrackFeaturesDeviceCollection> featuresInput_token_;
    const device::EDPutToken<TrackScoresDeviceCollection> scoresPut_token_;
    torch::AlpakaModel model_;
    const int warmupIterations_ = 3;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TrackTorchClassifierAlpaka);
