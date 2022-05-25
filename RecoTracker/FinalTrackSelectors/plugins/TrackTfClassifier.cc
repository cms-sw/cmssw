#include "RecoTracker/FinalTrackSelectors/interface/TrackMVAClassifier.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/global/EDProducer.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
#include "RecoTracker/FinalTrackSelectors/interface/getBestVertex.h"

#include "PhysicsTools/TensorFlow/interface/TfGraphRecord.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"
#include "PhysicsTools/TensorFlow/interface/TfGraphDefWrapper.h"

namespace {
  class TfDnn {
  public:
    TfDnn(const edm::ParameterSet& cfg, edm::ConsumesCollector iC)
        : tfDnnLabel_(cfg.getParameter<std::string>("tfDnnLabel")),
          tfDnnToken_(iC.esConsumes(edm::ESInputTag("", tfDnnLabel_))),
          session_(nullptr),
          bsize_(cfg.getParameter<int>("batchSize"))

    {}

    static const char* name() { return "trackTfClassifierDefault"; }

    static void fillDescriptions(edm::ParameterSetDescription& desc) {
      desc.add<std::string>("tfDnnLabel", "trackSelectionTf");
      desc.add<int>("batchSize", 16);
    }
    void beginStream() {}

    void initEvent(const edm::EventSetup& es) {
      if (session_ == nullptr) {
        session_ = es.getData(tfDnnToken_).getSession();
      }
    }

    std::vector<float> operator()(reco::TrackCollection const& tracks,
                                  reco::BeamSpot const& beamSpot,
                                  reco::VertexCollection const& vertices) const {
      int size_in = (int)tracks.size();
      int nbatches = size_in / bsize_;

      std::vector<float> output;
      output.resize(size_in);

      tensorflow::Tensor input1(tensorflow::DT_FLOAT, {bsize_, 29});
      tensorflow::Tensor input2(tensorflow::DT_FLOAT, {bsize_, 1});

      for (auto nb = 0; nb < nbatches + 1; nb++) {
        for (auto nt = 0; nt < bsize_; nt++) {
          int itrack = nt + bsize_ * nb;
          if (itrack >= size_in)
            continue;
          const auto& trk = tracks[itrack];

          const auto& bestVertex = getBestVertex(trk, vertices);

          input1.matrix<float>()(nt, 0) = trk.pt();
          input1.matrix<float>()(nt, 1) = trk.innerMomentum().x();
          input1.matrix<float>()(nt, 2) = trk.innerMomentum().y();
          input1.matrix<float>()(nt, 3) = trk.innerMomentum().z();
          input1.matrix<float>()(nt, 4) = trk.innerMomentum().rho();
          input1.matrix<float>()(nt, 5) = trk.outerMomentum().x();
          input1.matrix<float>()(nt, 6) = trk.outerMomentum().y();
          input1.matrix<float>()(nt, 7) = trk.outerMomentum().z();
          input1.matrix<float>()(nt, 8) = trk.outerMomentum().rho();
          input1.matrix<float>()(nt, 9) = trk.ptError();
          input1.matrix<float>()(nt, 10) = trk.dxy(bestVertex);
          input1.matrix<float>()(nt, 11) = trk.dz(bestVertex);
          input1.matrix<float>()(nt, 12) = trk.dxy(beamSpot.position());
          input1.matrix<float>()(nt, 13) = trk.dz(beamSpot.position());
          input1.matrix<float>()(nt, 14) = trk.dxyError();
          input1.matrix<float>()(nt, 15) = trk.dzError();
          input1.matrix<float>()(nt, 16) = trk.normalizedChi2();
          input1.matrix<float>()(nt, 17) = trk.eta();
          input1.matrix<float>()(nt, 18) = trk.phi();
          input1.matrix<float>()(nt, 19) = trk.etaError();
          input1.matrix<float>()(nt, 20) = trk.phiError();
          input1.matrix<float>()(nt, 21) = trk.hitPattern().numberOfValidPixelHits();
          input1.matrix<float>()(nt, 22) = trk.hitPattern().numberOfValidStripHits();
          input1.matrix<float>()(nt, 23) = trk.ndof();
          input1.matrix<float>()(nt, 24) =
              trk.hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_INNER_HITS);
          input1.matrix<float>()(nt, 25) =
              trk.hitPattern().numberOfLostTrackerHits(reco::HitPattern::MISSING_OUTER_HITS);
          input1.matrix<float>()(nt, 26) =
              trk.hitPattern().trackerLayersTotallyOffOrBad(reco::HitPattern::MISSING_INNER_HITS);
          input1.matrix<float>()(nt, 27) =
              trk.hitPattern().trackerLayersTotallyOffOrBad(reco::HitPattern::MISSING_OUTER_HITS);
          input1.matrix<float>()(nt, 28) =
              trk.hitPattern().trackerLayersWithoutMeasurement(reco::HitPattern::TRACK_HITS);

          //Original algo as its own input, it will enter the graph so that it gets one-hot encoded, as is the preferred
          //format for categorical inputs, where the labels do not have any metric amongst them
          input2.matrix<float>()(nt, 0) = trk.originalAlgo();
        }

        //The names for the input tensors get locked when freezing the trained tensorflow model. The NamedTensors must
        //match those names
        tensorflow::NamedTensorList inputs;
        inputs.resize(2);
        inputs[0] = tensorflow::NamedTensor("x", input1);
        inputs[1] = tensorflow::NamedTensor("y", input2);
        std::vector<tensorflow::Tensor> outputs;

        //evaluate the input
        tensorflow::run(const_cast<tensorflow::Session*>(session_), inputs, {"Identity"}, &outputs);

        for (auto nt = 0; nt < bsize_; nt++) {
          int itrack = nt + bsize_ * nb;
          if (itrack >= size_in)
            continue;
          float out0 = 2.0 * outputs[0].matrix<float>()(nt, 0) - 1.0;
          output[itrack] = out0;
        }
      }
      return output;
    }

    const std::string tfDnnLabel_;
    const edm::ESGetToken<TfGraphDefWrapper, TfGraphRecord> tfDnnToken_;
    const tensorflow::Session* session_;
    const int bsize_;
  };
}  // namespace

template <>
void trackMVAClassifierImpl::ComputeMVA<void>::operator()(::TfDnn const& mva,
                                                          reco::TrackCollection const& tracks,
                                                          reco::BeamSpot const& beamSpot,
                                                          reco::VertexCollection const& vertices,
                                                          TrackMVAClassifierBase::MVAPairCollection& mvas) {
  const auto& scores = mva(tracks, beamSpot, vertices);
  size_t current = 0;

  for (auto score : scores) {
    std::pair<float, bool> output(score, true);
    mvas[current++] = output;
  }
}

namespace {
  using TrackTfClassifier = TrackMVAClassifier<TfDnn>;
}  // namespace

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE(TrackTfClassifier);
