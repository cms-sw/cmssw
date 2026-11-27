#include "RecoTICL/Inference/interface/TracksterInferenceByShowerShape.h"
#include "RecoTICL/Inference/interface/TracksterInferenceAlgoFactory.h"
#include "RecoTICL/Inference/interface/TracksterLayerClusterProfile.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <functional>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <utility>

#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/ParameterSet/interface/allowedValues.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace ticl {

  namespace {
    constexpr std::size_t kNTypes =
        std::tuple_size_v<std::remove_cvref_t<decltype(std::declval<const Trackster&>().id_probabilities())>>;
  }  // namespace

  TracksterInferenceByShowerShape::TracksterInferenceByShowerShape(const edm::ParameterSet& conf,
                                                                   TICLONNXGlobalCache const* cache)
      : TracksterInferenceAlgoBase(conf, cache),
        mixture_(conf.getParameter<std::string>("mode") == "mixture"),
        inputNames_(conf.getParameter<std::vector<std::string>>("inputNames")),
        outputNames_(conf.getParameter<std::vector<std::string>>("outputNames")),
        minEnergy_(conf.getParameter<double>("minEnergy")),
        doPID_(conf.getParameter<int>("doPID")) {
    const std::string modelPath = conf.getParameter<std::string>("onnxModelPath");
    if (cache_ != nullptr && !modelPath.empty())
      onnxSession_ = cache_->getByModelPathString(modelPath);
    if (onnxSession_ == nullptr)
      return;

    // The model must match the mode: nOut values per trackster in the output, and an input row of the mode width.
    const std::size_t nOut = mixture_ ? kNTypes : 1;
    const auto nFeatures = static_cast<int64_t>(mixture_ ? kNMixtureFeatures : kNFeatures);
    if (outputNames_.size() != 1)
      throw cms::Exception("Configuration")
          << "TracksterInferenceByShowerShape: outputNames must have one name, it has " << outputNames_.size();
    const auto& modelOutputs = onnxSession_->getOutputNames();
    if (std::find(modelOutputs.begin(), modelOutputs.end(), outputNames_[0]) == modelOutputs.end())
      throw cms::Exception("Configuration")
          << "TracksterInferenceByShowerShape: the model " << modelPath << " has no output '" << outputNames_[0] << "'";
    const auto& shape = onnxSession_->getOutputShape(outputNames_[0]);
    const auto perTrackster = std::accumulate(
        shape.begin() + std::min<std::size_t>(1, shape.size()), shape.end(), int64_t{1}, std::multiplies<>());
    if (perTrackster != static_cast<int64_t>(nOut))
      throw cms::Exception("Configuration")
          << "TracksterInferenceByShowerShape: the output '" << outputNames_[0] << "' of the model " << modelPath
          << " has " << perTrackster << " values per trackster, mode '" << (mixture_ ? "mixture" : "em") << "' needs "
          << nOut;
    cms::Ort::FloatArrays input(1, std::vector<float>(nFeatures, 0.f));
    try {
      onnxSession_->run(inputNames_, input, {{1, nFeatures}}, outputNames_, 1);
    } catch (std::exception const& e) {
      throw cms::Exception("Configuration")
          << "TracksterInferenceByShowerShape: the model " << modelPath << " does not accept the input of mode '"
          << (mixture_ ? "mixture" : "em") << "' (" << nFeatures << " values per trackster): " << e.what();
    }
  }

  std::array<float, TracksterInferenceByShowerShape::kNFeatures> TracksterInferenceByShowerShape::features(
      const Trackster& trackster) {
    const float energy = trackster.raw_energy();
    const auto& barycenter = trackster.barycenter();
    const float eta = barycenter.eta();
    std::array<float, 3> eigenvalues{};
    for (int k = 0; k < 3; ++k)
      eigenvalues[k] = std::max(trackster.eigenvalues()[k], 1e-6f);
    const auto axis = trackster.eigenvectors(0);
    const float axisEta = std::asinh(axis.z() / std::max(std::hypot(axis.x(), axis.y()), 1e-6f));
    return {std::log1p(energy),
            trackster.raw_em_energy() / std::max(energy, 1e-6f),
            std::abs(eta),
            std::abs(barycenter.z()),
            std::log1p(static_cast<float>(trackster.vertices().size())),
            std::log(eigenvalues[0]),
            std::log(eigenvalues[1]),
            std::log(eigenvalues[2]),
            eigenvalues[0] / (eigenvalues[0] + eigenvalues[1] + eigenvalues[2]),
            trackster.sigmas()[0],
            trackster.sigmas()[1],
            trackster.sigmas()[2],
            trackster.sigmasPCA()[0],
            trackster.sigmasPCA()[1],
            trackster.sigmasPCA()[2],
            std::abs(axisEta - eta)};
  }

  void TracksterInferenceByShowerShape::runInference(const std::vector<reco::CaloCluster>& layerClusters,
                                                     std::vector<Trackster>& tracksters,
                                                     const ticlgeom::Tools& /*rhtools*/) const {
    if (doPID_ == 0 || onnxSession_ == nullptr || tracksters.empty())
      return;
    std::vector<int> indices;
    cms::Ort::FloatArrays input(1);
    for (int i = 0; i < static_cast<int>(tracksters.size()); ++i) {
      if (tracksters[i].raw_energy() < minEnergy_)
        continue;
      const auto x = features(tracksters[i]);
      input[0].insert(input[0].end(), x.begin(), x.end());
      if (mixture_) {
        const auto profile = layerClusterProfile(tracksters[i], layerClusters);
        input[0].insert(input[0].end(), profile.begin(), profile.end());
      }
      indices.push_back(i);
    }
    if (indices.empty())
      return;
    const auto n = static_cast<int64_t>(indices.size());
    const auto nFeatures = static_cast<int64_t>(mixture_ ? kNMixtureFeatures : kNFeatures);
    const std::size_t nOut = mixture_ ? kNTypes : 1;
    const auto output = onnxSession_->run(inputNames_, input, {{n, nFeatures}}, outputNames_, n);
    if (output.empty() || output[0].size() != nOut * indices.size())
      throw cms::Exception("LogicError") << "TracksterInferenceByShowerShape: the model gives "
                                         << (output.empty() ? 0 : output[0].size()) << " values for " << indices.size()
                                         << " tracksters, expected " << nOut << " per trackster";
    for (size_t k = 0; k < indices.size(); ++k) {
      auto& trackster = tracksters[indices[k]];
      trackster.zeroProbabilities();
      if (mixture_) {
        for (std::size_t c = 0; c < nOut; ++c)
          trackster.setIdProbability(static_cast<Trackster::ParticleType>(c),
                                     std::clamp(output[0][nOut * k + c], 0.f, 1.f));
      } else {
        const float s = std::clamp(output[0][k], 0.f, 1.f);
        trackster.setIdProbability(Trackster::ParticleType::photon, s);
        trackster.setIdProbability(Trackster::ParticleType::charged_hadron, 1.f - s);
      }
    }
  }

  void TracksterInferenceByShowerShape::fillPSetDescription(edm::ParameterSetDescription& iDesc) {
    TracksterInferenceAlgoBase::fillPSetDescription(iDesc);
    iDesc.ifValue(edm::ParameterDescription<std::string>(
                      "mode",
                      "em",
                      true,
                      edm::Comment("em: input [N, 16], output [N] EM score; mixture: input [N, 22] (with the "
                                   "layer-cluster profile), output [N, 8] probabilities in ParticleType order, and it "
                                   "needs outputNames = ['pid'].")),
                  edm::allowedValues<std::string>("em", "mixture"));
    iDesc.add<std::string>("onnxModelPath", "RecoTICL/Inference/data/showerShapePID/shapePID_v1.onnx")
        ->setComment("ONNX model of the mode.");
    iDesc.add<std::vector<std::string>>("inputNames", {"input"});
    iDesc.add<std::vector<std::string>>("outputNames", {"em"});
    iDesc.add<double>("minEnergy", 1.0)->setComment("Min raw energy [GeV] of a trackster to get an identification.");
    iDesc.add<int>("doPID", 1);
  }

}  // namespace ticl
