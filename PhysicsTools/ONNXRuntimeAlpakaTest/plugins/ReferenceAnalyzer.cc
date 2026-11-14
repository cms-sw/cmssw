#include <algorithm>
#include <cmath>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "DataFormats/PortableTestObjects/interface/ImageHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/LogitsHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/MultiHeadNetHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/ParticleHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/SimpleNetHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

namespace onnxtest {

  // Compare the results of the ONNX Runtime alpaka modules with the same models run on the CPU by the plain
  // cms::Ort::ONNXRuntime interface, with the inputs laid out by hand in the PyTorch convention: [batch, features]
  // for the particles, and [batch, channel, row, column] for the images.
  // This checks the shapes, strides, packing and unpacking of the tensors, and the numerical accuracy of the GPU
  // backends. The results of the models that have not been run are ignored.
  class ReferenceAnalyzer : public edm::global::EDAnalyzer<> {
  public:
    ReferenceAnalyzer(const edm::ParameterSet& params)
        : particles_token_{consumes(params.getParameter<edm::InputTag>("particles"))},
          images_token_{consumes(params.getParameter<edm::InputTag>("images"))},
          simple_net_model_{params.getParameter<edm::FileInPath>("simpleNetModel").fullPath()},
          multi_head_net_model_{params.getParameter<edm::FileInPath>("multiHeadNetModel").fullPath()},
          resnet_model_{params.getParameter<edm::FileInPath>("tinyResNetModel").fullPath()},
          tolerance_(params.getParameter<double>("tolerance")) {
      for (auto const& tag : params.getParameter<std::vector<edm::InputTag>>("simpleNet"))
        simple_net_tokens_.emplace_back(tag.label(), mayConsume<portabletest::SimpleNetHostCollection>(tag));
      for (auto const& tag : params.getParameter<std::vector<edm::InputTag>>("multiHeadNet"))
        multi_head_net_tokens_.emplace_back(tag.label(), mayConsume<portabletest::MultiHeadNetHostCollection>(tag));
      for (auto const& tag : params.getParameter<std::vector<edm::InputTag>>("tinyResNet"))
        resnet_tokens_.emplace_back(tag.label(), mayConsume<portabletest::LogitsHostCollection>(tag));
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      const std::string data = "PhysicsTools/ONNXRuntimeAlpakaTest/data/";
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("particles");
      desc.add<edm::InputTag>("images");
      desc.add<edm::FileInPath>("simpleNetModel", edm::FileInPath(data + "SimpleNet.onnx"));
      desc.add<edm::FileInPath>("multiHeadNetModel", edm::FileInPath(data + "MultiHeadNet.onnx"));
      desc.add<edm::FileInPath>("tinyResNetModel", edm::FileInPath(data + "TinyResNet.onnx"));
      desc.add<std::vector<edm::InputTag>>("simpleNet", {})->setComment("Results of the SimpleNet model.");
      desc.add<std::vector<edm::InputTag>>("multiHeadNet", {})->setComment("Results of the MultiHeadNet model.");
      desc.add<std::vector<edm::InputTag>>("tinyResNet", {})->setComment("Results of the TinyResNet model.");
      desc.add<double>("tolerance", 1e-4)
          ->setComment("Maximum difference, relative to 1 + |reference|; the default allows FP32 on GPUs, not TF32.");
      descriptions.addWithDefaultLabel(desc);
    }

    void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const override {
      std::ostringstream errors;

      auto particles_handle = event.getHandle(particles_token_);
      if (particles_handle.isValid() and particles_handle->const_view().metadata().size() > 0) {
        auto const& particles = particles_handle->const_view();
        const int n = particles.metadata().size();
        cms::Ort::FloatArrays input(1);
        input[0].reserve(n * 3);
        for (int i = 0; i < n; ++i) {
          input[0].push_back(particles[i].pt());
          input[0].push_back(particles[i].eta());
          input[0].push_back(particles[i].phi());
        }

        std::vector<float> simple_net;
        for (auto const& [label, token] : simple_net_tokens_) {
          auto handle = event.getHandle(token);
          if (not handle.isValid())
            continue;
          if (simple_net.empty())
            simple_net = simple_net_model_.run({"particles"}, input, {{n, 3}}, {"regression_head"}, n)[0];
          auto const& view = handle->const_view();
          compare(
              errors, label, n, view.metadata().size(), 1, simple_net, [&](int i, int) { return view[i].reco_pt(); });
        }

        cms::Ort::FloatArrays multi_head_net;
        constexpr int classes = portabletest::ClassificationHead::RowsAtCompileTime;
        for (auto const& [label, token] : multi_head_net_tokens_) {
          auto handle = event.getHandle(token);
          if (not handle.isValid())
            continue;
          if (multi_head_net.empty())
            multi_head_net = multi_head_net_model_.run(
                {"particles"}, input, {{n, 3}}, {"regression_head", "classification_head"}, n);
          auto const& view = handle->const_view();
          compare(errors, label + " regression head", n, view.metadata().size(), 1, multi_head_net[0], [&](int i, int) {
            return view[i].regression_head();
          });
          compare(errors,
                  label + " classification head",
                  n,
                  view.metadata().size(),
                  classes,
                  multi_head_net[1],
                  [&](int i, int k) { return view[i].classification_head()(k); });
        }
      }

      auto images_handle = event.getHandle(images_token_);
      if (images_handle.isValid() and images_handle->const_view().metadata().size() > 0) {
        auto const& images = images_handle->const_view();
        const int n = images.metadata().size();
        constexpr int rows = portabletest::ColorChannel::RowsAtCompileTime;
        constexpr int cols = portabletest::ColorChannel::ColsAtCompileTime;
        constexpr int classes = portabletest::LogitsType::RowsAtCompileTime;
        std::vector<float> logits;
        for (auto const& [label, token] : resnet_tokens_) {
          auto handle = event.getHandle(token);
          if (not handle.isValid())
            continue;
          if (logits.empty()) {
            cms::Ort::FloatArrays input(1);
            input[0].reserve(n * 3 * rows * cols);
            for (int b = 0; b < n; ++b)
              for (auto const& channel : {images[b].r(), images[b].g(), images[b].b()})
                for (int i = 0; i < rows; ++i)
                  for (int j = 0; j < cols; ++j)
                    input[0].push_back(channel(i, j));
            logits = resnet_model_.run({"images"}, input, {{n, 3, rows, cols}}, {"logits"}, n)[0];
          }
          auto const& view = handle->const_view();
          compare(errors, label, n, view.metadata().size(), classes, logits, [&](int i, int k) {
            return view[i].logits()(k);
          });
        }
      }

      if (not errors.str().empty()) {
        throw cms::Exception("TestFailed")
            << "Event " << event.id().event() << ": the results differ from the CPU reference" << errors.str();
      }
    }

  private:
    template <typename F>
    void compare(std::ostringstream& errors,
                 std::string const& label,
                 int size,
                 int result_size,
                 int features,
                 std::vector<float> const& reference,
                 F&& result) const {
      if (result_size != size) {
        errors << "\n  " << label << ": size " << result_size << ", expected " << size;
        return;
      }
      int failures = 0;
      double max_diff = 0.;
      for (int i = 0; i < size; ++i) {
        for (int k = 0; k < features; ++k) {
          const double expected = reference[i * features + k];
          const double value = result(i, k);
          const double diff = std::abs(value - expected);
          if (not(diff <= tolerance_ * (1. + std::abs(expected)))) {
            ++failures;
            max_diff = std::max(max_diff, std::isfinite(diff) ? diff : INFINITY);
          }
        }
      }
      if (failures > 0) {
        errors << "\n  " << label << ": " << failures << " values differ, by up to " << max_diff;
      }
    }

    const edm::EDGetTokenT<portabletest::ParticleHostCollection> particles_token_;
    const edm::EDGetTokenT<portabletest::ImageHostCollection> images_token_;
    std::vector<std::pair<std::string, edm::EDGetTokenT<portabletest::SimpleNetHostCollection>>> simple_net_tokens_;
    std::vector<std::pair<std::string, edm::EDGetTokenT<portabletest::MultiHeadNetHostCollection>>>
        multi_head_net_tokens_;
    std::vector<std::pair<std::string, edm::EDGetTokenT<portabletest::LogitsHostCollection>>> resnet_tokens_;
    const cms::Ort::ONNXRuntime simple_net_model_;
    const cms::Ort::ONNXRuntime multi_head_net_model_;
    const cms::Ort::ONNXRuntime resnet_model_;
    const double tolerance_;
  };

}  // namespace onnxtest

DEFINE_FWK_MODULE(onnxtest::ReferenceAnalyzer);
