#include <cmath>
#include <sstream>
#include <string>

#include "DataFormats/PortableTestObjects/interface/HitHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/LogitsHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/MultiHeadNetHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/ParticleHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/SimpleNetHostCollection.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/global/EDAnalyzer.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"

namespace onnxtest {

  // Check the results of the ONNX Runtime inference against the known properties of each model.
  // The results of the models that have not been run are ignored.
  // The ReferenceAnalyzer compares the results of SimpleNet, MultiHeadNet and TinyResNet with a CPU computation.
  class InspectionSink : public edm::global::EDAnalyzer<> {
  public:
    InspectionSink(const edm::ParameterSet& params)
        : particles_token_{consumes(params.getParameter<edm::InputTag>("particles"))},
          simple_net_token_{consumes(params.getParameter<edm::InputTag>("simple_net"))},
          simple_net_feature_major_token_{consumes(params.getParameter<edm::InputTag>("simple_net_feature_major"))},
          simple_net_minibatch_token_{consumes(params.getParameter<edm::InputTag>("simple_net_minibatch"))},
          masked_net_token_{consumes(params.getParameter<edm::InputTag>("masked_net"))},
          masked_net_reversed_token_{consumes(params.getParameter<edm::InputTag>("masked_net_reversed"))},
          masked_net_empty_mask_token_{consumes(params.getParameter<edm::InputTag>("masked_net_empty_mask"))},
          multi_head_net_token_{consumes(params.getParameter<edm::InputTag>("multi_head_net"))},
          logits_token_{consumes(params.getParameter<edm::InputTag>("resnet"))},
          logits_minibatch_token_{consumes(params.getParameter<edm::InputTag>("resnet_minibatch"))},
          hits_token_{consumes(params.getParameter<edm::InputTag>("hits"))},
          scaled_net_token_{consumes(params.getParameter<edm::InputTag>("scaled_net"))},
          scaled_net_minibatch_token_{consumes(params.getParameter<edm::InputTag>("scaled_net_minibatch"))} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("particles");
      desc.add<edm::InputTag>("simple_net");
      desc.add<edm::InputTag>("simple_net_feature_major");
      desc.add<edm::InputTag>("simple_net_minibatch");
      desc.add<edm::InputTag>("masked_net");
      desc.add<edm::InputTag>("masked_net_reversed");
      desc.add<edm::InputTag>("masked_net_empty_mask");
      desc.add<edm::InputTag>("multi_head_net");
      desc.add<edm::InputTag>("resnet");
      desc.add<edm::InputTag>("resnet_minibatch");
      desc.add<edm::InputTag>("hits");
      desc.add<edm::InputTag>("scaled_net");
      desc.add<edm::InputTag>("scaled_net_minibatch");
      descriptions.addWithDefaultLabel(desc);
    }

    void analyze(edm::StreamID, edm::Event const& event, edm::EventSetup const&) const override {
      Checker check(event.id().event());

      auto particles_handle = event.getHandle(particles_token_);
      if (particles_handle.isValid()) {
        auto const& particles = particles_handle->const_view();
        for (int32_t i = 0; i < particles.metadata().size(); ++i) {
          check(0.f <= particles[i].pt() and particles[i].pt() <= 1.f, "Particles", i, "pt out of range");
          check(0.f <= particles[i].eta() and particles[i].eta() <= 1.f, "Particles", i, "eta out of range");
          check(0.f <= particles[i].phi() and particles[i].phi() <= 1.f, "Particles", i, "phi out of range");
        }

        // SimpleNet: finite results, independent of the layout and of the mini-batches
        auto simple_net_handle = event.getHandle(simple_net_token_);
        if (simple_net_handle.isValid()) {
          auto const& simple_net = simple_net_handle->const_view();
          check(simple_net.metadata().size() == particles.metadata().size(), "SimpleNet", -1, "size mismatch");
          for (int32_t i = 0; i < simple_net.metadata().size(); ++i) {
            check(std::isfinite(simple_net[i].reco_pt()), "SimpleNet", i, "not finite");
          }
          for (auto const& [token, name] : {std::pair{simple_net_feature_major_token_, "SimpleNetFeatureMajor"},
                                            std::pair{simple_net_minibatch_token_, "SimpleNetMiniBatch"}}) {
            auto handle = event.getHandle(token);
            if (handle.isValid()) {
              auto const& other = handle->const_view();
              check(other.metadata().size() == simple_net.metadata().size(), name, -1, "size mismatch");
              for (int32_t i = 0; i < other.metadata().size(); ++i) {
                check(close(other[i].reco_pt(), simple_net[i].reco_pt()), name, i, "differs from SimpleNet");
              }
            }
          }
        }

        // MaskedNet: the eta feature is always masked out, so the result is pt + phi, also when the inputs are added
        // in a different order than the model inputs, and matched by name
        for (auto const& [token, name] :
             {std::pair{masked_net_token_, "MaskedNet"}, std::pair{masked_net_reversed_token_, "MaskedNetReversed"}}) {
          auto handle = event.getHandle(token);
          if (handle.isValid()) {
            auto const& masked_net = handle->const_view();
            check(masked_net.metadata().size() == particles.metadata().size(), name, -1, "size mismatch");
            for (int32_t i = 0; i < masked_net.metadata().size(); ++i) {
              check(close(masked_net[i].reco_pt(), particles[i].pt() + particles[i].phi()),
                    name,
                    i,
                    "result is not pt + phi");
            }
          }
        }

        // MaskedNet with an empty mask: the inference is skipped, and the output is filled with zeros
        auto masked_net_empty_mask_handle = event.getHandle(masked_net_empty_mask_token_);
        if (masked_net_empty_mask_handle.isValid()) {
          auto const& masked_net = masked_net_empty_mask_handle->const_view();
          check(masked_net.metadata().size() == particles.metadata().size(), "MaskedNetEmptyMask", -1, "size mismatch");
          for (int32_t i = 0; i < masked_net.metadata().size(); ++i) {
            check(masked_net[i].reco_pt() == 0.f, "MaskedNetEmptyMask", i, "output not filled with zeros");
          }
        }

        // MultiHeadNet: the classification head is a probability distribution
        auto multi_head_net_handle = event.getHandle(multi_head_net_token_);
        if (multi_head_net_handle.isValid()) {
          auto const& multi_head_net = multi_head_net_handle->const_view();
          check(multi_head_net.metadata().size() == particles.metadata().size(), "MultiHeadNet", -1, "size mismatch");
          for (int32_t i = 0; i < multi_head_net.metadata().size(); ++i) {
            check(std::isfinite(multi_head_net[i].regression_head()), "MultiHeadNet", i, "regression head not finite");
            check(isProbability(multi_head_net[i].classification_head()),
                  "MultiHeadNet",
                  i,
                  "classification head is not a probability distribution");
          }
        }
      }

      // TinyResNet: the logits are a probability distribution, independent of the mini-batches
      auto logits_handle = event.getHandle(logits_token_);
      if (logits_handle.isValid()) {
        auto const& logits = logits_handle->const_view();
        for (int32_t i = 0; i < logits.metadata().size(); ++i) {
          check(isProbability(logits[i].logits()), "TinyResNet", i, "logits are not a probability distribution");
        }
        auto logits_minibatch_handle = event.getHandle(logits_minibatch_token_);
        if (logits_minibatch_handle.isValid()) {
          auto const& other = logits_minibatch_handle->const_view();
          check(other.metadata().size() == logits.metadata().size(), "TinyResNetMiniBatch", -1, "size mismatch");
          for (int32_t i = 0; i < other.metadata().size(); ++i) {
            for (int k = 0; k < portabletest::LogitsType::RowsAtCompileTime; ++k) {
              check(close(other[i].logits()(k), logits[i].logits()(k)),
                    "TinyResNetMiniBatch",
                    i,
                    "differs from TinyResNet");
            }
          }
        }
      }

      // ScaledNet: the result is hitScale * (x + y + z), with and without mini-batches
      auto hits_handle = event.getHandle(hits_token_);
      if (hits_handle.isValid()) {
        auto const& hits = hits_handle->const_view();
        check(hits.hitScale() == 1.5f, "Hits", -1, "unexpected hitScale");
        for (auto const& [token, name] : {std::pair{scaled_net_token_, "ScaledNet"},
                                          std::pair{scaled_net_minibatch_token_, "ScaledNetMiniBatch"}}) {
          auto handle = event.getHandle(token);
          if (handle.isValid()) {
            auto const& scaled_net = handle->const_view();
            check(scaled_net.metadata().size() == hits.metadata().size(), name, -1, "size mismatch");
            for (int32_t i = 0; i < scaled_net.metadata().size(); ++i) {
              check(close(scaled_net[i].reco_pt(), hits.hitScale() * (hits[i].x() + hits[i].y() + hits[i].z())),
                    name,
                    i,
                    "result is not hitScale * (x + y + z)");
            }
          }
        }
      }

      check.finish();
    }

  private:
    // collect the failures, and report them at the end of the event
    class Checker {
    public:
      explicit Checker(edm::EventNumber_t event) : event_(event) {}

      void operator()(bool condition, std::string_view model, int32_t index, std::string_view message) {
        if (condition)
          return;
        if (failures_ < 10) {
          errors_ << "\n  " << model;
          if (index >= 0)
            errors_ << "[" << index << "]";
          errors_ << ": " << message;
        }
        ++failures_;
      }

      void finish() const {
        if (failures_ > 0) {
          throw cms::Exception("TestFailed")
              << "Event " << event_ << ": " << failures_ << " checks failed" << errors_.str();
        }
      }

    private:
      const edm::EventNumber_t event_;
      int failures_ = 0;
      std::ostringstream errors_;
    };

    static bool close(float a, float b) { return std::abs(a - b) <= 1e-5f + 1e-4f * std::abs(b); }

    template <typename T>
    static bool isProbability(T const& values) {
      float sum = 0.f;
      for (int k = 0; k < values.size(); ++k) {
        if (not(values(k) >= 0.f and values(k) <= 1.f))
          return false;
        sum += values(k);
      }
      return std::abs(sum - 1.f) < 1e-4f;
    }

    const edm::EDGetTokenT<portabletest::ParticleHostCollection> particles_token_;
    const edm::EDGetTokenT<portabletest::SimpleNetHostCollection> simple_net_token_;
    const edm::EDGetTokenT<portabletest::SimpleNetHostCollection> simple_net_feature_major_token_;
    const edm::EDGetTokenT<portabletest::SimpleNetHostCollection> simple_net_minibatch_token_;
    const edm::EDGetTokenT<portabletest::SimpleNetHostCollection> masked_net_token_;
    const edm::EDGetTokenT<portabletest::SimpleNetHostCollection> masked_net_reversed_token_;
    const edm::EDGetTokenT<portabletest::SimpleNetHostCollection> masked_net_empty_mask_token_;
    const edm::EDGetTokenT<portabletest::MultiHeadNetHostCollection> multi_head_net_token_;
    const edm::EDGetTokenT<portabletest::LogitsHostCollection> logits_token_;
    const edm::EDGetTokenT<portabletest::LogitsHostCollection> logits_minibatch_token_;
    const edm::EDGetTokenT<portabletest::HitHostCollection> hits_token_;
    const edm::EDGetTokenT<portabletest::SimpleNetHostCollection> scaled_net_token_;
    const edm::EDGetTokenT<portabletest::SimpleNetHostCollection> scaled_net_minibatch_token_;
  };

}  // namespace onnxtest

DEFINE_FWK_MODULE(onnxtest::InspectionSink);
