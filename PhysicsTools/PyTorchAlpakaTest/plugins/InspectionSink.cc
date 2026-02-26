#include <cassert>

#include <fmt/format.h>

#include "DataFormats/PortableTestObjects/interface/ParticleHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/ImageHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/LogitsHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/SimpleNetHostCollection.h"
#include "DataFormats/PortableTestObjects/interface/MultiHeadNetHostCollection.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "PhysicsTools/PyTorchAlpakaTest/interface/Environment.h"

namespace torchtest {

  inline edm::InputTag getBackendTag(edm::InputTag const& tag) {
    return edm::InputTag(tag.label(), "backend", tag.process());
  }

  class InspectionSink : public edm::stream::EDAnalyzer<> {
  public:
    InspectionSink(const edm::ParameterSet& params)
        : environment_{static_cast<Environment>(params.getUntrackedParameter<int>("environment"))},
          particles_token_{consumes(params.getParameter<edm::InputTag>("particles"))},
          simple_net_token_{consumes(params.getParameter<edm::InputTag>("simple_net"))},
          masked_net_token_{consumes(params.getParameter<edm::InputTag>("masked_net"))},
          multi_head_net_token_{consumes(params.getParameter<edm::InputTag>("multi_head_net"))},
          images_token_{consumes(params.getParameter<edm::InputTag>("images"))},
          logits_token_{consumes(params.getParameter<edm::InputTag>("resnet18"))},
          particles_backend_{consumes(getBackendTag(params.getParameter<edm::InputTag>("particles")))},
          simple_net_backend_{consumes(getBackendTag(params.getParameter<edm::InputTag>("simple_net")))},
          masked_net_backend_{consumes(getBackendTag(params.getParameter<edm::InputTag>("masked_net")))},
          multi_head_net_backend_{consumes(getBackendTag(params.getParameter<edm::InputTag>("multi_head_net")))},
          images_backend_{consumes(getBackendTag(params.getParameter<edm::InputTag>("images")))},
          logits_backend_{consumes(getBackendTag(params.getParameter<edm::InputTag>("resnet18")))} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.addUntracked<int>("environment", static_cast<int>(Environment::kProduction));
      desc.add<edm::InputTag>("particles");
      desc.add<edm::InputTag>("simple_net");
      desc.add<edm::InputTag>("masked_net");
      desc.add<edm::InputTag>("multi_head_net");
      desc.add<edm::InputTag>("images");
      desc.add<edm::InputTag>("resnet18");
      descriptions.addWithDefaultLabel(desc);
    }

    void analyze(edm::Event const& event, edm::EventSetup const&) override {
      if (environment_ >= Environment::kDevelopment) {
        constexpr int total_len = 72;
        auto label = fmt::format("EVENT: {}", event.id().event());
        int pad = total_len - static_cast<int>(label.size());
        int pad_left = pad / 2;
        int pad_right = pad - pad_left - 1;
        fmt::print("\n{0} {1} {2}\n\n", std::string(pad_left, '-'), label, std::string(pad_right, '-'));
      }

      // particles
      auto particles_handle = event.getHandle(particles_token_);
      auto simple_net_handle = event.getHandle(simple_net_token_);
      auto masked_net_handle = event.getHandle(masked_net_token_);
      auto multi_head_net_handle = event.getHandle(multi_head_net_token_);
      auto images_handle = event.getHandle(images_token_);
      auto logits_handle = event.getHandle(logits_token_);

      // debug
      if (environment_ >= Environment::kDevelopment) {
        // particles
        if (particles_handle.isValid()) {
          auto const& particles = *particles_handle;
          auto const particles_backend = static_cast<cms::alpakatools::Backend>(event.get(particles_backend_));
          if (simple_net_handle.isValid() || masked_net_handle.isValid() || multi_head_net_handle.isValid()) {
            print(particles.const_view(), cms::alpakatools::toString(particles_backend));
            // assert ranges
            for (int32_t idx = 0; idx < particles.const_view().metadata().size(); idx++) {
              assert(0.0 <= particles.const_view().pt()[idx] && particles.const_view().pt()[idx] <= 1.0);
              assert(0.0 <= particles.const_view().eta()[idx] && particles.const_view().eta()[idx] <= 1.0);
              assert(0.0 <= particles.const_view().phi()[idx] && particles.const_view().phi()[idx] <= 1.0);
            }
          }
          // simple_net
          if (simple_net_handle.isValid()) {
            auto const& simple_net = *simple_net_handle;
            auto const simple_net_backend = static_cast<cms::alpakatools::Backend>(event.get(simple_net_backend_));
            print(simple_net.const_view(), cms::alpakatools::toString(simple_net_backend));
          }
          // masked_net
          if (masked_net_handle.isValid()) {
            auto const& masked_net = *masked_net_handle;
            auto const masked_net_backend = static_cast<cms::alpakatools::Backend>(event.get(masked_net_backend_));
            print(masked_net.const_view(), cms::alpakatools::toString(masked_net_backend), "MaskedNetCollection");
            // assert, eta feature is always masked and do not contribute
            for (int32_t idx = 0; idx < masked_net.const_view().metadata().size(); idx++) {
              assert(masked_net.const_view().reco_pt()[idx] ==
                     particles.const_view().pt()[idx] + particles.const_view().phi()[idx]);
            }
          }
          // multihead_net
          if (multi_head_net_handle.isValid()) {
            auto const& multi_head_net = *multi_head_net_handle;
            auto const multi_head_net_backend =
                static_cast<cms::alpakatools::Backend>(event.get(multi_head_net_backend_));
            print(multi_head_net.const_view(), cms::alpakatools::toString(multi_head_net_backend));
            // assert, regressiona and classification heads
            const int dims = portabletest::ClassificationHead::RowsAtCompileTime;
            for (int32_t idx = 0; idx < multi_head_net.const_view().metadata().size(); idx++) {
              auto r = multi_head_net.const_view()[idx].regression_head();
              auto c = multi_head_net.const_view()[idx].classification_head();
              float sum = 0.0;
              for (int i = 0; i < dims; i++) {
                sum += c[i];
              }
              assert(4.0 <= r && r <= 5.0);
              assert(std::abs(sum - 1.0) < 1e-4);
            }
          }
        }
        // images
        if (images_handle.isValid() && logits_handle.isValid()) {
          auto const& images = *images_handle;
          auto const images_backend = static_cast<cms::alpakatools::Backend>(event.get(images_backend_));
          print(images.const_view(), cms::alpakatools::toString(images_backend));

          auto const& logits = *logits_handle;
          auto const logits_backend = static_cast<cms::alpakatools::Backend>(event.get(logits_backend_));
          print(logits.const_view(), cms::alpakatools::toString(logits_backend));

          const int dims = portabletest::LogitsType::RowsAtCompileTime;
          for (int32_t idx = 0; idx < logits.const_view().metadata().size(); idx++) {
            float sum = 0.0f;
            const auto& logit = logits.const_view()[idx];
            for (int i = 0; i < dims; i++) {
              sum += logit.logits()[i];
            }
            assert(std::abs(sum - 1.0) < 1e-4);
          }
        }
      }
    }

    void beginStream(edm::StreamID stream) override {
      id_ = stream.value();
      if (environment_ >= Environment::kDevelopment) {
        fmt::print("=========================================================================\n");
      }
    }

    void endStream() override {
      if (environment_ >= Environment::kDevelopment) {
        fmt::print("=========================================================================\n");
        fmt::print("[INFO] OK - HeterogeneousML on CMSSW stream: {}\n", id_);
        fmt::print("=========================================================================\n");
      }
    }

  private:
    int id_ = 0;
    const Environment environment_;

    const edm::EDGetTokenT<portabletest::ParticleHostCollection> particles_token_;
    const edm::EDGetTokenT<portabletest::SimpleNetHostCollection> simple_net_token_;
    const edm::EDGetTokenT<portabletest::SimpleNetHostCollection> masked_net_token_;
    const edm::EDGetTokenT<portabletest::MultiHeadNetHostCollection> multi_head_net_token_;
    const edm::EDGetTokenT<portabletest::ImageHostCollection> images_token_;
    const edm::EDGetTokenT<portabletest::LogitsHostCollection> logits_token_;

    const edm::EDGetTokenT<unsigned short> particles_backend_;
    const edm::EDGetTokenT<unsigned short> simple_net_backend_;
    const edm::EDGetTokenT<unsigned short> masked_net_backend_;
    const edm::EDGetTokenT<unsigned short> multi_head_net_backend_;
    const edm::EDGetTokenT<unsigned short> images_backend_;
    const edm::EDGetTokenT<unsigned short> logits_backend_;

    const int32_t kMaxView = 5;

    void print(const portabletest::LogitsHostCollection::ConstView& logits, const std::string_view logits_backend) {
      const int rows = portabletest::LogitsType::RowsAtCompileTime;
      constexpr auto line = "+------+------+------+------+------+------+------+------+------+------+\n";
      fmt::memory_buffer buffer;

      fmt::format_to(std::back_inserter(buffer), "[DEBUG] LogitsCollection[{}x{}]:\n", logits.metadata().size(), rows);
      fmt::format_to(std::back_inserter(buffer), "{}", line);
      fmt::format_to(std::back_inserter(buffer), "|");
      for (int i = 0; i < rows; ++i) {
        fmt::format_to(std::back_inserter(buffer), " {:>.2f} |", logits[0].logits()[i]);
      }
      fmt::format_to(std::back_inserter(buffer), "\n{}", line);
      fmt::print("{}\n", fmt::to_string(buffer));
    }

    void print(const portabletest::ImageHostCollection::ConstView& images, const std::string_view images_backend) {
      const auto size = images.metadata().size();
      const int rows = portabletest::ColorChannel::RowsAtCompileTime;
      const int cols = portabletest::ColorChannel::ColsAtCompileTime;
      constexpr auto line = "+-------+-------+-------+-------+-------+-------+-------+-------+-------+\n";
      fmt::memory_buffer buffer;

      fmt::format_to(
          std::back_inserter(buffer), "[DEBUG] ImageCollection[{}x3x{}x{}] ({}):\n", size, cols, rows, images_backend);
      fmt::format_to(std::back_inserter(buffer), "{}", line);
      const auto& img = images[0];
      auto printChannelFn = [&](auto&& channel, const char* name) {
        constexpr int total_len = 70;
        auto label = fmt::format("{}", name);
        int pad = total_len - static_cast<int>(label.size());
        int pad_left = pad / 2;
        int pad_right = pad - pad_left - 1;
        fmt::format_to(std::back_inserter(buffer), "|");
        fmt::format_to(
            std::back_inserter(buffer), "{0} {1} {2}", std::string(pad_left, ' '), label, std::string(pad_right, ' '));
        fmt::format_to(std::back_inserter(buffer), "|\n");
        fmt::format_to(std::back_inserter(buffer), "{}", line);
        for (int i = 0; i < cols; ++i) {
          fmt::format_to(std::back_inserter(buffer), "|");
          for (int j = 0; j < rows; ++j) {
            fmt::format_to(std::back_inserter(buffer), " {:>2.3f} |", channel(i, j));
          }
          fmt::format_to(std::back_inserter(buffer), "\n");
        }
        fmt::format_to(std::back_inserter(buffer), "{}", line);
      };

      printChannelFn(img.r(), "RED");
      printChannelFn(img.g(), "GREEN");
      printChannelFn(img.b(), "BLUE");
      fmt::print("{}\n", fmt::to_string(buffer));
    }

    void print(const portabletest::MultiHeadNetHostCollection::ConstView& multi_head_net,
               const std::string_view multi_head_net_backend) {
      constexpr auto line = "+-------+-----------------+-------+-------+-------+\n";
      const auto size = multi_head_net.metadata().size();
      fmt::memory_buffer buffer;

      // Header message
      fmt::format_to(
          std::back_inserter(buffer), "[DEBUG] MultiHeadNetCollection[{}] ({}):\n", size, multi_head_net_backend);
      fmt::format_to(std::back_inserter(buffer), "{}", "+-------+-----------------+-----------------------+\n");
      fmt::format_to(std::back_inserter(buffer),
                     "| {:>5} | {:>15} | {:>21} |\n",
                     "index",
                     "regression_head",
                     "classification_head");
      fmt::format_to(std::back_inserter(buffer), "{}", line);

      // Table rows (preview)
      int32_t range = (environment_ >= Environment::kTest) ? size : std::min<int32_t>(kMaxView, size);
      for (int32_t i = 0; i < range; ++i) {
        fmt::format_to(std::back_inserter(buffer),
                       "| {:5d} | {:>15.2f} | {:5.2f} | {:5.2f} | {:5.2f} |\n",
                       static_cast<int>(i),
                       multi_head_net[i].regression_head(),
                       multi_head_net[i].classification_head()[0],
                       multi_head_net[i].classification_head()[1],
                       multi_head_net[i].classification_head()[2]);
      }

      // Ellipsis row if truncated
      if (range < kMaxView) {
        fmt::format_to(std::back_inserter(buffer),
                       "| {:>5} | {:>15} | {:>5} | {:>5} | {:>5} |\n",
                       "...",
                       "...",
                       "...",
                       "...",
                       "...");
      }

      fmt::format_to(std::back_inserter(buffer), "{}", line);
      fmt::print("{}\n", fmt::to_string(buffer));
    }

    void print(const portabletest::SimpleNetHostCollection::ConstView& simple_net,
               const std::string_view simple_net_backend,
               const std::string& label = "SimpleNetCollection") {
      constexpr auto line = "+-------+---------+\n";
      const auto size = simple_net.metadata().size();
      fmt::memory_buffer buffer;

      // Header message
      fmt::format_to(std::back_inserter(buffer), "[DEBUG] {}[{}] ({}):\n", label, size, simple_net_backend);
      fmt::format_to(std::back_inserter(buffer), "{}", line);
      fmt::format_to(std::back_inserter(buffer), "| {:>5} | {:>7} |\n", "index", "reco_pt");
      fmt::format_to(std::back_inserter(buffer), "{}", line);

      // Table rows (preview)
      int32_t range = (environment_ >= Environment::kTest) ? size : std::min<int32_t>(kMaxView, size);
      for (int32_t i = 0; i < range; ++i) {
        fmt::format_to(
            std::back_inserter(buffer), "| {:5d} | {:7.2f} |\n", static_cast<int>(i), simple_net[i].reco_pt());
      }

      // Ellipsis row if truncated
      if (range < kMaxView) {
        fmt::format_to(std::back_inserter(buffer), "| {:>5} | {:>7} |\n", "...", "...");
      }

      fmt::format_to(std::back_inserter(buffer), "{}", line);
      fmt::print("{}\n", fmt::to_string(buffer));
    }

    void print(const portabletest::ParticleHostCollection::ConstView& particles,
               const std::string_view particles_backend) {
      constexpr auto line = "+-------+---------+---------+---------+\n";
      const auto size = particles.metadata().size();
      fmt::memory_buffer buffer;

      // Header message
      fmt::format_to(std::back_inserter(buffer), "[DEBUG] ParticleCollection[{}] ({}):\n", size, particles_backend);
      fmt::format_to(std::back_inserter(buffer), "{}", line);
      fmt::format_to(std::back_inserter(buffer), "| {:>5} | {:>7} | {:>7} | {:>7} |\n", "index", "pt", "eta", "phi");
      fmt::format_to(std::back_inserter(buffer), "{}", line);

      // Table rows (preview)
      int32_t range = (environment_ >= Environment::kTest) ? size : std::min<int32_t>(kMaxView, size);
      for (int32_t i = 0; i < range; ++i) {
        fmt::format_to(std::back_inserter(buffer),
                       "| {:5d} | {:7.2f} | {:7.2f} | {:7.2f} |\n",
                       static_cast<int>(i),
                       particles[i].pt(),
                       particles[i].eta(),
                       particles[i].phi());
      }

      // Ellipsis row if truncated
      if (range < kMaxView) {
        fmt::format_to(std::back_inserter(buffer), "| {:>5} | {:>7} | {:>7} | {:>7} |\n", "...", "...", "...", "...");
      }

      fmt::format_to(std::back_inserter(buffer), "{}", line);
      fmt::print("{}\n", fmt::to_string(buffer));
    }
  };

}  // namespace torchtest

DEFINE_FWK_MODULE(torchtest::InspectionSink);
