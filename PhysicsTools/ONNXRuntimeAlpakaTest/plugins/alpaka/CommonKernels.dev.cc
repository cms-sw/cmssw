#include <algorithm>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "PhysicsTools/ONNXRuntimeAlpakaTest/plugins/alpaka/CommonKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest::kernels {

  void randomFillParticleCollection(Queue& queue, portabletest::ParticleDeviceCollection& particles, uint32_t seed) {
    const uint32_t threads_per_block = 64;
    const uint32_t blocks_per_grid =
        cms::alpakatools::divide_up_by(particles.view().metadata().size(), threads_per_block);
    const auto grid = cms::alpakatools::make_workdiv<Acc1D>(std::max(blocks_per_grid, 1u), threads_per_block);

    alpaka::exec<Acc1D>(
        queue,
        grid,
        [] ALPAKA_FN_ACC(Acc1D const& acc, portabletest::ParticleDeviceCollection::View particles_view, uint32_t seed) {
          for (int32_t thread_idx : cms::alpakatools::uniform_elements(acc, particles_view.metadata().size())) {
            auto rnd_gen = alpaka::rand::engine::createDefault(acc, seed, thread_idx);
            auto dist = alpaka::rand::distribution::createUniformReal<float>(acc);
            particles_view[thread_idx].pt() = dist(rnd_gen);
            particles_view[thread_idx].eta() = dist(rnd_gen);
            particles_view[thread_idx].phi() = dist(rnd_gen);
          }
        },
        particles.view(),
        seed);
  }

  void randomFillImageCollection(Queue& queue, portabletest::ImageDeviceCollection& images, uint32_t seed) {
    constexpr int32_t rows = portabletest::ColorChannel::RowsAtCompileTime;
    constexpr int32_t cols = portabletest::ColorChannel::ColsAtCompileTime;
    const uint32_t threads_per_block = 64;
    const uint32_t blocks_per_grid =
        cms::alpakatools::divide_up_by(images.view().metadata().size() * rows * cols, threads_per_block);
    const auto grid = cms::alpakatools::make_workdiv<Acc1D>(std::max(blocks_per_grid, 1u), threads_per_block);

    alpaka::exec<Acc1D>(
        queue,
        grid,
        [] ALPAKA_FN_ACC(Acc1D const& acc, portabletest::ImageDeviceCollection::View images_view, uint32_t seed) {
          for (int32_t index : cms::alpakatools::uniform_elements(acc, images_view.metadata().size() * rows * cols)) {
            int32_t b = index / (rows * cols);
            int32_t i = (index / cols) % rows;
            int32_t j = index % cols;
            auto rnd_gen = alpaka::rand::engine::createDefault(acc, seed, index);
            auto dist = alpaka::rand::distribution::createUniformReal<float>(acc);
            images_view[b].r()(i, j) = dist(rnd_gen);
            images_view[b].g()(i, j) = dist(rnd_gen);
            images_view[b].b()(i, j) = dist(rnd_gen);
          }
        },
        images.view(),
        seed);
  }

  void randomFillHitCollection(Queue& queue, portabletest::HitDeviceCollection& hits, uint32_t seed) {
    const uint32_t threads_per_block = 64;
    const uint32_t blocks_per_grid = cms::alpakatools::divide_up_by(hits.view().metadata().size(), threads_per_block);
    const auto grid = cms::alpakatools::make_workdiv<Acc1D>(std::max(blocks_per_grid, 1u), threads_per_block);

    alpaka::exec<Acc1D>(
        queue,
        grid,
        [] ALPAKA_FN_ACC(Acc1D const& acc, portabletest::HitDeviceCollection::View hits_view, uint32_t seed) {
          if (cms::alpakatools::once_per_grid(acc)) {
            hits_view.hitScale() = 1.5f;
          }
          for (int32_t thread_idx : cms::alpakatools::uniform_elements(acc, hits_view.metadata().size())) {
            auto rnd_gen = alpaka::rand::engine::createDefault(acc, seed, thread_idx);
            auto dist = alpaka::rand::distribution::createUniformReal<float>(acc);
            hits_view[thread_idx].x() = dist(rnd_gen);
            hits_view[thread_idx].y() = dist(rnd_gen);
            hits_view[thread_idx].z() = dist(rnd_gen);
          }
        },
        hits.view(),
        seed);
  }

  void fillMask(Queue& queue, portabletest::MaskDeviceCollection& mask) {
    const uint32_t threads_per_block = 64;
    const uint32_t blocks_per_grid = cms::alpakatools::divide_up_by(mask.view().metadata().size(), threads_per_block);
    const auto grid = cms::alpakatools::make_workdiv<Acc1D>(std::max(blocks_per_grid, 1u), threads_per_block);

    alpaka::exec<Acc1D>(
        queue,
        grid,
        [] ALPAKA_FN_ACC(Acc1D const& acc, portabletest::MaskDeviceCollection::View mask_view) {
          for (int32_t thread_idx : cms::alpakatools::uniform_elements(acc, mask_view.metadata().size())) {
            // mask eta feature only
            mask_view[thread_idx].mask()[0] = 0;
            mask_view[thread_idx].mask()[1] = 1;
            mask_view[thread_idx].mask()[2] = 0;
          }
        },
        mask.view());
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::onnxtest::kernels
