#include "PhysicsTools/PyTorchAlpakaTest/plugins/alpaka/CommonKernels.h"

#include "alpaka/alpaka.hpp"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest::kernels {

  void randomFillParticleCollection(Queue& queue, portabletest::ParticleDeviceCollection& particles) {
    const uint32_t threads_per_block = 64;
    const uint32_t blocks_per_grid = particles.view().metadata().size();
    const auto grid = cms::alpakatools::make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);

    alpaka::exec<Acc1D>(
        queue,
        grid,
        [] ALPAKA_FN_ACC(Acc1D const& acc, portabletest::ParticleDeviceCollection::View particles_view) {
          for (int32_t thread_idx : cms::alpakatools::uniform_elements(acc, particles_view.metadata().size())) {
            auto rnd_gen = alpaka::rand::engine::createDefault(acc, 43, thread_idx);
            auto dist = alpaka::rand::distribution::createUniformReal<float>(acc);
            particles_view[thread_idx].pt() = dist(rnd_gen);
            particles_view[thread_idx].eta() = dist(rnd_gen);
            particles_view[thread_idx].phi() = dist(rnd_gen);
          }
        },
        particles.view());
  }

  struct RandomFillImageCollectionKernel {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc, portabletest::ImageDeviceCollection::View images_view) const {
      Vec3D size = Vec3D{images_view.metadata().size(), 9, 9};
      for (Vec3D index_3d : cms::alpakatools::uniform_elements_nd(acc, size)) {
        // Order OpenCL like not CUDA style (reversed)
        int b = index_3d[0];
        int i = index_3d[1];
        int j = index_3d[2];

        const auto seed = (b + 1) * (i + 1) * (j + 1);
        auto rnd_gen = alpaka::rand::engine::createDefault(acc, 21, seed);
        auto dist = alpaka::rand::distribution::createUniformReal<float>(acc);

        float pixel = dist(rnd_gen);
        images_view[b].r()(i, j) = pixel;
        images_view[b].g()(i, j) = pixel;
        images_view[b].b()(i, j) = pixel;
      }
    }
  };

  void randomFillImageCollection(Queue& queue, portabletest::ImageDeviceCollection& images) {
    const uint32_t items = 4;  // 4x4x4=64
    const uint32_t groups = images.view().metadata().size();
    const auto grid = cms::alpakatools::make_workdiv<Acc3D>({groups, groups, groups}, {items, items, items});

    alpaka::exec<Acc3D>(
        queue,
        grid,
        [] ALPAKA_FN_ACC(Acc3D const& acc, portabletest::ImageDeviceCollection::View images_view) {
          Vec3D size = Vec3D{images_view.metadata().size(), 9, 9};
          for (Vec3D index_3d : cms::alpakatools::uniform_elements_nd(acc, size)) {
            // Order OpenCL like not CUDA style (reversed)
            int b = index_3d[0];
            int i = index_3d[1];
            int j = index_3d[2];

            const auto seed = (b + 1) * (i + 1) * (j + 1);
            auto rnd_gen = alpaka::rand::engine::createDefault(acc, 43, seed);
            auto dist = alpaka::rand::distribution::createUniformReal<float>(acc);

            float pixel = dist(rnd_gen);
            images_view[b].r()(i, j) = pixel;
            images_view[b].g()(i, j) = pixel;
            images_view[b].b()(i, j) = pixel;
          }
        },
        images.view());
  }

  void fillMask(Queue& queue, portabletest::MaskDeviceCollection& mask) {
    const uint32_t threads_per_block = 64;
    const uint32_t blocks_per_grid = mask.view().metadata().size();
    const auto grid = cms::alpakatools::make_workdiv<Acc1D>(blocks_per_grid, threads_per_block);

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

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::torchtest::kernels
