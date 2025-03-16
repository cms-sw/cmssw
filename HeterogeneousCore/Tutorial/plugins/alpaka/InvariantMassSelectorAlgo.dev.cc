#include <alpaka/alpaka.hpp>

#include "DataFormats/HeterogeneousTutorial/interface/JetsSoA.h"
#include "DataFormats/HeterogeneousTutorial/interface/TripletsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/atomicInc.h"
#include "HeterogeneousCore/AlpakaInterface/interface/atomicIncSaturate.h"
#include "HeterogeneousCore/Tutorial/interface/JetsSelectionSoA.h"

#include "InvariantMassSelectorAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial {

  struct ApplySelectionsKernel {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  InvariantMassSelection const* cuts,
                                  JetsSoA::ConstView jets,
                                  JetsSelectionSoA::View selection,
                                  int32_t* size) const {
      // Possible optimisations:
      //   - mark pointers as __restrict__
      //   - count the selected jets in a per-group shared variable, and update the global one only once
      //   - count the selected jets in a per-thread variable, and update the per-group one only once

      // Make a strided loop over the kernel grid, covering up to jets size elements.
      for (int32_t i : cms::alpakatools::uniform_elements(acc, jets.metadata().size())) {
        auto const& jet = jets[i];
        if (jet.pt() >= cuts->pT_min and jet.pt() < cuts->pT_max and fabsf(jet.eta()) >= cuts->eta_min and
            fabsf(jet.eta()) < cuts->eta_max) {
          selection[i] = true;
          // Increment the number of selected jets atomically across all kernel blocks.
          atomicInc(acc, size, alpaka::hierarchy::Blocks{});
        } else {
          selection[i] = false;
        }
      }
    }
  };

  void InvariantMassSelectorAlgo::applySelections(Queue& queue,
                                                  InvariantMassSelection const* cuts,
                                                  JetsSoA::ConstView const& jets,
                                                  JetsSelectionSoA::View& selection,
                                                  int32_t* size) {
    // Use 64 items per group.
    // This value is arbitrary, but it's a reasonable starting point.
    uint32_t items = 64;

    // Use as many groups as needed to cover the whole problem.
    // If this value is too large, a smaller number of blocks can give better performance.
    uint32_t groups = cms::alpakatools::divide_up_by(jets.metadata().size(), items);

    // Launch the kernel.
    auto grid = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue, grid, ApplySelectionsKernel{}, cuts, jets, selection, size);
  }

  struct FindDoubletsKernel {
    ALPAKA_FN_ACC void operator()(Acc2D const& acc,
                                  InvariantMassSelection const* cuts,
                                  JetsSoA::ConstView jets,
                                  JetsSelectionSoA::ConstView selection,
                                  TripletsSoA::View ntuplets) const {
      // Possible optimisations:
      //   - mark pointers as __restrict__
      //   - split the loop along X and Y, and cache the properties of the outermost loop

      // Make a 2D strided loop over the kernel grid, covering up to jets size x jets size elements.
      Vec2D size = Vec2D{jets.metadata().size(), jets.metadata().size()};
      for (Vec2D ij : cms::alpakatools::uniform_elements_nd(acc, size)) {
        int i = ij[0];  // outermost loop
        int j = ij[1];  // innermost loop

        // Check each combination only once.
        if (i >= j) {
          continue;
        }

        // Skip jets that did not pass the cuts
        if (not selection[i].valid() or not selection[j].valid()) {
          continue;
        }

        auto const& jet_i = jets[i];
        float px_i = jet_i.pt() * alpaka::math::cos(acc, jet_i.phi());
        float py_i = jet_i.pt() * alpaka::math::sin(acc, jet_i.phi());
        float pz_i = jet_i.pt() * alpaka::math::sinh(acc, jet_i.eta());
        float m_i = jet_i.mass();

        auto const& jet_j = jets[j];
        float px_j = jet_j.pt() * alpaka::math::cos(acc, jet_j.phi());
        float py_j = jet_j.pt() * alpaka::math::sin(acc, jet_j.phi());
        float pz_j = jet_j.pt() * alpaka::math::sinh(acc, jet_j.eta());
        float m_j = jet_i.mass();

        // Compute the invariant mass of the two jets.
        float m2 = (px_i + px_j) * (px_i + px_j) +  //
                   (py_i + py_j) * (py_i + py_j) +  //
                   (pz_i + pz_j) * (pz_i + pz_j) -  //
                   (m_i + m_j) * (m_i + m_j);

        if (m2 >= cuts->mass_min * cuts->mass_min and m2 < cuts->mass_max * cuts->mass_max) {
          // Increment the number of ntuplets atomically across all kernel blocks.
          int index = atomicIncSaturate(acc, &ntuplets.size(), ntuplets.metadata().size(), alpaka::hierarchy::Blocks{});

          // Check that index did not overflow.
          if (index >= ntuplets.metadata().size()) {
            // TODO print only once
            printf("FindDoubletsKernel: found too many ntuplets, %d vs %d.\nAdditional ntuplets will be ignored.\n",
                   index,
                   ntuplets.metadata().size());
            continue;
          }

          // Store the doublet.
          ntuplets[index] = {i, j, kEmpty};
        }
      }
    }
  };

  void InvariantMassSelectorAlgo::findDoulets(Queue& queue,
                                              InvariantMassSelection const* cuts,
                                              JetsSoA::ConstView const& jets,
                                              JetsSelectionSoA::ConstView const& selection,
                                              TripletsSoA::View& ntuplets) {
    // Possible optimisations:
    //   - optimise the group and grid size

    // Use 8 x 8 = 64 items per group.
    // This value is arbitrary, but it's a reasonable starting point.
    uint32_t items = 8;

    // Use as many groups as needed to cover the whole problem.
    // If this value is too large, a smaller number of blocks can give better performance.
    uint32_t groups = cms::alpakatools::divide_up_by(jets.metadata().size(), items);

    // Launch the kernel.
    auto grid = cms::alpakatools::make_workdiv<Acc2D>({groups, groups}, {items, items});
    alpaka::exec<Acc2D>(queue, grid, FindDoubletsKernel{}, cuts, jets, selection, ntuplets);
  }

  struct FindTripletsKernel {
    ALPAKA_FN_ACC void operator()(Acc3D const& acc,
                                  InvariantMassSelection const* cuts,
                                  JetsSoA::ConstView jets,
                                  JetsSelectionSoA::ConstView selection,
                                  TripletsSoA::View ntuplets) const {
      // Possible optimisations:
      //   - mark pointers as __restrict__
      //   - split the loop along X, Y, and Z, and cache the properties of the outermost loops

      // Make a 3D strided loop over the kernel grid, covering up to jets size x jets size x jets size elements.
      Vec3D size = Vec3D{jets.metadata().size(), jets.metadata().size(), jets.metadata().size()};
      for (Vec3D ijk : cms::alpakatools::uniform_elements_nd(acc, size)) {
        int i = ijk[0];  // outermost loop
        int j = ijk[1];
        int k = ijk[2];  // innermost loop

        // Check each combination only once.
        if (i >= j or i >= k or j >= k) {
          continue;
        }

        // Skip jets that did not pass the cuts
        if (not selection[i].valid() or not selection[j].valid() or not selection[k].valid()) {
          continue;
        }

        auto const& jet_i = jets[i];
        float px_i = jet_i.pt() * alpaka::math::cos(acc, jet_i.phi());
        float py_i = jet_i.pt() * alpaka::math::sin(acc, jet_i.phi());
        float pz_i = jet_i.pt() * alpaka::math::sinh(acc, jet_i.eta());
        float m_i = jet_i.mass();

        auto const& jet_j = jets[j];
        float px_j = jet_j.pt() * alpaka::math::cos(acc, jet_j.phi());
        float py_j = jet_j.pt() * alpaka::math::sin(acc, jet_j.phi());
        float pz_j = jet_j.pt() * alpaka::math::sinh(acc, jet_j.eta());
        float m_j = jet_i.mass();

        auto const& jet_k = jets[k];
        float px_k = jet_k.pt() * alpaka::math::cos(acc, jet_k.phi());
        float py_k = jet_k.pt() * alpaka::math::sin(acc, jet_k.phi());
        float pz_k = jet_k.pt() * alpaka::math::sinh(acc, jet_k.eta());
        float m_k = jet_i.mass();

        // Compute the invariant mass of the two jets.
        float m2 = (px_i + px_j + px_k) * (px_i + px_j + px_k) +  //
                   (py_i + py_j + py_k) * (py_i + py_j + py_k) +  //
                   (pz_i + pz_j + pz_k) * (pz_i + pz_j + pz_k) -  //
                   (m_i + m_j + m_k) * (m_i + m_j + m_k);

        if (m2 >= cuts->mass_min * cuts->mass_min and m2 < cuts->mass_max * cuts->mass_max) {
          // Increment the number of ntuplets atomically across all kernel blocks.
          int index = atomicIncSaturate(acc, &ntuplets.size(), ntuplets.metadata().size(), alpaka::hierarchy::Blocks{});

          // Check that index did not overflow.
          if (index >= ntuplets.metadata().size()) {
            // TODO print only once
            printf("FindTripletsKernel: found too many ntuplets, %d vs %d.\nAdditional ntuplets will be ignored.\n",
                   index,
                   ntuplets.metadata().size());
            continue;
          }

          // Store the doublet.
          ntuplets[index] = {i, j, k};
        }
      }
    }
  };

  void InvariantMassSelectorAlgo::findTriplets(Queue& queue,
                                               InvariantMassSelection const* cuts,
                                               JetsSoA::ConstView const& jets,
                                               JetsSelectionSoA::ConstView const& selection,
                                               TripletsSoA::View& ntuplets) {
    // Possible optimisations:
    //   - optimise the group and grid size

    // Use 4 x 4 x 4 = 64 items per group.
    // This value is arbitrary, but it's a reasonable starting point.
    uint32_t items = 4;

    // Use as many groups as needed to cover the whole problem.
    // If this value is too large, a smaller number of blocks can give better performance.
    uint32_t groups = cms::alpakatools::divide_up_by(jets.metadata().size(), items);

    // Launch the kernel.
    auto grid = cms::alpakatools::make_workdiv<Acc3D>({groups, groups, groups}, {items, items, items});
    alpaka::exec<Acc3D>(queue, grid, FindTripletsKernel{}, cuts, jets, selection, ntuplets);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial
