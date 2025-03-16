#ifndef HeterogeneousCore_Tutorial_plugins_alpaka_InvariantMassSelectorAlgo_h
#define HeterogeneousCore_Tutorial_plugins_alpaka_InvariantMassSelectorAlgo_h

#include "DataFormats/HeterogeneousTutorial/interface/JetsSoA.h"
#include "DataFormats/HeterogeneousTutorial/interface/TripletsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/Tutorial/interface/JetsSelectionSoA.h"

namespace tutorial {

  struct InvariantMassSelection {
    float pT_min;
    float pT_max;
    float eta_min;
    float eta_max;
    float mass_min;
    float mass_max;
  };

}  // namespace tutorial

namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial {

  using namespace ::tutorial;

  struct InvariantMassSelectorAlgo {
    static void applySelections(
        Queue& queue,
        InvariantMassSelection const* cuts,  // pointer to InvariantMassSelection in device memory
        JetsSoA::ConstView const& jets,      // SoA with input jets, in device memory
        JetsSelectionSoA::View& selection,   // SoA to fill with selected jets, in device memory
        int32_t* size);                      // number of selected jets, in pinned host memory accessible on the device

    static void findDoulets(  //
        Queue& queue,
        InvariantMassSelection const* cuts,            // pointer to InvariantMassSelection in device memory
        JetsSoA::ConstView const& jets,                // SoA with input jets, in device memory
        JetsSelectionSoA::ConstView const& selection,  // SoA with jet selection, in device memory
        TripletsSoA::View& ntuplets);                  // SoA to fill with pairs, in device memory

    static void findTriplets(  //
        Queue& queue,
        InvariantMassSelection const* cuts,            // pointer to InvariantMassSelection in device memory
        JetsSoA::ConstView const& jets,                // SoA with input jets, in device memory
        JetsSelectionSoA::ConstView const& selection,  // SoA with jet selection, in device memory
        TripletsSoA::View& ntuplets);                  // SoA to fill with triplets, in device memory
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial

#endif  // HeterogeneousCore_Tutorial_plugins_alpaka_InvariantMassSelectorAlgo_h
