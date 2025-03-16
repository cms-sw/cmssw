#ifndef HeterogeneousCore_Tutorial_plugins_alpaka_PFJetsSoACorrectorAlgo_h
#define HeterogeneousCore_Tutorial_plugins_alpaka_PFJetsSoACorrectorAlgo_h

#include "DataFormats/HeterogeneousTutorial/interface/JetsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/Tutorial/interface/Table.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial {

  using namespace ::tutorial;

  struct PFJetsSoACorrectorAlgo {
    static void applyJetCorrections(Queue& queue,
                                    JetsSoA::ConstView const& input,
                                    JetsSoA::View& output,
                                    Table const& correction);
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial

#endif  // HeterogeneousCore_Tutorial_plugins_alpaka_PFJetsSoACorrectorAlgo_h
