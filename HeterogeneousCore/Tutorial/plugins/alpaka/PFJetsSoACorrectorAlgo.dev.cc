#include <alpaka/alpaka.hpp>

#include "DataFormats/HeterogeneousTutorial/interface/JetsSoA.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/Tutorial/interface/Table.h"

#include "PFJetsSoACorrectorAlgo.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial {

  using namespace ::tutorial;

  class JetCorrectionsKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  JetsSoA::ConstView input,
                                  JetsSoA::View output,
                                  Table correction) const {
      // make a strided loop over the kernel grid, covering up to "size" elements
      for (int32_t i : cms::alpakatools::uniform_elements(acc, input.metadata().size())) {
        auto entry = input[i];
        float factor = correction.get(entry.pt(), entry.eta());
        output[i] = {entry.pt() * factor, entry.eta(), entry.phi(), entry.mass()};
      }
    }
  };

  void PFJetsSoACorrectorAlgo::applyJetCorrections(Queue& queue,
                                                   JetsSoA::ConstView const& input,
                                                   JetsSoA::View& output,
                                                   Table const& correction) {
    // Use 64 items per group.
    // This value is arbitrary, but it's a reasonable starting point.
    uint32_t items = 64;

    // Use as many groups as needed to cover the whole problem.
    // If this value is too large, a smaller number of blocks can give better performance.
    uint32_t groups = cms::alpakatools::divide_up_by(input.metadata().size(), items);

    auto grid = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue, grid, JetCorrectionsKernel{}, input, output, correction);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::tutorial
