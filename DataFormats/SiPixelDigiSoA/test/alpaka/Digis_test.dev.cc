#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisDevice.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisHost.h"
#include "DataFormats/SiPixelDigiSoA/interface/alpaka/SiPixelDigisSoACollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "Digis_test.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::testDigisSoA {

  class TestFillKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, SiPixelDigisSoAView digi_view) const {
      for (int32_t j : cms::alpakatools::uniform_elements(acc, digi_view.metadata().size())) {
        digi_view[j].clus() = j;
        digi_view[j].rawIdArr() = j * 2;
        digi_view[j].xx() = j * 3;
        digi_view[j].moduleId() = j * 4;
      }
    }
  };

  class TestVerifyKernel {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, SiPixelDigisSoAConstView digi_view) const {
      for (uint32_t j : cms::alpakatools::uniform_elements(acc, digi_view.metadata().size())) {
        ALPAKA_ASSERT_ACC(digi_view[j].clus() == int(j));
        ALPAKA_ASSERT_ACC(digi_view[j].rawIdArr() == j * 2);
        ALPAKA_ASSERT_ACC(digi_view[j].xx() == j * 3);
        ALPAKA_ASSERT_ACC(digi_view[j].moduleId() == j * 4);
      }
    }
  };

  void runKernels(SiPixelDigisSoAView digi_view, Queue& queue) {
    uint32_t items = 64;
    uint32_t groups = cms::alpakatools::divide_up_by(digi_view.metadata().size(), items);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel{}, digi_view);
    alpaka::exec<Acc1D>(queue, workDiv, TestVerifyKernel{}, digi_view);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testDigisSoA
