#include <alpaka/alpaka.hpp>

#include "CondFormats/SiStripObjects/interface/SiStripClusterizerConditionsSoA.h"
#include "CondFormats/SiStripObjects/interface/SiStripClusterizerConditionsHost.h"
#include "CondFormats/SiStripObjects/interface/alpaka/SiStripClusterizerConditionsDevice.h"

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "TestSiStripClusterizerConditionsDevice.h"

using namespace alpaka;

namespace ALPAKA_ACCELERATOR_NAMESPACE::testConditionsSoA {
  class TestFillKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  SiStripClusterizerConditionsDetToFedsView DetToFeds_view,
                                  SiStripClusterizerConditionsData_fedchView Data_fedchSoA_view,
                                  SiStripClusterizerConditionsData_stripView Data_stripSoA_view,
                                  SiStripClusterizerConditionsData_apvView Data_apvSoA_view) const {
      for (uint32_t j : cms::alpakatools::uniform_elements(acc, DetToFeds_view.metadata().size())) {
        DetToFeds_view[j].detid_() = j * 2;
        DetToFeds_view[j].ipair_() = (uint16_t)((j) % 65536);
        DetToFeds_view[j].fedid_() = (uint16_t)((j + 1) % 65536);
        DetToFeds_view[j].fedch_() = (uint8_t)(j % 256);
      }

      for (uint32_t j : cms::alpakatools::uniform_elements(acc, Data_fedchSoA_view.metadata().size())) {
        Data_fedchSoA_view[j].detID_() = (uint32_t)(j);
        Data_fedchSoA_view[j].iPair_() = (uint16_t)(j % 65536);
        Data_fedchSoA_view[j].invthick_() = (float)(j * 1.0);
      }

      for (uint32_t j : cms::alpakatools::uniform_elements(acc, Data_stripSoA_view.metadata().size())) {
        Data_stripSoA_view[j].noise_() = (uint16_t)(j % 65536);
      }

      for (uint32_t j : cms::alpakatools::uniform_elements(acc, Data_apvSoA_view.metadata().size())) {
        Data_apvSoA_view[j].gain_() = (float)(j * -1.0f);
      }
    }
  };

  class TestVerifyKernel {
  public:
    template <typename TAcc, typename = std::enable_if_t<isAccelerator<TAcc>>>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                  SiStripClusterizerConditionsDetToFedsView DetToFeds_view,
                                  SiStripClusterizerConditionsData_fedchView Data_fedchSoA_view,
                                  SiStripClusterizerConditionsData_stripView Data_stripSoA_view,
                                  SiStripClusterizerConditionsData_apvView Data_apvSoA_view) const {
      for (uint32_t j : cms::alpakatools::uniform_elements(acc, DetToFeds_view.metadata().size())) {
        ALPAKA_ASSERT_ACC(DetToFeds_view[j].detid_() == j * 2);
        ALPAKA_ASSERT_ACC(DetToFeds_view[j].ipair_() == (uint16_t)((j) % 65536));
        ALPAKA_ASSERT_ACC(DetToFeds_view[j].fedid_() == (uint16_t)((j + 1) % 65536));
        ALPAKA_ASSERT_ACC(DetToFeds_view[j].fedch_() == (uint8_t)(j % 256));
      }

      for (uint32_t j : cms::alpakatools::uniform_elements(acc, Data_fedchSoA_view.metadata().size())) {
        ALPAKA_ASSERT_ACC(Data_fedchSoA_view[j].detID_() == (uint32_t)(j));
        ALPAKA_ASSERT_ACC(Data_fedchSoA_view[j].iPair_() == (uint16_t)(j % 65536));
        ALPAKA_ASSERT_ACC(Data_fedchSoA_view[j].invthick_() == (float)(j * 1.0));
      }

      for (uint32_t j : cms::alpakatools::uniform_elements(acc, Data_stripSoA_view.metadata().size())) {
        ALPAKA_ASSERT_ACC(Data_stripSoA_view[j].noise_() == (uint16_t)(j % 65536));
      }

      for (uint32_t j : cms::alpakatools::uniform_elements(acc, Data_apvSoA_view.metadata().size())) {
        ALPAKA_ASSERT_ACC(Data_apvSoA_view[j].gain_() == (float)(j * -1.0f));
      }
    }
  };

  void runKernels(SiStripClusterizerConditionsDetToFedsView DetToFedsView,
                  SiStripClusterizerConditionsData_fedchView Data_fedchView,
                  SiStripClusterizerConditionsData_stripView Data_stripView,
                  SiStripClusterizerConditionsData_apvView Data_apvView,
                  Queue& queue) {
    uint32_t items = 640;
    uint32_t groups = cms::alpakatools::divide_up_by(64, items);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(groups, items);
    alpaka::exec<Acc1D>(queue, workDiv, TestFillKernel{}, DetToFedsView, Data_fedchView, Data_stripView, Data_apvView);
    alpaka::exec<Acc1D>(
        queue, workDiv, TestVerifyKernel{}, DetToFedsView, Data_fedchView, Data_stripView, Data_apvView);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::testConditionsSoA