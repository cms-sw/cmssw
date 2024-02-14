#include <cassert>

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/EcalObjects/interface/alpaka/EcalMultifitParametersDevice.h"
#include "CondFormats/EcalObjects/interface/EcalMultifitParametersSoA.h"
#include "CondFormats/DataRecord/interface/EcalMultifitParametersRcd.h"

#include "DataFormats/EcalDigi/interface/EcalConstants.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ModuleFactory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/host.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
  class EcalMultifitParametersHostESProducer : public ESProducer {
  public:
    EcalMultifitParametersHostESProducer(edm::ParameterSet const&);
    ~EcalMultifitParametersHostESProducer() override = default;

    static void fillDescriptions(edm::ConfigurationDescriptions&);
    std::unique_ptr<EcalMultifitParametersHost> produce(EcalMultifitParametersRcd const&);

  private:
    std::vector<float> ebTimeFitParameters_;
    std::vector<float> eeTimeFitParameters_;
    std::vector<float> ebAmplitudeFitParameters_;
    std::vector<float> eeAmplitudeFitParameters_;
  };

  EcalMultifitParametersHostESProducer::EcalMultifitParametersHostESProducer(edm::ParameterSet const& iConfig)
      : ESProducer(iConfig) {
    setWhatProduced(this);

    auto const ebTimeFitParamsFromPSet = iConfig.getParameter<std::vector<double>>("EBtimeFitParameters");
    auto const eeTimeFitParamsFromPSet = iConfig.getParameter<std::vector<double>>("EEtimeFitParameters");
    // Assert that there are as many parameters as the EcalMultiFitParametersSoA expects
    assert(ebTimeFitParamsFromPSet.size() == kNTimeFitParams);
    assert(eeTimeFitParamsFromPSet.size() == kNTimeFitParams);
    ebTimeFitParameters_.assign(ebTimeFitParamsFromPSet.begin(), ebTimeFitParamsFromPSet.end());
    eeTimeFitParameters_.assign(eeTimeFitParamsFromPSet.begin(), eeTimeFitParamsFromPSet.end());

    auto const ebAmplFitParamsFromPSet = iConfig.getParameter<std::vector<double>>("EBamplitudeFitParameters");
    auto const eeAmplFitParamsFromPSet = iConfig.getParameter<std::vector<double>>("EEamplitudeFitParameters");
    // Assert that there are as many parameters as the EcalMultiFitParametersSoA expects
    assert(ebAmplFitParamsFromPSet.size() == kNAmplitudeFitParams);
    assert(eeAmplFitParamsFromPSet.size() == kNAmplitudeFitParams);
    ebAmplitudeFitParameters_.assign(ebAmplFitParamsFromPSet.begin(), ebAmplFitParamsFromPSet.end());
    eeAmplitudeFitParameters_.assign(eeAmplFitParamsFromPSet.begin(), eeAmplFitParamsFromPSet.end());
  }

  void EcalMultifitParametersHostESProducer::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
    edm::ParameterSetDescription desc;
    desc.add<std::vector<double>>("EBtimeFitParameters",
                                  {-2.015452e+00,
                                   3.130702e+00,
                                   -1.234730e+01,
                                   4.188921e+01,
                                   -8.283944e+01,
                                   9.101147e+01,
                                   -5.035761e+01,
                                   1.105621e+01});
    desc.add<std::vector<double>>("EEtimeFitParameters",
                                  {-2.390548e+00,
                                   3.553628e+00,
                                   -1.762341e+01,
                                   6.767538e+01,
                                   -1.332130e+02,
                                   1.407432e+02,
                                   -7.541106e+01,
                                   1.620277e+01});
    desc.add<std::vector<double>>("EBamplitudeFitParameters", {1.138, 1.652});
    desc.add<std::vector<double>>("EEamplitudeFitParameters", {1.890, 1.400});
    descriptions.addWithDefaultLabel(desc);
  }

  std::unique_ptr<EcalMultifitParametersHost> EcalMultifitParametersHostESProducer::produce(
      EcalMultifitParametersRcd const& iRecord) {
    size_t const sizeone = 1;
    auto product = std::make_unique<EcalMultifitParametersHost>(sizeone, cms::alpakatools::host());
    auto view = product->view();

    std::memcpy(view.timeFitParamsEB().data(), ebTimeFitParameters_.data(), sizeof(float) * kNTimeFitParams);
    std::memcpy(view.timeFitParamsEE().data(), eeTimeFitParameters_.data(), sizeof(float) * kNTimeFitParams);

    std::memcpy(
        view.amplitudeFitParamsEB().data(), ebAmplitudeFitParameters_.data(), sizeof(float) * kNAmplitudeFitParams);
    std::memcpy(
        view.amplitudeFitParamsEE().data(), eeAmplitudeFitParameters_.data(), sizeof(float) * kNAmplitudeFitParams);

    return product;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_EVENTSETUP_ALPAKA_MODULE(EcalMultifitParametersHostESProducer);
