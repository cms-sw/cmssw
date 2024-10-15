#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "EcalRecHitBuilder.h"
#include "EnergyComputationKernels.h"

//#define DEBUG
//#define ECAL_RECO_ALPAKA_DEBUG

namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::rechit {

  using namespace cms::alpakatools;

  // host version, to be called by the plugin
  void create_ecal_rechit(Queue& queue,
                          InputProduct const& ebUncalibRecHits,
                          InputProduct const& eeUncalibRecHits,
                          OutputProduct& ebRecHits,
                          OutputProduct& eeRecHits,
                          EcalRecHitConditionsDevice const& conditionsDev,
                          EcalRecHitParametersDevice const& parametersDev,
                          edm::TimeValue_t const& eventTime,
                          ConfigurationParameters const& configParams) {
    auto const nchannels = static_cast<uint32_t>(ebUncalibRecHits.const_view().metadata().size()) +
                           static_cast<uint32_t>(eeUncalibRecHits.const_view().metadata().size());

    //
    // kernel create rechit
    //
    uint32_t constexpr nchannels_per_block = 16;
    auto constexpr threads = nchannels_per_block;
    auto const blocks = cms::alpakatools::divide_up_by(nchannels, threads);
    auto workDiv = cms::alpakatools::make_workdiv<Acc1D>(blocks, threads);
    alpaka::exec<Acc1D>(queue,
                        workDiv,
                        KernelCreateEcalRechit{},
                        ebUncalibRecHits.const_view(),
                        eeUncalibRecHits.const_view(),
                        ebRecHits.view(),
                        eeRecHits.view(),
                        conditionsDev.const_view(),
                        parametersDev.const_view(),
                        eventTime,
                        // configuration
                        configParams.killDeadChannels,
                        configParams.recoverEBIsolatedChannels,
                        configParams.recoverEEIsolatedChannels,
                        configParams.recoverEBVFE,
                        configParams.recoverEEVFE,
                        configParams.recoverEBFE,
                        configParams.recoverEEFE,
                        configParams.EBLaserMIN,
                        configParams.EELaserMIN,
                        configParams.EBLaserMAX,
                        configParams.EELaserMAX,
                        configParams.flagmask);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::ecal::rechit
