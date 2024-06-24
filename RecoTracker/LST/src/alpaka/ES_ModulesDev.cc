#include "RecoTracker/LSTCore/interface/alpaka/LST.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/typelookup.h"

// Temporary hack: The DevHost instantiation is needed in both CPU and GPU plugins,
// whereas the (non-host-)Device instantiation only in the GPU plugin
TYPELOOKUP_DATA_REG(SDL::LSTESHostData<SDL::Dev>);
TYPELOOKUP_DATA_REG(SDL::LSTESDeviceData<SDL::DevHost>);
TYPELOOKUP_DATA_REG(ALPAKA_ACCELERATOR_NAMESPACE::ESDeviceProduct<std::unique_ptr<SDL::LSTESHostData<SDL::Dev>>>);
TYPELOOKUP_ALPAKA_TEMPLATED_DATA_REG(SDL::LSTESHostData);
TYPELOOKUP_ALPAKA_TEMPLATED_DATA_REG(SDL::LSTESDeviceData);
