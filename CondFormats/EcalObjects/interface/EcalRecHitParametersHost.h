#ifndef CondFormats_EcalObjects_interface_EcalRecHitParametersHost_h
#define CondFormats_EcalObjects_interface_EcalRecHitParametersHost_h

#include "CondFormats/EcalObjects/interface/EcalRecHitParametersSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

using EcalRecHitParametersHost = PortableHostCollection<EcalRecHitParametersSoA>;

#endif
