#ifndef CondFormats_EcalObjects_interface_EcalRecHitConditionsHost_h
#define CondFormats_EcalObjects_interface_EcalRecHitConditionsHost_h

#include "CondFormats/EcalObjects/interface/EcalRecHitConditionsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

using EcalRecHitConditionsHost = PortableHostCollection<EcalRecHitConditionsSoA>;

#endif
