#ifndef CondFormats_EcalObjects_interface_EcalMultifitConditionsHost_h
#define CondFormats_EcalObjects_interface_EcalMultifitConditionsHost_h

#include "CondFormats/EcalObjects/interface/EcalMultifitConditionsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

using EcalMultifitConditionsHost = PortableHostCollection<EcalMultifitConditionsSoA>;

#endif
