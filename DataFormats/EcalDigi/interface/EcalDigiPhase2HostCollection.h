#ifndef DataFormats_EcalDigi_EcalDigiPhase2HostCollection_h
#define DataFormats_EcalDigi_EcalDigiPhase2HostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/EcalDigi/interface/EcalDigiPhase2SoA.h"

// EcalDigiPhase2SoA in host memory
using EcalDigiPhase2HostCollection = PortableHostCollection<EcalDigiPhase2SoA>;

#endif
