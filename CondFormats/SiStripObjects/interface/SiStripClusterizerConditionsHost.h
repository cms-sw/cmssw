#ifndef CondFormats_SiStripObjects_interface_SiStripClusterizerConditionsHost_h
#define CondFormats_SiStripObjects_interface_SiStripClusterizerConditionsHost_h

#include "CondFormats/SiStripObjects/interface/SiStripClusterizerConditionsSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

using SiStripClusterizerConditionsHost = PortableHostMultiCollection<SiStripClusterizerConditionsDetToFedsSoA,
                                                                     SiStripClusterizerConditionsData_fedchSoA,
                                                                     SiStripClusterizerConditionsData_stripSoA,
                                                                     SiStripClusterizerConditionsData_apvSoA>;
using SiStripClusterizerConditionsDetToFedsHost = PortableHostCollection<SiStripClusterizerConditionsDetToFedsSoA>;

#endif  // CondFormats_SiStripObjects_interface_SiStripClusterizerConditionsHost_h
