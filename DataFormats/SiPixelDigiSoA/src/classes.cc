#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigiErrorsSoA.h"
#include "DataFormats/SiPixelDigiSoA/interface/SiPixelDigisSoA.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<SiPixelDigiErrorsSoA>);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(PortableHostCollection<SiPixelDigisSoA>);
