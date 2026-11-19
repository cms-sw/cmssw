#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/Portable/interface/PortableHostObjectReadRules.h"
#include "DataFormats/HGCalReco/interface/MtdHostCollection.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsHostCollection.h"
#include "DataFormats/HGCalReco/interface/HGCalSoARecHitsExtraHostCollection.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(MtdHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(HGCalSoARecHitsHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(HGCalSoARecHitsExtraHostCollection);
