#include "DataFormats/FTLRecHitSoA/interface/BTLBaseRecHitHostCollection.h"
#include "DataFormats/FTLRecHitSoA/interface/BTLRecHitHostCollection.h"
#include "DataFormats/FTLRecHitSoA/interface/ETLBaseRecHitHostCollection.h"
#include "DataFormats/FTLRecHitSoA/interface/ETLRecHitHostCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(btlrechit::BTLBaseRecHitHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(btlrechit::BTLRecHitHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(etlrechit::ETLBaseRecHitHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(etlrechit::ETLRecHitHostCollection);
