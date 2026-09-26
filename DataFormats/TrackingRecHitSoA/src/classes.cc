#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/StubsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(reco::HitPortableCollectionHost);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(reco::StubPortableCollectionHost);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(reco::OTRecHitPortableCollectionHost);
