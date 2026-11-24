#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/VectorHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/OTRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsMaskingHost.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(reco::HitPortableCollectionHost);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(reco::VectorHitPortableCollectionHost);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(reco::OTRecHitPortableCollectionHost);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(reco::TrackingRecHitsMaskingHost);
