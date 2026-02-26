// Include the Eigen core library before including the SoA definitions
#include <Eigen/Core>

#include "DataFormats/ParticleFlowReco/interface/CaloRecHitHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFClusterHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitFractionHostCollection.h"
#include "DataFormats/ParticleFlowReco/interface/PFRecHitHostCollection.h"
#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(reco::CaloRecHitHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(reco::PFClusterHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(reco::PFRecHitFractionHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(reco::PFRecHitHostCollection);
