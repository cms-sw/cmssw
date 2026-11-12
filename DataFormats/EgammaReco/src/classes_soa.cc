#include <Eigen/Core>
#include "DataFormats/Common/interface/Wrapper.h"
#include "DataFormats/EgammaReco/interface/SuperClusterHostCollection.h"
#include "DataFormats/EgammaReco/interface/SuperClusterSoA.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedHostCollection.h"
#include "DataFormats/EgammaReco/interface/ElectronSeedSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"

SET_PORTABLEHOSTCOLLECTION_READ_RULES(reco::SuperClusterHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(reco::ElectronSeedHostCollection);
