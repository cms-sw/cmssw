#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_SEAL_MODULE();

#include <DQMOffline/Ecal/interface/EBClusterTaskExtras.h>
DEFINE_ANOTHER_FWK_MODULE(EBClusterTaskExtras);

#include <DQMOffline/Ecal/interface/EEClusterTaskExtras.h>
DEFINE_ANOTHER_FWK_MODULE(EEClusterTaskExtras);
