#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTEcalIsolationProducers.h"
#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTHcalIsolationProducers.h"
#include "RecoEgamma/EgammaHLTProducers/interface/EgammaHLTTrackIsolationProducers.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(EgammaHLTEcalIsolationProducers)
DEFINE_ANOTHER_FWK_MODULE(EgammaHLTHcalIsolationProducers)
DEFINE_ANOTHER_FWK_MODULE(EgammaHLTTrackIsolationProducers)
