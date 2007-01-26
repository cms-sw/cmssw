#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/Egamma/interface/HLTEgammaEtFilter.h"
#include "HLTrigger/Egamma/interface/HLTEgammaDoubleEtFilter.h"
#include "HLTrigger/Egamma/interface/HLTEgammaEcalIsolFilter.h"
#include "HLTrigger/Egamma/interface/HLTEgammaHcalIsolFilter.h"
#include "HLTrigger/Egamma/interface/HLTPhotonTrackIsolFilter.h"
#include "HLTrigger/Egamma/interface/HLTElectronTrackIsolFilter.h"
#include "HLTrigger/Egamma/interface/HLTElectronPixelMatchFilter.h"
#include "HLTrigger/Egamma/interface/HLTElectronEoverpFilter.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(HLTEgammaEtFilter);
DEFINE_ANOTHER_FWK_MODULE(HLTEgammaDoubleEtFilter);
DEFINE_ANOTHER_FWK_MODULE(HLTEgammaEcalIsolFilter);
DEFINE_ANOTHER_FWK_MODULE(HLTEgammaHcalIsolFilter);
DEFINE_ANOTHER_FWK_MODULE(HLTPhotonTrackIsolFilter);
DEFINE_ANOTHER_FWK_MODULE(HLTElectronTrackIsolFilter);
DEFINE_ANOTHER_FWK_MODULE(HLTElectronPixelMatchFilter);
DEFINE_ANOTHER_FWK_MODULE(HLTElectronEoverpFilter);
