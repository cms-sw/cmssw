#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoTauTag/RecoTau/interface/PFRecoTauProducer.h"
#include "RecoTauTag/RecoTau/interface/CaloRecoTauProducer.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(PFRecoTauProducer);
DEFINE_ANOTHER_FWK_MODULE(CaloRecoTauProducer);
