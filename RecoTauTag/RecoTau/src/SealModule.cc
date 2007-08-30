#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoTauTag/RecoTau/interface/PFRecoTauTagInfoProducer.h"
#include "RecoTauTag/RecoTau/interface/PFRecoTauProducer.h"
#include "RecoTauTag/RecoTau/interface/CaloRecoTauTagInfoProducer.h"
#include "RecoTauTag/RecoTau/interface/CaloRecoTauProducer.h"
#include "RecoTauTag/RecoTau/interface/DiscriminationByIsolation.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(PFRecoTauTagInfoProducer);
DEFINE_ANOTHER_FWK_MODULE(PFRecoTauProducer);
DEFINE_ANOTHER_FWK_MODULE(CaloRecoTauTagInfoProducer);
DEFINE_ANOTHER_FWK_MODULE(CaloRecoTauProducer);
DEFINE_ANOTHER_FWK_MODULE(DiscriminationByIsolation);
