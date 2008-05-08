#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoTauTag/RecoTau/interface/PFRecoTauTagInfoProducer.h"
#include "RecoTauTag/RecoTau/interface/PFRecoTauProducer.h"
#include "RecoTauTag/RecoTau/interface/PFRecoTauDiscriminationByIsolation.h"
#include "RecoTauTag/RecoTau/interface/PFRecoTauDiscriminationAgainstElectron.h"
#include "RecoTauTag/RecoTau/interface/PFRecoTauDiscriminationAgainstMuon.h"
#include "RecoTauTag/RecoTau/interface/CaloRecoTauTagInfoProducer.h"
#include "RecoTauTag/RecoTau/interface/CaloRecoTauProducer.h"
#include "RecoTauTag/RecoTau/interface/CaloRecoTauDiscriminationByIsolation.h"
#include "RecoTauTag/RecoTau/interface/CaloRecoTauDiscriminationAgainstElectron.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(PFRecoTauTagInfoProducer);
DEFINE_ANOTHER_FWK_MODULE(PFRecoTauProducer);
DEFINE_ANOTHER_FWK_MODULE(PFRecoTauDiscriminationByIsolation);
DEFINE_ANOTHER_FWK_MODULE(PFRecoTauDiscriminationAgainstElectron);
DEFINE_ANOTHER_FWK_MODULE(PFRecoTauDiscriminationAgainstMuon);
DEFINE_ANOTHER_FWK_MODULE(CaloRecoTauTagInfoProducer);
DEFINE_ANOTHER_FWK_MODULE(CaloRecoTauProducer);
DEFINE_ANOTHER_FWK_MODULE(CaloRecoTauDiscriminationByIsolation);
DEFINE_ANOTHER_FWK_MODULE(CaloRecoTauDiscriminationAgainstElectron);
