#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "RecoEgamma/PhotonIdentification/plugins/PhotonIDProducer.h"
#include "RecoEgamma/PhotonIdentification/plugins/PhotonIDSimpleAnalyzer.h"

DEFINE_FWK_MODULE(PhotonIDProducer);
DEFINE_FWK_MODULE(PhotonIDSimpleAnalyzer);

