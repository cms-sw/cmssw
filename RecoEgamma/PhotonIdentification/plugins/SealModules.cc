#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"
DEFINE_SEAL_MODULE();
#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/EventSetupInitTrait.h"

#include "RecoEgamma/PhotonIdentification/plugins/PhotonIDProducer.h"
#include "RecoEgamma/PhotonIdentification/plugins/PhotonIDSimpleAnalyzer.h"


DEFINE_ANOTHER_FWK_MODULE(PhotonIDProducer);
DEFINE_ANOTHER_FWK_MODULE(PhotonIDSimpleAnalyzer);

