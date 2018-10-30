#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "PhysicsTools/SelectorUtils/interface/VersionedIdProducer.h"

typedef VersionedIdProducer<reco::GsfElectronPtr> VersionedGsfElectronIdProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(VersionedGsfElectronIdProducer);
