#include "DataFormats/PatCandidates/interface/Electron.h"
#include "PhysicsTools/SelectorUtils/interface/VersionedIdProducer.h"

typedef VersionedIdProducer<pat::ElectronPtr> VersionedPatElectronIdProducer;

//define this as a plug-in
DEFINE_FWK_MODULE(VersionedPatElectronIdProducer);
