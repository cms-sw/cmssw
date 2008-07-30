#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "PhysicsTools/CandAlgos/interface/ShallowCloneProducer.h"

typedef ShallowCloneProducer<reco::GsfElectronCollection> GsfElectronShallowCloneProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( GsfElectronShallowCloneProducer );
