#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "PhysicsTools/CandAlgos/interface/CloneProducer.h"

typedef CloneProducer<reco::GsfElectronCollection> PixelMatchGsfElectronCloneProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( PixelMatchGsfElectronCloneProducer );
