#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"
#include "PhysicsTools/CandAlgos/interface/CloneProducer.h"

typedef CloneProducer<reco::PixelMatchGsfElectronCollection> PixelMatchGsfElectronCloneProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( PixelMatchGsfElectronCloneProducer );
