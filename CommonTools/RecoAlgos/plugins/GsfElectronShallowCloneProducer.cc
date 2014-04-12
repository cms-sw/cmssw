#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "CommonTools/CandAlgos/interface/ShallowCloneProducer.h"

typedef ShallowCloneProducer<reco::GsfElectronCollection> PixelMatchGsfElectronShallowCloneProducer;

#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_FWK_MODULE( PixelMatchGsfElectronShallowCloneProducer );
