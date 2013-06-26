#include "DPGAnalysis/Skims/interface/TriggerMatchProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
typedef TriggerMatchProducer< reco::GsfElectron > trgMatchGsfElectronProducer;
DEFINE_FWK_MODULE( trgMatchGsfElectronProducer );
