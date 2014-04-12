#include "DPGAnalysis/Skims/interface/TriggerMatchProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/EgammaCandidates/interface/Electron.h"
typedef TriggerMatchProducer<reco::Electron> trgMatchElectronProducer;
DEFINE_FWK_MODULE( trgMatchElectronProducer );
