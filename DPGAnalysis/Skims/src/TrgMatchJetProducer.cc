#include "DPGAnalysis/Skims/interface/TriggerMatchProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/JetReco/interface/Jet.h"
typedef TriggerMatchProducer< reco::Jet > trgMatchJetProducer;
DEFINE_FWK_MODULE( trgMatchJetProducer );
