#include "DPGAnalysis/Skims/interface/TriggerMatchProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/METReco/interface/MET.h"
typedef TriggerMatchProducer< reco::MET > trgMatchMETProducer;
DEFINE_FWK_MODULE( trgMatchMETProducer );
