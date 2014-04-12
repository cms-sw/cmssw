#include "DPGAnalysis/Skims/interface/TriggerMatchProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/MuonReco/interface/Muon.h"
typedef TriggerMatchProducer< reco::Muon > trgMatchMuonProducer;
DEFINE_FWK_MODULE( trgMatchMuonProducer );
