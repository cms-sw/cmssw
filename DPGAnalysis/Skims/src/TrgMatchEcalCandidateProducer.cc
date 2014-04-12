#include "DPGAnalysis/Skims/interface/TriggerMatchProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
typedef TriggerMatchProducer<reco::RecoEcalCandidate> trgMatchEcalCandidateProducer;
DEFINE_FWK_MODULE( trgMatchEcalCandidateProducer );
