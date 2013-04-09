#include "DPGAnalysis/Skims/interface/TriggerMatchProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/RecoCandidate/interface/RecoChargedCandidate.h"
typedef TriggerMatchProducer<reco::RecoChargedCandidate> trgMatchChargedCandidateProducer;
DEFINE_FWK_MODULE( trgMatchChargedCandidateProducer );
