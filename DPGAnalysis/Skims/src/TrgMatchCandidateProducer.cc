#include "DPGAnalysis/Skims/interface/TriggerMatchProducer.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DataFormats/Candidate/interface/Candidate.h"
typedef TriggerMatchProducer<reco::Candidate> trgMatchCandidateProducer;
DEFINE_FWK_MODULE( trgMatchCandidateProducer );
