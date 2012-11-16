#include "CommonTools/UtilAlgos/interface/ProductFromFwdPtrProducer.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "CommonTools/ParticleFlow/interface/PFCandidateWithSrcPtrFactory.h"

typedef edm::ProductFromFwdPtrProducer< reco::PFCandidate, 
                                        reco::PFCandidateWithSrcPtrFactory >  PFCandidateFromFwdPtrProducer;

