#include "CommonTools/UtilAlgos/interface/ProductFromFwdPtrProducer.h"
#include "CommonTools/ParticleFlow/interface/PFCandidateWithSrcPtrFactory.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"



typedef edm::ProductFromFwdPtrProducer<reco::PFCandidate, reco::PFCandidateWithSrcPtrFactory> PFCandidateProductFromFwdPtrProducer;


