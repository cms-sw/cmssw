#include "CommonTools/UtilAlgos/interface/FwdPtrProducer.h"
#include "CommonTools/ParticleFlow/interface/PFCandidateFwdPtrFactory.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"



typedef edm::FwdPtrProducer<reco::PFCandidate, reco::PFCandidateFwdPtrFactory> PFCandidateFwdPtrProducer;


