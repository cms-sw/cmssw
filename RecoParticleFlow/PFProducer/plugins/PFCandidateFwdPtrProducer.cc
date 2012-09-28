#include "CommonTools/UtilAlgos/plugins/FwdPtrProducer.h"


#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"



typedef edm::FwdPtrProducer<reco::PFCandidate> PFCandidateFwdPtrProducer;


