#include "CommonTools/UtilAlgos/plugins/ProductFromFwdPtrProducer.h"


#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Framework/interface/EventSetup.h"



typedef edm::ProductFromFwdPtrProducer<reco::PFCandidate> PFCandidateProductFromFwdPtrProducer;


