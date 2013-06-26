#include "CommonTools/UtilAlgos/interface/FwdPtrCollectionFilter.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/PdgIdSelector.h"
#include "CommonTools/ParticleFlow/interface/PFCandidateWithSrcPtrFactory.h"

typedef edm::FwdPtrCollectionFilter< reco::PFCandidate, 
                                     reco::StringCutObjectSelectorHandler<reco::PFCandidate,false>, 
                                     reco::PFCandidateWithSrcPtrFactory >  PFCandidateFwdPtrCollectionStringFilter;
typedef edm::FwdPtrCollectionFilter< reco::PFCandidate, reco::PdgIdSelectorHandler, 
                                     reco::PFCandidateWithSrcPtrFactory >  PFCandidateFwdPtrCollectionPdgIdFilter;

