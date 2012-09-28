#include "CommonTools/UtilAlgos/plugins/FwdPtrCollectionFilter.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/PdgIdSelector.h"


typedef edm::FwdPtrCollectionFilter< reco::PFCandidate, reco::StringCutObjectSelectorHandler<reco::PFCandidate,false> >  PFCandidateFwdPtrCollectionStringFilter;
typedef edm::FwdPtrCollectionFilter< reco::PFCandidate, reco::PdgIdSelectorHandler >  PFCandidateFwdPtrCollectionPdgIdFilter;

