/* \class PFCandidateMerger
 * 
 * Merges two lists of PFCandidates
 *
 * \author: L. Gray (FNAL)
 *
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/Merger.h"

#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"

typedef Merger<reco::PFCandidateCollection> PFCandidateListMerger;

DEFINE_FWK_MODULE( PFCandidateListMerger );
