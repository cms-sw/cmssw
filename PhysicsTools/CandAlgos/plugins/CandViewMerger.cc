/* \class CandViewMerger
 * 
 * Producer of merged Candidate collection 
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/Merger.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef Merger<reco::CandidateView, reco::CandidateCollection> CandViewMerger;

DEFINE_FWK_MODULE( CandViewMerger );
