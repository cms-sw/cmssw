/* \class CandMerger
 * 
 * Producer of merged Candidate collection 
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/Merger.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef Merger<reco::CandidateCollection> CandMerger;

DEFINE_FWK_MODULE( CandMerger );
