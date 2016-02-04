/* \class CandRefMerger
 * 
 * Producer of merged Candidate reference collection 
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/Merger.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef Merger<reco::CandidateBaseRefVector> CandRefMerger;

DEFINE_FWK_MODULE( CandRefMerger );
