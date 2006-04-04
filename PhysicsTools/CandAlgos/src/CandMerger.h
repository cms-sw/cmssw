#ifndef CandAlgos_CandMerger_h
#define CandAlgos_CandMerger_h
/** 
 * Framework module to merge an arbitrary number of candidate collections  
 * 
 * $Id: CandMerger.h,v 1.1 2006/03/03 13:11:12 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/UtilAlgos/interface/Merger.h"
#include "DataFormats/Common/interface/ClonePolicy.h"

namespace cand {
  namespace modules {
    /// merge an arbitrary number of candidate collections  
    typedef Merger<reco::CandidateCollection, edm::ClonePolicy<reco::Candidate> > CandMerger;
  }
}

#endif
