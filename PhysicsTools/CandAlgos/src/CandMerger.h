#ifndef CandAlgos_CandMerger_h
#define CandAlgos_CandMerger_h
/** 
 * Framework module to merge an arbitrary number of candidate collections  
 * 
 * $Id$
 *
 */
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/UtilAlgos/interface/Merger.h"
#include "DataFormats/Common/interface/ClonePolicy.h"

namespace candmodules {
  /// merge an arbitrary number of candidate collections  
  typedef Merger<reco::CandidateCollection, edm::ClonePolicy<reco::Candidate> > CandMerger;
}

#endif
