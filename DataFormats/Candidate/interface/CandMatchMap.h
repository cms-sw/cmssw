#ifndef Candidate_CandMatchMap_h
#define Candidate_CandMatchMap_h
/* \class reco::CandMatchMap
 * 
 * One-to-one Candidate association map by reference
 *
 * \author Luca Lista, INFN
 */
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include "DataFormats/Common/interface/OneToOneGeneric.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace reco {
  typedef edm::AssociationMap<
            edm::OneToOne<reco::CandidateCollection, 
                          reco::CandidateCollection
            > 
          > CandMatchMap;

  typedef edm::AssociationMap<
            edm::OneToOneGeneric<reco::CandidateView, 
                                 reco::CandidateView
            > 
          > CandViewMatchMap;
}

#endif
