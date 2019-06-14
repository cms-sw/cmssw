#ifndef Candidate_CandMatchMapMany_h
#define Candidate_CandMatchMapMany_h
/* \class reco::CandMatchMapMany
 * 
 * One-to-Many Candidate association map by reference
 *
 * \author Luca Lista, INFN
 */
#include "DataFormats/Common/interface/AssociationMap.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace reco {
  typedef edm::AssociationMap<edm::OneToManyWithQuality<reco::CandidateCollection, reco::CandidateCollection, double> >
      CandMatchMapMany;
}

#endif
