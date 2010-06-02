#ifndef AnalysisDataFormats_TagAndProbe_CandidateAssociation_h
#define AnalysisDataFormats_TagAndProbe_CandidateAssociation_h

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Common/interface/AssociationMap.h"

namespace reco
{
   typedef edm::AssociationMap< edm::OneToManyWithQualityGeneric< CandidateView, CandidateView, bool > > CandViewCandViewAssociation;
}

#endif

