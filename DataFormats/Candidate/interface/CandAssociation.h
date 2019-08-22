#ifndef Candidate_CandAssociation_h
#define Candidate_CandAssociation_h
//
// \author Luca Lista, INFN
//
#include "DataFormats/Common/interface/AssociationVector.h"
#include "DataFormats/Common/interface/ValueMap.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include <vector>

namespace edm {
  namespace helper {
    struct CandMasterKeyReference {
      template <typename CandRef>
      static const CandRef& get(const CandRef& t, edm::ProductID id) {
        if (id == t.id())
          return t;
        else
          return t->masterClone().template castTo<CandRef>();
      }
    };

    template <>
    struct AssociationKeyReferenceTrait<reco::CandidateCollection> {
      typedef CandMasterKeyReference type;
    };
  }  // namespace helper
}  // namespace edm

namespace reco {
  typedef edm::AssociationVector<CandidateRefProd, std::vector<float> > CandFloatAssociations;
  typedef edm::AssociationVector<CandidateRefProd, std::vector<double> > CandDoubleAssociations;
  typedef edm::AssociationVector<CandidateRefProd, std::vector<int> > CandIntAssociations;
  typedef edm::AssociationVector<CandidateRefProd, std::vector<unsigned int> > CandUIntAssociations;
  typedef edm::AssociationVector<CandidateBaseRefProd, std::vector<float> > CandViewFloatAssociations;
  typedef edm::AssociationVector<CandidateBaseRefProd, std::vector<double> > CandViewDoubleAssociations;
  typedef edm::AssociationVector<CandidateBaseRefProd, std::vector<int> > CandViewIntAssociations;
  typedef edm::AssociationVector<CandidateBaseRefProd, std::vector<unsigned int> > CandViewUIntAssociations;
  typedef edm::ValueMap<CandidateBaseRef> CandRefValueMap;
}  // namespace reco

#endif
