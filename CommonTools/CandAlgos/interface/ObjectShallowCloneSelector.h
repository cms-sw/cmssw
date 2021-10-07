#ifndef CandAlgos_ObjectShallowCloneSelector_h
#define CandAlgos_ObjectShallowCloneSelector_h
/* \class RefVectorShallowCloneStoreMananger
 *
 * \author Luca Lista, INFN
 *
 */
#include "CommonTools/UtilAlgos/interface/ObjectSelectorLegacy.h"
#include "DataFormats/Candidate/interface/ShallowCloneCandidate.h"
#include "DataFormats/Common/interface/RefVector.h"

template <typename Selector,
          typename SizeSelector = NonNullNumberSelector,
          typename PostProcessor = helper::NullPostProcessor<reco::CandidateCollection> >
class ObjectShallowCloneSelector : public ObjectSelectorLegacy<Selector, reco::CandidateCollection, SizeSelector> {
public:
  explicit ObjectShallowCloneSelector(const edm::ParameterSet& cfg)
      : ObjectSelectorLegacy<Selector, reco::CandidateCollection, SizeSelector, PostProcessor>(cfg) {}
};

#endif
