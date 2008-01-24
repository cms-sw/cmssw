#include "DataFormats/PatCandidates/interface/EventHypothesis.h"
#include <algorithm>

void pat::EventHypothesis::add(const reco::CandidateBaseRef &ref, int32_t role) {
    particles_.push_back(value_type(ref,role));
}

const reco::CandidateBaseRef & pat::EventHypothesis::operator[](int32_t role) const {
    static reco::CandidateBaseRef null_;
    const_iterator it = std::find_if(begin(), end(), ByRole(role));
    return (it == end() ? null_ : it->first);
}
