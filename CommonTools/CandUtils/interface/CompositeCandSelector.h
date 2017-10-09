#ifndef CommonTools_CandUtils_CompositeCandSelector_h
#define CommonTools_CandUtils_CompositeCandSelector_h
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/EDMException.h"

template<typename Selector, typename T1 = reco::Candidate, typename T2 = T1, unsigned int nDau = 2>
class CompositeCandSelector {
public:
  explicit CompositeCandSelector(const Selector& select) : select_(select) { }
  bool operator()(const reco::Candidate & cmp) const {
    if(cmp.numberOfDaughters() != nDau)
      throw edm::Exception(edm::errors::InvalidReference)
        << "candidate has " << cmp.numberOfDaughters() 
        << ", while CompositeCandSelector "
        << "requires " << nDau << " daughters.\n";
    const T1 * dau1 = dynamic_cast<const T1 *>(cmp.daughter(0));
    if(dau1 == 0)  
      throw edm::Exception(edm::errors::InvalidReference)
        << "candidate's first daughter is not of the type required "
        << "by CompositeCandSelector.\n";
    const T2 * dau2 = dynamic_cast<const T2 *>(cmp.daughter(1));
    if(dau2 == 0)  
      throw edm::Exception(edm::errors::InvalidReference)
        << "candidate's second daughter is not of the type required "
        << "by CompositeCandSelector.\n";
    return select_(*dau1, *dau2);
  }
private:
  Selector select_;
};

// specializations for nDau = 3, 4, ... could go here if needed

#endif
