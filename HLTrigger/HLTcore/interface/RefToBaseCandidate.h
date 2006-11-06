#ifndef RefToBaseCandidate_h
#define RefToBaseCandidate_h

/** \class RefToBaseCandidate
 *
 *  
 *  This free-standing function constructs a RefToBase<Candidate>,
 *  based on a RefProd<C> and a key. The collection C can be either a
 *  ConcreteCollection of Candidates or an HLTFilterObjectWithRefs.
 *
 *  $Date: 2006/10/26 17:05:49 $
 *  $Revision: 1.1 $
 *
 *  \author Martin Grunewald
 *
 */

#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefToBase.h"
#include "DataFormats/HLTReco/interface/HLTFilterObject.h"

//
// function declaration
//

namespace edm {

  template <typename C>
    edm::RefToBase<reco::Candidate> RefToBaseCandidate
    (const edm::RefProd<C>                             & refprod,
     unsigned int key) {
    return edm::RefToBase<reco::Candidate>(Ref<C>(refprod,key));
  };

  edm::RefToBase<reco::Candidate> RefToBaseCandidate
    (const edm::RefProd<reco::HLTFilterObjectWithRefs> & refprod,
     unsigned int key);

}

#endif //RefToBaseCandidate_h
