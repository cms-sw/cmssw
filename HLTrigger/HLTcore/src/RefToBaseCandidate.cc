/** \class RefToBaseCandidate
 *
 *  
 *  See header file for documentation
 *
 *  $Date: 2006/08/14 15:26:42 $
 *  $Revision: 1.10 $
 *
 *  \author Martin Grunewald
 *
 */

#include "HLTrigger/HLTcore/interface/RefToBaseCandidate.h"

//
// function definition
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
     unsigned int key) {
    return refprod->getParticleRef(key);
  };

}
