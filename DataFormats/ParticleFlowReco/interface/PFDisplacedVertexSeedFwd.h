#ifndef ParticleFlowReco_PFDisplacedVertexSeedFwd_h
#define ParticleFlowReco_PFDisplacedVertexSeedFwd_h
#include <vector>


#include "DataFormats/Common/interface/Ref.h"
/* #include "DataFormats/Common/interface/RefVector.h" */
/* #include "DataFormats/Common/interface/RefProd.h" */

namespace reco {
  class PFDisplacedVertexSeed;

  /// collection of PFDisplacedVertexSeed objects
  typedef std::vector<PFDisplacedVertexSeed> PFDisplacedVertexSeedCollection;  
  
  
  /// persistent reference to a PFDisplacedVertexSeed objects
  typedef edm::Ref<PFDisplacedVertexSeedCollection> PFDisplacedVertexSeedRef;

  /// handle to a PFDisplacedVertexSeed collection
  typedef edm::Handle<PFDisplacedVertexSeedCollection> PFDisplacedVertexSeedHandle;


  /// iterator over a vector of references to PFDisplacedVertexSeed objects
  /*   typedef PFDisplacedVertexSeedRefVector::iterator PFDisplacedVertexSeed_iterator; */
}

#endif
