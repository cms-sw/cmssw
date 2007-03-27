#ifndef RecoTauTag_Pi0Tau_Tau3DFwd_h
#define RecoTauTag_Pi0Tau_Tau3DFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

namespace reco {
  class Tau3D;

  /// collection of Tau3D objects
  typedef std::vector<Tau3D> Tau3DCollection;

  /// persistent reference to Tau3D objects
  typedef edm::Ref<Tau3DCollection> Tau3DRef;

  /// reference to Tau3D collection
  typedef edm::RefProd<Tau3DCollection> Tau3DRefProd;

  /// vector of references to Tau3D objects all in the same collection
  typedef edm::RefVector<Tau3DCollection> Tau3DRefVector;

  /// iterator over a vector of references to Tau3D objects
  typedef Tau3DRefVector::iterator tau3D_iterator;
}

#endif
