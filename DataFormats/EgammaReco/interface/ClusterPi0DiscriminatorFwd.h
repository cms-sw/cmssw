#ifndef EgammaReco_ClusterPi0DiscriminatorFwd_h
#define EgammaReco_ClusterPi0DiscriminatorFwd_h
// $Id: ClusterPi0DiscriminatorFwd.h,v 1.5 2006/03/20 14:06:37 llista Exp $
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/RefVector.h"

namespace reco {
  class ClusterPi0Discriminator;

  /// collection of ClusterPi0Discriminator objects
  typedef std::vector<ClusterPi0Discriminator> ClusterPi0DiscriminatorCollection;

  /// persistent reference to a ClusterPi0Discriminator object
  typedef edm::Ref<ClusterPi0DiscriminatorCollection> ClusterPi0DiscriminatorRef;

  /// reference to a ClusterPi0Discriminator collection
  typedef edm::RefProd<ClusterPi0DiscriminatorCollection> ClusterPi0DiscriminatorRefProd;

  /// vector of references to ClusterPi0Discriminator objects in the same collection
  typedef edm::RefVector<ClusterPi0DiscriminatorCollection> ClusterPi0DiscriminatorRefVector;

  /// iterator over a vector of references to ClusterPi0Discriminator objects
  typedef ClusterPi0DiscriminatorRefVector::iterator clusterPi0Discriminator_iterator;
}

#endif
