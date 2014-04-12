#ifndef EgammaReco_BasicClusterFwd_h
#define EgammaReco_BasicClusterFwd_h
#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"
// #include "DataFormats/Common/interface/ExtCollection.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "DataFormats/EgammaReco/interface/BasicCluster.h"

namespace reco {
  //class BasicCluster;
  /*
  struct BasicClusterRefProds {
    BasicClusterRefProds() { }
    edm::RefProd<EcalRecHitCollection> recHits() const { return recHits_; }
    void setRecHits( edm::RefProd<EcalRecHitCollection> ref ) { recHits_ = ref; }
  private:
    edm::RefProd<EcalRecHitCollection> recHits_;
  };

  typedef edm::ExtCollection<std::vector<BasicCluster>, BasicClusterRefProds> BasicClusterCollection;
  */

  /// collection of BasicCluster objects
  typedef std::vector<BasicCluster> BasicClusterCollection;

  /// persistent reference to BasicCluster objects
  typedef edm::Ref<BasicClusterCollection> BasicClusterRef;

  /// reference to BasicCluster collection
  typedef edm::RefProd<BasicClusterCollection> BasicClusterRefProd;

  /// vector of references to BasicCluster objects all in the same collection
  typedef edm::RefVector<BasicClusterCollection> BasicClusterRefVector;

  /// iterator over a vector of references to BasicCluster objects
  typedef BasicClusterRefVector::iterator basicCluster_iterator;
}

#endif
