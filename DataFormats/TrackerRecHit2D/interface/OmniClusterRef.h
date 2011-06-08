#ifndef TrackerRecHit2D_OmniClusterRef_H
#define TrackerRecHit2D_OmniClusterRef_H


#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/RefGetter.h"

class OmniClusterRef {

public:

  typedef edm::Ref<edmNew::DetSetVector<SiStripCluster>,SiStripCluster > ClusterRef;
  typedef edm::Ref< edm::LazyGetter<SiStripCluster>, SiStripCluster, edm::FindValue<SiStripCluster> >  ClusterRegionalRef;

  OmniClusterRef(): index_(0x80000000) {}
  explicit OmniClusterRef(ClusterRef const & ref) { setRef(ref); }
  explicit OmniClusterRef(ClusterRegionalRef const & ref) { setRef(ref); }


  ClusterRegionalRef cluster_regional()  const { 
    return isRegional() ?  ClusterRegionalRef(product_,index()) : ClusterRegionalRef();
  }
  
  ClusterRef cluster()  const { 
    return isRegional() ? ClusterRef() : ClusterRef(product_,index());
  }

  bool operator==(OmniClusterRef const & lh) const { 
    return index_ == lh.index_ // in principle this is enough!
      && product_ == lh.product_;
  }

public:

  unsigned int index() const { return index_ & (~0x40000000);}

  void setRef(ClusterRef const & ref) {
    product_ = ref.refCore();
    index_ = ref.key() & (~0x40000000);
  }

  void setRef(ClusterRegionalRef const & ref) {
    product_ = ref.refCore();
    index_ = ref.key() | 0x40000000;  // bit 30 on (bit 31 on = invalid...)
  }

  bool isRegional() const { return index_ & 0x40000000; }


  edm::RefCore product_;
  unsigned int index_;


};

#endif // TrackerRecHit2D_OmniClusterRef_H

