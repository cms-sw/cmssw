#ifndef TrackerRecHit2D_OmniClusterRef_H
#define TrackerRecHit2D_OmniClusterRef_H


#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/RefGetter.h"

class OmniClusterRef {

public:
  typedef edm::Ref<edmNew::DetSetVector<SiPixelCluster>,SiPixelCluster > ClusterPixelRef;
  typedef edm::Ref<edmNew::DetSetVector<SiStripCluster>,SiStripCluster > ClusterRef;
  typedef edm::Ref< edm::LazyGetter<SiStripCluster>, SiStripCluster, edm::FindValue<SiStripCluster> >  ClusterRegionalRef;
  
  OmniClusterRef(): index_(0x80000000) {}
  explicit OmniClusterRef(ClusterPixelRef const & ref) { setRef(ref); }
  explicit OmniClusterRef(ClusterRef const & ref) { setRef(ref); }
  explicit OmniClusterRef(ClusterRegionalRef const & ref) { setRef(ref); }
  
  ClusterPixelRef cluster_pixel()  const { 
    return isPixel() ?  ClusterPixelRef(product_,index()) : ClusterPixelRef();
  }
  
  ClusterRegionalRef cluster_regional()  const { 
    return isRegional() ?  ClusterRegionalRef(product_,index()) : ClusterRegionalRef();
  }
  
  ClusterRef cluster()  const { return  cluster_strip();}

  ClusterRef cluster_strip()  const { 
    return (isPixel() || isRegional()) ? ClusterRef() : ClusterRef(product_,index());
  }
  
  bool operator==(OmniClusterRef const & lh) const { 
    return index_ == lh.index_ // in principle this is enough!
      && product_ == lh.product_;
  }
  
public:
  
  unsigned int index() const { return index_ & (~0x60000000);}
  
  void setRef(ClusterPixelRef const & ref) {
    product_ = ref.refCore();
    index_ = ref.key();
  }
  
  
  void setRef(ClusterRef const & ref) {
    product_ = ref.refCore();
    index_ = ref.key() | 0x20000000; // bit 29 on
  }
  
  void setRef(ClusterRegionalRef const & ref) {
    product_ = ref.refCore();
    index_ = ref.key() | 0x60000000;  // bit 30 and 29 on (bit 31 on = invalid...)
  }
  
  bool isValid() const { return !(index_ & 0x80000000); }
  bool isPixel() const { return !isStrip(); }
  bool isStrip() const { return index_ & 0x20000000; }
  bool isRegional() const { return index_ & 0x60000000; }
  
  
  edm::RefCore product_;
  unsigned int index_;
  
  
};

#endif // TrackerRecHit2D_OmniClusterRef_H

