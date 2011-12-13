#ifndef TrackerRecHit2D_OmniClusterRef_H
#define TrackerRecHit2D_OmniClusterRef_H

#include "DataFormats/Common/interface/RefCoreWithIndex.h"

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
  
  OmniClusterRef() : me(edm::RefCore(),0x80000000) {}
  explicit OmniClusterRef(ClusterPixelRef const & ref) : me(ref.refCore(),ref.key()){}
  explicit OmniClusterRef(ClusterRef const & ref) : me(ref.refCore(),ref.key() | 0x20000000){}  // bit 29 on
  explicit OmniClusterRef(ClusterRegionalRef const & ref) : me(ref.refCore(),ref.key() | 0x60000000){} // bit 30 and 29 on (bit 31 on = invalid...)
  
  ClusterPixelRef cluster_pixel()  const { 
    return isPixel() ?  ClusterPixelRef(me.toRefCore(),index()) : ClusterPixelRef();
  }
  
  ClusterRegionalRef cluster_regional()  const { 
    return isRegional() ?  ClusterRegionalRef(me.toRefCore(),index()) : ClusterRegionalRef();
  }
  
  ClusterRef cluster()  const { return  cluster_strip();}

  ClusterRef cluster_strip()  const { 
    return (isPixel() || isRegional()) ? ClusterRef() : ClusterRef(me.toRefCore(),index());
  }
  
  bool operator==(OmniClusterRef const & lh) const { 
    return rawIndex() == lh.rawIndex(); // in principle this is enough!
  }
  
public:

  unsigned int rawIndex() const { return me.index();}
  
  unsigned int index() const { return rawIndex() & (~0x60000000);}
  

  bool isValid() const { return !(rawIndex() & 0x80000000); }
  bool isPixel() const { return !isStrip(); }
  bool isStrip() const { return rawIndex() & 0x20000000; }
  bool isRegional() const { return (rawIndex() & 0x60000000)==0x60000000; }
  
  edm::RefCoreWithIndex me;
 
  
};

#endif // TrackerRecHit2D_OmniClusterRef_H

