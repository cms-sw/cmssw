#ifndef TrackerRecHit2D_OmniClusterRef_H
#define TrackerRecHit2D_OmniClusterRef_H

#include "DataFormats/Common/interface/RefCoreWithIndex.h"

#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Common/interface/DetSetVectorNew.h"

class OmniClusterRef {

  static const unsigned int kInvalid = 0x80000000; // bit 31 on
  static const unsigned int kIsStrip = 0x20000000; // bit 29 on
  static const unsigned int kIsRegional = 0x60000000; // bit 30 and 29 on  (will become fastsim???)

public:
  typedef edm::Ref<edmNew::DetSetVector<SiPixelCluster>,SiPixelCluster > ClusterPixelRef;
  typedef edm::Ref<edmNew::DetSetVector<SiStripCluster>,SiStripCluster > ClusterStripRef;
  
  OmniClusterRef() : me(edm::RefCore(),kInvalid) {}
  explicit OmniClusterRef(ClusterPixelRef const & ref) : me(ref.refCore(), (ref.isNonnull() ? ref.key() : kInvalid) ){}
  explicit OmniClusterRef(ClusterStripRef const & ref) : me(ref.refCore(), (ref.isNonnull() ? ref.key() | kIsStrip : kInvalid) ){}
  
  ClusterPixelRef cluster_pixel()  const { 
    return (isPixel() && isValid()) ?  ClusterPixelRef(me.toRefCore(),index()) : ClusterPixelRef();
  }

  ClusterStripRef cluster_strip()  const { 
    return isStrip() ? ClusterStripRef(me.toRefCore(),index()) : ClusterStripRef();
  }
  
  SiPixelCluster const & pixelCluster() const {
    return *ClusterPixelRef(me.toRefCore(),index());
  }
  SiStripCluster const & stripCluster() const {
    return *ClusterStripRef(me.toRefCore(),index());
  }
  
  bool operator==(OmniClusterRef const & lh) const { 
    return rawIndex() == lh.rawIndex(); // in principle this is enough!
  }

  bool operator<(OmniClusterRef const & lh) const { 
    return rawIndex() < lh.rawIndex(); // in principle this is enough!
  }
  
public:
  // edm Ref interface
  /* auto */ edm::ProductID id() const { return me.id();}
  unsigned int key() const { return index();}


  unsigned int rawIndex() const { return me.index();}
  
  unsigned int index() const { return rawIndex() & (~kIsRegional);}
  

  bool isValid() const { return !(rawIndex() & kInvalid); }
  bool isPixel() const { return !isStrip(); } //NOTE: non-valid will also show up as a pixel
  bool isStrip() const { return rawIndex() & kIsStrip; }
  // bool isRegional() const { return (rawIndex() & kIsRegional)==kIsRegional; }
  // bool isNonRegionalStrip() const {return (rawIndex() & kIsRegional)==kIsStrip;}

  edm::RefCoreWithIndex me;
 
  
};

#endif // TrackerRecHit2D_OmniClusterRef_H

