#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include <iostream>
#include <cassert>

int main() {

  edm::ProductID pid;

  typedef edm::Ref<edmNew::DetSetVector<SiPixelCluster>,SiPixelCluster > ClusterPixelRef;
  typedef edm::Ref<edmNew::DetSetVector<SiStripCluster>,SiStripCluster > ClusterRef;
  typedef edm::Ref< edm::LazyGetter<SiStripCluster>, SiStripCluster, edm::FindValue<SiStripCluster> >  ClusterRegionalRef;

  ClusterPixelRef pixelRef(pid,nullptr, 2,nullptr);
  ClusterRef stripRef(pid,nullptr, 2,nullptr);
  ClusterRegionalRef regionalRef(pid,nullptr, 2,nullptr);

  OmniClusterRef invalid;
  OmniClusterRef opixel(pixelRef);
  OmniClusterRef ostrip(stripRef);
  OmniClusterRef oregional(regionalRef);

  assert(invalid.index()> (1<<16));
  assert(opixel.index()==2);
  assert(ostrip.index()==2);
  assert(oregional.index()==2);

  assert(!invalid.isValid());
  assert(invalid.isPixel());  // empty looks as pixel!
  assert(!invalid.isStrip());
  assert(!invalid.isRegional());

  assert(opixel.isValid());
  assert(opixel.isPixel());
  assert(!opixel.isStrip());
  assert(!opixel.isRegional());

  assert(ostrip.isValid());
  assert(!ostrip.isPixel());
  assert(ostrip.isStrip());
  assert(!ostrip.isRegional());


  assert(oregional.isValid());
  assert(!oregional.isPixel());
  assert(oregional.isStrip());
  assert(oregional.isRegional());

}
