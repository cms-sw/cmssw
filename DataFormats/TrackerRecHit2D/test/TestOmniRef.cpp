#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include <iostream>
#include <cassert>

int main() {

  typedef edm::Ref<edmNew::DetSetVector<SiPixelCluster>,SiPixelCluster > ClusterPixelRef;
  typedef edm::Ref<edmNew::DetSetVector<SiStripCluster>,SiStripCluster > ClusterRef;
  typedef edm::Ref< edm::LazyGetter<SiStripCluster>, SiStripCluster, edm::FindValue<SiStripCluster> >  ClusterRegionalRef;

  ClusterPixelRef pixelRef;
  ClusterRef stripRef;
  ClusterRegionalRef regionalRef;

  OmniClusterRef invalid;
  OmniClusterRef opixel(pixelRef);
  OmniClusterRef ostrip(stripRef);
  OmniClusterRef oregional(regionalRef);

  assert(!invalid.isValid());
  assert(!invalid.isPixel());
  assert(!invalid.isStrip());
  assert(!invalid.isRegional());

  assert(!opixel.isValid());
  assert(opixel.isPixel());
  assert(!opixel.isStrip());
  assert(!opixel.isRegional());

  assert(!ostrip.isValid());
  assert(!ostrip.isPixel());
  assert(ostrip.isStrip());
  assert(!ostrip.isRegional());


  assert(!oregional.isValid());
  assert(!oregional.isPixel());
  assert(oregional.isStrip());
  assert(oregional.isRegional());

}
