#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/TestHandle.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "DataFormats/TrackerRecHit2D/interface/OmniClusterRef.h"
#include<cassert>




int strip() {


  typedef edm::Ref<edmNew::DetSetVector<SiPixelCluster>,SiPixelCluster > ClusterPixelRef;
  typedef edm::Ref<edmNew::DetSetVector<SiStripCluster>,SiStripCluster > ClusterStripRef;

  using Coll = edmNew::DetSetVector<SiStripCluster>;

  Coll coll;
  edm::TestHandle<Coll> collH(&coll, edm::ProductID(1, 1));

  typedef Coll::FastFiller FF;

  {
    FF ff(coll,1);
    ff.push_back(SiStripCluster());
    ff.push_back(SiStripCluster());
    ff.push_back(SiStripCluster());
    ff.push_back(SiStripCluster());
  }

  ClusterStripRef sref = edmNew::makeRefTo(collH,&coll.data()[2]);

  OmniClusterRef oref(sref);
  assert(oref.isValid());
  assert(oref.index()==2);
  assert(oref.isStrip());
  assert(!oref.isPixel());
  assert(oref.subCluster()==0);

  OmniClusterRef oref2(sref,3);
  OmniClusterRef oref3(sref,3);
  assert(oref2.isValid());
  assert(oref2.index()==2);
  assert(oref2.isStrip());
  assert(!oref2.isPixel());
  assert(oref2.subCluster()==3);
  assert(!(oref2==oref));
  assert((oref2==oref3));

 



  return 0;
}

int pixel() {


  typedef edm::Ref<edmNew::DetSetVector<SiPixelCluster>,SiPixelCluster > ClusterPixelRef;
  typedef edm::Ref<edmNew::DetSetVector<SiStripCluster>,SiStripCluster > ClusterStripRef;

  using Coll = edmNew::DetSetVector<SiPixelCluster>;

  Coll coll;
  edm::TestHandle<Coll> collH(&coll, edm::ProductID(1, 1));

  typedef Coll::FastFiller FF;

  {
    FF ff(coll,1);
    ff.push_back(SiPixelCluster());
    ff.push_back(SiPixelCluster());
    ff.push_back(SiPixelCluster());
    ff.push_back(SiPixelCluster());
  }

  ClusterPixelRef sref = edmNew::makeRefTo(collH,&coll.data()[2]);

  OmniClusterRef oref(sref);
  assert(oref.isValid());
  assert(oref.index()==2);
  assert(!oref.isStrip());
  assert(oref.isPixel());
  assert(oref.subCluster()==0);

  OmniClusterRef oref2(sref,3);
  OmniClusterRef oref3(sref,3);
  assert(oref2.isValid());
  assert(oref2.index()==2);
  assert(!oref2.isStrip());
  assert(oref2.isPixel());
  assert(oref2.subCluster()==3);
  assert(!(oref2==oref));
  assert((oref2==oref3));

  return 0;
 
} 

int main() {
  
  return strip()+pixel();
}

