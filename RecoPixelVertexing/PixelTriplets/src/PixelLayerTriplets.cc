#include "RecoPixelVertexing/PixelTriplets/interface/PixelLayerTriplets.h"

vector<PixelLayerTriplets::LayerPairAndLayers> PixelLayerTriplets::layers()
{
  
  typedef vector<const LayerWithHits* > ThirdLayer;
  vector<LayerPairAndLayers> result;

  {
    ThirdLayer thirds;
    LayerPair base(lh1,lh2);
    thirds.push_back(lh3);
    thirds.push_back(pos1);
    thirds.push_back(neg1);
    result.push_back( LayerPairAndLayers(base,thirds));
  }

  {
    ThirdLayer thirds;
    LayerPair base(lh1,pos1);
    thirds.push_back(pos2); 
    result.push_back( LayerPairAndLayers(base,thirds));
  }

  {
    ThirdLayer thirds;
    LayerPair base(lh1,neg1);
    thirds.push_back(neg2); 
    result.push_back( LayerPairAndLayers(base,thirds));
  }



  return result;
}

void PixelLayerTriplets::init(const SiPixelRecHitCollection &coll,const edm::EventSetup& iSetup)
{
  edm::ESHandle<GeometricSearchTracker> track;
  iSetup.get<TrackerRecoGeometryRecord>().get( track );

  map_range1=coll.get(acc.pixelBarrelLayer(1));
  map_range2=coll.get(acc.pixelBarrelLayer(2));
  map_range3=coll.get(acc.pixelBarrelLayer(3));

  //MP check the side
  map_diskneg1=coll.get(acc.pixelForwardDisk(1,1));
  map_diskneg2=coll.get(acc.pixelForwardDisk(1,2));
  map_diskpos1=coll.get(acc.pixelForwardDisk(2,1));
  map_diskpos2=coll.get(acc.pixelForwardDisk(2,2));

//  vector<BarrelDetLayer*> bl;
//  vector<ForwardDetLayer*> fpos;
//  vector<ForwardDetLayer*> fneg;

  bl=track->barrelLayers(); 
  fpos=track->posForwardLayers();
  fneg=track->negForwardLayers();

  //BarrelLayers
  const PixelBarrelLayer*  bl1=dynamic_cast<PixelBarrelLayer*>(bl[0]);
  const PixelBarrelLayer*  bl2=dynamic_cast<PixelBarrelLayer*>(bl[1]);
  const PixelBarrelLayer*  bl3=dynamic_cast<PixelBarrelLayer*>(bl[2]);
  //ForwardLayers
  const PixelForwardLayer*  fpos1=dynamic_cast<PixelForwardLayer*>(fpos[0]);
  const PixelForwardLayer*  fpos2=dynamic_cast<PixelForwardLayer*>(fpos[1]);
  const PixelForwardLayer*  fneg1=dynamic_cast<PixelForwardLayer*>(fneg[0]);
  const PixelForwardLayer*  fneg2=dynamic_cast<PixelForwardLayer*>(fneg[1]);

  //LayersWithHits

  lh1=new  LayerWithHits(bl1,map_range1);
  lh2=new  LayerWithHits(bl2,map_range2);
  lh3=new  LayerWithHits(bl3,map_range3);

  pos1=new  LayerWithHits(fpos1,map_diskpos1);
  pos2=new  LayerWithHits(fpos2,map_diskpos2);
  neg1=new  LayerWithHits(fneg1,map_diskneg1);
  neg2=new  LayerWithHits(fneg2,map_diskneg2);


}
