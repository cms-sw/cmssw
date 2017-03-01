#include "RecoPixelVertexing/PixelTriplets/interface/CosmicLayerTriplets.h"
#include "RecoTracker/TkHitPairs/interface/SeedLayerPairs.h"
#include "RecoTracker/TkDetLayers/interface/GeometricSearchTracker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/Records/interface/TrackerTopologyRcd.h"

using std::vector;
vector<CosmicLayerTriplets::LayerPairAndLayers> CosmicLayerTriplets::layers()
{
  vector<LayerPairAndLayers> result;

  if (_geometry=="STANDARD"){
    vector<const LayerWithHits*> third;
    third.push_back(lh3);
    result.push_back(LayerPairAndLayers(SeedLayerPairs::LayerPair(lh1,lh2),third));
  
  }
  if (_geometry=="TECPAIRS_TOBTRIPLETS"){
    vector<const LayerWithHits*> third;
    third.push_back(lh3);
    result.push_back(LayerPairAndLayers(SeedLayerPairs::LayerPair(lh1,lh2),third));
  
  }
  if (_geometry=="MTCC"){
    vector<const LayerWithHits*> third1;
    vector<const LayerWithHits*> third2;
    vector<const LayerWithHits*> third3;
    vector<const LayerWithHits*> third4;
    third1.clear();third2.clear();third3.clear();third4.clear();
    third1.push_back(lh1);
    result.push_back(LayerPairAndLayers(SeedLayerPairs::LayerPair(lh3,lh2),third1));
    third2.push_back(lh1);
    result.push_back(LayerPairAndLayers(SeedLayerPairs::LayerPair(lh2,lh1),third2));
    third3.push_back(lh2);
    result.push_back(LayerPairAndLayers(SeedLayerPairs::LayerPair(lh3,lh2),third3));
    third4.push_back(lh4);
    result.push_back(LayerPairAndLayers(SeedLayerPairs::LayerPair(lh2,lh3),third4));
  }
  return result;
}
CosmicLayerTriplets::~CosmicLayerTriplets()
{
  for(vector<LayerWithHits*>::const_iterator it=allLayersWithHits.begin(); it!=allLayersWithHits.end();it++){
    delete *it;
  }
}



void CosmicLayerTriplets::init(const SiStripRecHit2DCollection &collstereo,
			       const SiStripRecHit2DCollection &collrphi, 
			       const SiStripMatchedRecHit2DCollection &collmatched,
			       std::string geometry,
			       const edm::EventSetup& iSetup){

  _geometry=geometry;
  if(watchTrackerGeometry_.check(iSetup)){
    edm::ESHandle<GeometricSearchTracker> track;
    iSetup.get<TrackerRecoGeometryRecord>().get( track ); 
    bl=track->barrelLayers(); 
  }
  edm::ESHandle<TrackerTopology> httopo;
  iSetup.get<TrackerTopologyRcd>().get(httopo);
  const TrackerTopology& ttopo = *httopo;

  for(vector<LayerWithHits*>::const_iterator it=allLayersWithHits.begin(); 
      it!=allLayersWithHits.end();it++){
    delete *it;
  }

  allLayersWithHits.clear();
  LogDebug("CosmicSeedFinder") <<"Reconstruction for geometry  "<<_geometry;
  if (_geometry=="STANDARD"  || _geometry=="TECPAIRS_TOBTRIPLETS"){
    const DetLayer*  bl1=dynamic_cast<DetLayer const*>(bl[10]);
    const DetLayer*  bl2=dynamic_cast<DetLayer const*>(bl[11]);
    const DetLayer*  bl3=dynamic_cast<DetLayer const*>(bl[12]);
    //   //LayersWithHits
    lh1=new  LayerWithHits(bl1,collrphi,ttopo.tobDetIdLayerComparator(4));   allLayersWithHits.push_back(lh1);
    lh2=new  LayerWithHits(bl2,collrphi,ttopo.tobDetIdLayerComparator(5));   allLayersWithHits.push_back(lh2);
    lh3=new  LayerWithHits(bl3,collrphi,ttopo.tobDetIdLayerComparator(6));   allLayersWithHits.push_back(lh3);
  }
  if(_geometry=="MTCC"){ 

    const DetLayer*  bl1=dynamic_cast<DetLayer const*>(bl[0]);
    const DetLayer*  bl2=dynamic_cast<DetLayer const*>(bl[1]);
    const DetLayer*  bl3=dynamic_cast<DetLayer const*>(bl[2]);
    const DetLayer*  bl4=dynamic_cast<DetLayer const*>(bl[3]);

    lh1=new  LayerWithHits(bl1,collrphi,ttopo.tibDetIdLayerComparator(1)); allLayersWithHits.push_back(lh1);
    lh2=new  LayerWithHits(bl2,collrphi,ttopo.tibDetIdLayerComparator(2)); allLayersWithHits.push_back(lh2);
    lh3=new  LayerWithHits(bl3,collrphi,ttopo.tobDetIdLayerComparator(1)); allLayersWithHits.push_back(lh3);
    lh4=new  LayerWithHits(bl4,collrphi,ttopo.tobDetIdLayerComparator(2)); allLayersWithHits.push_back(lh4);
  }
}
