#include "RecoPixelVertexing/PixelTriplets/interface/CosmicLayerTriplets.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoPixelVertexing/PixelTriplets/interface/CosmicHitTripletGenerator.h"

#include <vector>

using namespace std;

CosmicHitTripletGenerator::CosmicHitTripletGenerator(CosmicLayerTriplets& layers,
						   const edm::EventSetup& iSetup)
{
  //  vector<LayerTriplets::LayerTriplet> layerTriplets = layers();
  vector<CosmicLayerTriplets::LayerPairAndLayers> layerTriplets = layers.layers();
  vector<CosmicLayerTriplets::LayerPairAndLayers>::const_iterator it;
  for (it = layerTriplets.begin(); it != layerTriplets.end(); it++) {
    vector<const LayerWithHits*>::const_iterator ilwh;
    for(ilwh=(*it).second.begin();ilwh!=(*it).second.end();ilwh++){
 //      const LayerWithHits* first=(*it).first.first;
//       const LayerWithHits* second=(*it).first.second;
//       const LayerWithHits* third=(*ilwh);
      //      add( (*it).first.first, (*it).first.second, (*it).second,iSetup);
      add( (*it).first.first, (*it).first.second, (*ilwh),iSetup);
    }
  }

}



CosmicHitTripletGenerator::~CosmicHitTripletGenerator()
{
}


void CosmicHitTripletGenerator::add(
				   const LayerWithHits *inner, 
				   const LayerWithHits* middle,
				   const LayerWithHits *outer,
				   const edm::EventSetup& iSetup) 
{ 
  theGenerators.push_back(std::make_unique<CosmicHitTripletGeneratorFromLayerTriplet>( inner,middle, outer, iSetup));
}

void CosmicHitTripletGenerator::hitTriplets(
					const TrackingRegion& region, 
					OrderedHitTriplets & pairs,
					const edm::EventSetup& iSetup)
{

  Container::const_iterator i;
  for (i=theGenerators.begin(); i!=theGenerators.end(); i++) {
    (**i).hitTriplets( region, pairs, iSetup); 
  }
 
}


