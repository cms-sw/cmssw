#ifndef CosmicHitTripletGenerator_H
#define CosmicHitTripletGenerator_H

#include <vector>
#include "RecoPixelVertexing/PixelTriplets/interface/OrderedHitTriplets.h"
#include "RecoPixelVertexing/PixelTriplets/interface/CosmicHitTripletGeneratorFromLayerTriplet.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "FWCore/Framework/interface/EventSetup.h"

class LayerWithHits;
class DetLayer;
class TrackingRegion;
class CosmicLayerTriplets;


/** \class CosmicHitTripletGenerator
 * Hides set of HitTripletGeneratorFromLayerTriplet generators.
 */

class CosmicHitTripletGenerator {

  typedef std::vector<std::unique_ptr<CosmicHitTripletGeneratorFromLayerTriplet> >   Container;

public:
  CosmicHitTripletGenerator(CosmicLayerTriplets& layers, const edm::EventSetup& iSetup);
  CosmicHitTripletGenerator(CosmicLayerTriplets& layers);


  ~CosmicHitTripletGenerator();

  /// add generators based on layers
    //  void  add(const DetLayer* inner, const DetLayer* outer);
    void  add(const LayerWithHits* inner, 
	      const LayerWithHits* middle,
	      const LayerWithHits* outer,
	      const edm::EventSetup& iSetup);

  void hitTriplets( const TrackingRegion& reg,
                    OrderedHitTriplets & prs,
                    const edm::EventSetup& iSetup);

private:


  Container        theGenerators;

};
#endif
