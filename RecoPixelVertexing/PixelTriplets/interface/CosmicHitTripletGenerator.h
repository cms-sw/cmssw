#ifndef CosmicHitTripletGenerator_H
#define CosmicHitTripletGenerator_H

#include <vector>
#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGenerator.h"
#include "RecoPixelVertexing/PixelTriplets/interface/LayerTriplets.h"
#include "RecoPixelVertexing/PixelTriplets/interface/CosmicHitTripletGeneratorFromLayerTriplet.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "FWCore/Framework/interface/EventSetup.h"

class LayerWithHits;
class DetLayer;
class TrackingRegion;
class HitTripletGeneratorFromLayerTriplet;


/** \class CosmicHitTripletGenerator
 * Hides set of HitTripletGeneratorFromLayerTriplet generators.
 */

class CosmicHitTripletGenerator : public HitTripletGenerator{

  typedef vector<CosmicHitTripletGeneratorFromLayerTriplet *>   Container;

public:
  CosmicHitTripletGenerator(LayerTriplets& layers, const edm::EventSetup& iSetup);
  CosmicHitTripletGenerator(LayerTriplets& layers);


  ~CosmicHitTripletGenerator();

  /// add generators based on layers
    //  void  add(const DetLayer* inner, const DetLayer* outer);
    void  add(const LayerWithHits* inner, 
	      const LayerWithHits* middle,
	      const LayerWithHits* outer,
	      const edm::EventSetup& iSetup);
  /// form base class
  virtual void hitTriplets( const TrackingRegion& reg, 
			 OrderedHitTriplets & prs, 
			 const edm::EventSetup& iSetup);

  /// from base class
  virtual CosmicHitTripletGenerator * clone() const 
    { return new CosmicHitTripletGenerator(*this); }

private:


  Container        theGenerators;

};
#endif
