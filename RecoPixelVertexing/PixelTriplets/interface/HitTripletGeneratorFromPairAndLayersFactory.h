#ifndef PixelTriplets_HitTripletGeneratorFromPairAndLayersFactory_H 
#define PixelTriplets_HitTripletGeneratorFromPairAndLayersFactory_H

#include "RecoPixelVertexing/PixelTriplets/interface/HitTripletGeneratorFromPairAndLayers.h"
#include <PluginManager/PluginFactory.h>

namespace edm {class ParameterSet;}

class HitTripletGeneratorFromPairAndLayersFactory : public  
    seal::PluginFactory< HitTripletGeneratorFromPairAndLayers* (const edm::ParameterSet&) > {
public:
  HitTripletGeneratorFromPairAndLayersFactory();
  virtual ~HitTripletGeneratorFromPairAndLayersFactory();
  static HitTripletGeneratorFromPairAndLayersFactory * get();
};
 
#endif
