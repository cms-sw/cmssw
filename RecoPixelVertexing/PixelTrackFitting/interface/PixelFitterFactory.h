#ifndef PixelTrackFitting_PixelFitterFactory_H 
#define PixelTrackFitting_PixelFitterFactory_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include <PluginManager/PluginFactory.h>

//class PixelFitter;
namespace edm {class ParameterSet;}

class PixelFitterFactory : public  
    seal::PluginFactory< PixelFitter* (const edm::ParameterSet&) > {
public:
  PixelFitterFactory();
  virtual ~PixelFitterFactory();
  static PixelFitterFactory * get();
};
 
#endif
