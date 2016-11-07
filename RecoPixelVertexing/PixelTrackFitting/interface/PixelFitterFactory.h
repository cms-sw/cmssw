#ifndef PixelTrackFitting_PixelFitterFactory_H 
#define PixelTrackFitting_PixelFitterFactory_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterBase.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

//class PixelFitter;
namespace edm {class ParameterSet;}

typedef edmplugin::PluginFactory<PixelFitterBase *(const edm::ParameterSet &)> PixelFitterFactory;
 
#endif
