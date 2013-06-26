#ifndef PixelTrackFitting_PixelTrackCleanerFactory_H 
#define PixelTrackFitting_PixelTrackCleanerFactory_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

//class PixelTrackCleaner;
namespace edm {class ParameterSet;}

typedef edmplugin::PluginFactory<PixelTrackCleaner *(const edm::ParameterSet &)> PixelTrackCleanerFactory;
 
#endif
