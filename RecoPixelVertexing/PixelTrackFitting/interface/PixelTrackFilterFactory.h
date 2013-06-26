#ifndef PixelTrackFitting_PixelTrackFilterFactory_H 
#define PixelTrackFitting_PixelTrackFilterFactory_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

//class PixelTrackFilter;
namespace edm {class ParameterSet; class EventSetup; }

typedef edmplugin::PluginFactory<PixelTrackFilter *(const edm::ParameterSet & )> PixelTrackFilterFactory;
typedef edmplugin::PluginFactory<PixelTrackFilter *(const edm::ParameterSet &,const edm::EventSetup&  )> PixelTrackFilterWithESFactory;
 
#endif
