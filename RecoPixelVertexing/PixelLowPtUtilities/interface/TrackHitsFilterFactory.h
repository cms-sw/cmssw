#ifndef _TrackHitsFilterFactory_h_
#define _TrackHitsFilterFactory_h_

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/TrackHitsFilter.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm {class ParameterSet; class EventSetup; }

typedef edmplugin::PluginFactory<TrackHitsFilter 
  *(const edm::ParameterSet &,
    const edm::EventSetup   &) > TrackHitsFilterFactory;

#endif
