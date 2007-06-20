#ifndef _TrackHitsFilterFactory_h_
#define _TrackHitsFilterFactory_h_

#include "RecoPixelVertexing/PixelLowPtUtilities/interface/TrackHitsFilter.h"
#include "FWCore/PluginManager/interface/PluginFactory.h"

namespace edm {class ParameterSet;}

typedef edmplugin::PluginFactory<TrackHitsFilter *(const edm::ParameterSet &)> TrackHitsFilterFactory;

/*
class TrackHitsFilterFactory : public  
    seal::PluginFactory< TrackHitsFilter* (const edm::ParameterSet&) >
{
  public:
    TrackHitsFilterFactory();
    virtual ~TrackHitsFilterFactory();
    static TrackHitsFilterFactory * get();
};
*/
 
#endif
