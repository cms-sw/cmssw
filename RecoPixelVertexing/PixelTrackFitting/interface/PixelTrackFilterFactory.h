#ifndef PixelTrackFitting_PixelTrackFilterFactory_H 
#define PixelTrackFitting_PixelTrackFilterFactory_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
#include <PluginManager/PluginFactory.h>

//class PixelTrackFilter;
namespace edm {class ParameterSet;}

class PixelTrackFilterFactory : public  
    seal::PluginFactory< PixelTrackFilter* (const edm::ParameterSet&) > {
public:
  PixelTrackFilterFactory();
  virtual ~PixelTrackFilterFactory();
  static PixelTrackFilterFactory * get();
};
 
#endif
