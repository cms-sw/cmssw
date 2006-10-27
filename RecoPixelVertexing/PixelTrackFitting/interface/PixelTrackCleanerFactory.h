#ifndef PixelTrackFitting_PixelTrackCleanerFactory_H 
#define PixelTrackFitting_PixelTrackCleanerFactory_H

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"
#include <PluginManager/PluginFactory.h>

//class PixelTrackCleaner;
namespace edm {class ParameterSet;}

class PixelTrackCleanerFactory : public  
    seal::PluginFactory< PixelTrackCleaner* (const edm::ParameterSet&) > {
public:
  PixelTrackCleanerFactory();
  virtual ~PixelTrackCleanerFactory();
  static PixelTrackCleanerFactory * get();
};
 
#endif
