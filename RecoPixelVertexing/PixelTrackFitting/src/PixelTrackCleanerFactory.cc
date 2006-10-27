#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>

PixelTrackCleanerFactory::PixelTrackCleanerFactory() 
  : seal::PluginFactory<PixelTrackCleaner*(const edm::ParameterSet & p)>("PixelTrackCleanerFactory")
{ }

PixelTrackCleanerFactory::~PixelTrackCleanerFactory()
{ }

PixelTrackCleanerFactory * PixelTrackCleanerFactory::get() 
{
  static PixelTrackCleanerFactory thePixelTrackCleanerFactory;
  return & thePixelTrackCleanerFactory;
}


