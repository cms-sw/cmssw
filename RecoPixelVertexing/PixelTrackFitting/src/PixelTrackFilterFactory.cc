#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>

PixelTrackFilterFactory::PixelTrackFilterFactory() 
  : seal::PluginFactory<PixelTrackFilter*(const edm::ParameterSet & p)>("PixelTrackFilterFactory")
{ }

PixelTrackFilterFactory::~PixelTrackFilterFactory()
{ }

PixelTrackFilterFactory * PixelTrackFilterFactory::get() 
{
  static PixelTrackFilterFactory thePixelTrackFilterFactory;
  return & thePixelTrackFilterFactory;
}


