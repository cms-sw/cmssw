#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>

PixelFitterFactory::PixelFitterFactory() 
  : seal::PluginFactory<PixelFitter*(const edm::ParameterSet & p)>("PixelFitterFactory")
{ }

PixelFitterFactory::~PixelFitterFactory()
{ }

PixelFitterFactory * PixelFitterFactory::get() 
{
  static PixelFitterFactory thePixelFitterFactory;
  return & thePixelFitterFactory;
}


