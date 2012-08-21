#ifndef RecoLuminosity_LumiProducer_NormFunctorPluginFactory_h
#define RecoLuminosity_LumiProducer_NormFunctorPluginFactory_h

#include "FWCore/PluginManager/interface/PluginFactory.h"
#include "RecoLuminosity/LumiProducer/interface/NormFunctor.h"
#include <string>

namespace lumi{
  typedef edmplugin::PluginFactory< lumi::NormFunctor*() > NormFunctorPluginFactory;
}//ns lumi
#endif
