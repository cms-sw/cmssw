#ifndef RecoMuon_TrackerSeedGenerator_TrackerSeedGeneratorFactory_H
#define RecoMuon_TrackerSeedGenerator_TrackerSeedGeneratorFactory_H

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"
namespace edm {class ParameterSet;}

#include <PluginManager/PluginFactory.h>

class TrackerSeedGeneratorFactory
   : public seal::PluginFactory< TrackerSeedGenerator * (const edm::ParameterSet&) > {
public:
  TrackerSeedGeneratorFactory();
  virtual ~TrackerSeedGeneratorFactory();
  static TrackerSeedGeneratorFactory * get();
};
#endif

