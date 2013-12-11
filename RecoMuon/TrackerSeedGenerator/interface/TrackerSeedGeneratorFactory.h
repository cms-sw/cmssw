#ifndef RecoMuon_TrackerSeedGenerator_TrackerSeedGeneratorFactory_H
#define RecoMuon_TrackerSeedGenerator_TrackerSeedGeneratorFactory_H

#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGenerator.h"
namespace edm {class ParameterSet; class ConsumesCollector;}

#include "FWCore/PluginManager/interface/PluginFactory.h"

//class TrackerSeedGeneratorFactory
//   : public seal::PluginFactory< TrackerSeedGenerator * (const edm::ParameterSet&) > {
//public:
//  TrackerSeedGeneratorFactory();
//  virtual ~TrackerSeedGeneratorFactory();
//  static TrackerSeedGeneratorFactory * get();
//};

typedef edmplugin::PluginFactory< TrackerSeedGenerator* (const edm::ParameterSet&, edm::ConsumesCollector&) > TrackerSeedGeneratorFactory;
#endif

