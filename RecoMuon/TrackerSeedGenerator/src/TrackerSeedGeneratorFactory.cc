#include "RecoMuon/TrackerSeedGenerator/interface/TrackerSeedGeneratorFactory.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include <iostream>

TrackerSeedGeneratorFactory::TrackerSeedGeneratorFactory()
  : seal::PluginFactory<TrackerSeedGenerator* (const edm::ParameterSet & p)>("TrackerSeedGeneratorFactory")
{ }

TrackerSeedGeneratorFactory::~TrackerSeedGeneratorFactory()
{ }

TrackerSeedGeneratorFactory * TrackerSeedGeneratorFactory::get()
{
  static TrackerSeedGeneratorFactory theTrackerSeedGeneratorFactory;
  return & theTrackerSeedGeneratorFactory;
}

