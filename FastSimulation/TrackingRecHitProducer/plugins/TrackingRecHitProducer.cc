#include "TrackingRecHitProducer.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"

#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithmFactory.h"


TrackingRecHitProducer::TrackingRecHitProducer(const edm::ParameterSet& config)
{
    edm::ConsumesCollector consumeCollector = consumesCollector();
    const edm::ParameterSet& pluginConfigs = config.getParameter<edm::ParameterSet>("plugins");
    const std::vector<std::string> psetNames = pluginConfigs.getParameterNamesForType<edm::ParameterSet>();

    for (unsigned int iplugin = 0; iplugin<psetNames.size(); ++iplugin)
    {
        const edm::ParameterSet& pluginConfig = pluginConfigs.getParameter<edm::ParameterSet>(psetNames[iplugin]);
        const std::string pluginName = pluginConfig.getParameter<std::string>("type");
        TrackingRecHitAlgorithm* recHitAlgorithm = TrackingRecHitAlgorithmFactory::get()->tryToCreate(pluginName,psetNames[iplugin],pluginConfig,consumeCollector);
        if (recHitAlgorithm)
        {
            _recHitAlgorithms.push_back(recHitAlgorithm);
        }
        else
        {
            edm::LogWarning("TrackingRecHitAlgorithm plugin not found: ") << "plugin name = "<<pluginName<<"\nconfiguration=\n"<<pluginConfig.dump();
        }
    }
}

void TrackingRecHitProducer::produce(edm::Event& event, const edm::EventSetup& eventSetup)
{
}

TrackingRecHitProducer::~TrackingRecHitProducer()
{
}


DEFINE_FWK_MODULE(TrackingRecHitProducer);
