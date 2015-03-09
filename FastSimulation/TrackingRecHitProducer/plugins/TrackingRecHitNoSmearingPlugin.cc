#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithmFactory.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"

#include "FWCore/Utilities/interface/FunctionWithDict.h"
#include "FWCore/Utilities/interface/ObjectWithDict.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"

#include <string>
#include <iostream>

class TrackingRecHitNoSmearingPlugin:
    public TrackingRecHitAlgorithm
{
    public:
        TrackingRecHitNoSmearingPlugin(
            const std::string& name,
            const edm::ParameterSet& config,
            edm::ConsumesCollector& consumesCollector
        )
        {
            std::cout<<"created plugin with name: "<<name<<std::endl;
        }

        virtual std::vector<SiTrackerGSRecHit2D> processDetId(const DetId& detId, const std::vector<const PSimHit*>& simHits) const
        {
            DetId id(detId);
            edm::ObjectWithDict obj(typeid(detId),&id);
            edm::TypeWithDict oType(typeid(detId));
            edm::FunctionWithDict fct = oType.functionMemberByName("subdetId");
            unsigned int r = 0;
            edm::ObjectWithDict ret(typeid(unsigned int),&r);
            fct.invoke(obj,&ret);
            std::cout<<"subdet: "<<r<<std::endl;
            std::vector<SiTrackerGSRecHit2D> recHits;
/*
            SiTrackerGSRecHit2D* recHit = new SiTrackerGSRecHit2D(
                  position,
                  error,
                  theDetUnit,
                  simHitCounter,
                  trackID,
                  eeID,
                  ClusterRef(FastTrackerClusterRefProd, simHitCounter),
                  alphaMult,
                  betaMult
              );
              */
            return recHits;
        }

};

DEFINE_EDM_PLUGIN(
    TrackingRecHitAlgorithmFactory,
    TrackingRecHitNoSmearingPlugin,
    "TrackingRecHitNoSmearingPlugin"
);

