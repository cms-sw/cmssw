#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithmFactory.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitProduct.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"

#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"

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
        ):
            TrackingRecHitAlgorithm(name,config,consumesCollector)
        {
            std::cout<<"created plugin with name: "<<name<<std::endl;
        }

        virtual TrackingRecHitProductPtr process(TrackingRecHitProductPtr product) const
        {
            std::cout<<getTrackerTopology()->print(product->getDetId())<<std::endl;
            
            return product;
        }

};

DEFINE_EDM_PLUGIN(
    TrackingRecHitAlgorithmFactory,
    TrackingRecHitNoSmearingPlugin,
    "TrackingRecHitNoSmearingPlugin"
);

