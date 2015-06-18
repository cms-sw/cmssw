#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithmFactory.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitProduct.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/TrackerRecHit2D/interface/SiTrackerGSRecHit2D.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <iostream>

class TrackingRecHitNoSmearingPlugin:
    public TrackingRecHitAlgorithm
{
    private:
        double _error2;
    public:
        TrackingRecHitNoSmearingPlugin(
            const std::string& name,
            const edm::ParameterSet& config,
            edm::ConsumesCollector& consumesCollector
        ):
            TrackingRecHitAlgorithm(name,config,consumesCollector),
            _error2(0.001*0.001)
        {
            if (config.exists("error"))
            {
                _error2 = config.getParameter<double>("error");
                _error2*=_error2;
            }
        }

        virtual TrackingRecHitProductPtr process(TrackingRecHitProductPtr product) const
        {
            //std::cout<<getTrackerTopology()->print(product->getDetId())<<std::endl;


            for (const PSimHit* simHit: product->getSimHits())
            {
                const Local3DPoint& position = simHit->localPosition();
                LocalError error(_error2,_error2,_error2);
                const GeomDet* geomDet = getTrackerGeometry()->idToDetUnit(product->getDetId());

                //TODO: this is only a minimal example
                SiTrackerGSRecHit2D recHit(
                    position,   //const LocalPoint &
                    error,      //const LocalError &
                    *geomDet,    //GeomDet const &idet
                    0,          //const int simhitId
                    0,          //const int simtrackId
                    0,          //const uint32_t eeId
                    SiTrackerGSRecHit2D::ClusterRef(),//ClusterRef const &cluster
                    -1,         //const int pixelMultiplicityX
                    -1          //const int pixelMultiplicityY
                );
                product->getRecHits().push_back(recHit);
            }
            return product;
        }

};

DEFINE_EDM_PLUGIN(
    TrackingRecHitAlgorithmFactory,
    TrackingRecHitNoSmearingPlugin,
    "TrackingRecHitNoSmearingPlugin"
);

