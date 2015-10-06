#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithmFactory.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitProduct.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastSingleTrackerRecHit.h"

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
                const GeomDet* geomDet = getTrackerGeometry().idToDetUnit(product->getDetId());

                //TODO: this is only a minimal example
                FastSingleTrackerRecHit recHit(
                    position,   //const LocalPoint &
                    error,      //const LocalError &
                    *geomDet,    //GeomDet const &idet
		    fastTrackerRecHitType::siPixel // since this is a dummy class anyway: pretend all hits are pixel hits (only effect: hits are defined in 2D (?))
		);
                product->addRecHit(recHit,{simHit});
            }
            return product;
        }
};

DEFINE_EDM_PLUGIN(
    TrackingRecHitAlgorithmFactory,
    TrackingRecHitNoSmearingPlugin,
    "TrackingRecHitNoSmearingPlugin"
);

