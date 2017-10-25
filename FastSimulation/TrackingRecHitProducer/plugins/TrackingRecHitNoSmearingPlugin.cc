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
        double _errorXX;
        double _errorXY;
        double _errorYY;
    public:
        TrackingRecHitNoSmearingPlugin(
            const std::string& name,
            const edm::ParameterSet& config,
            edm::ConsumesCollector& consumesCollector
        ):
            TrackingRecHitAlgorithm(name,config,consumesCollector),
            _errorXX(0.001*0.001),
            _errorXY(0.0),
            _errorYY(0.001*0.001)
        {
            if (config.exists("errorXX"))
            {
                _errorXX = config.getParameter<double>("errorXX");
            }
            
            if (config.exists("errorXY"))
            {
                _errorXY = config.getParameter<double>("errorXY");
            }
            
            if (config.exists("errorYY"))
            {
                _errorYY = config.getParameter<double>("errorYY");
            }
        }

        TrackingRecHitProductPtr process(TrackingRecHitProductPtr product) const override
        {
            for (const std::pair<unsigned int,const PSimHit*>& simHitIdPair: product->getSimHitIdPairs())
            {
                const PSimHit* simHit = simHitIdPair.second;
                const Local3DPoint& position = simHit->localPosition();
                LocalError error(_errorXX,_errorXY,_errorYY);
                const GeomDet* geomDet = getTrackerGeometry().idToDetUnit(product->getDetId());

                //TODO: this is only a minimal example
                FastSingleTrackerRecHit recHit(
                    position,   //const LocalPoint &
                    error,      //const LocalError &
                    *geomDet,    //GeomDet const &idet
		            fastTrackerRecHitType::siPixel // since this is a dummy class anyway: pretend all hits are pixel hits (only effect: hits are defined in 2D (?))
		        );
                product->addRecHit(recHit,{simHitIdPair});
            }
            
            return product;
        }
};

DEFINE_EDM_PLUGIN(
    TrackingRecHitAlgorithmFactory,
    TrackingRecHitNoSmearingPlugin,
    "TrackingRecHitNoSmearingPlugin"
);

