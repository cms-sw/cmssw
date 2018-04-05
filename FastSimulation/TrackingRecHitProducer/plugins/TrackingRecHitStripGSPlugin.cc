#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithmFactory.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitProduct.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastSingleTrackerRecHit.h"
#include "DataFormats/GeometrySurface/interface/Plane.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <iostream>

/* This plugin performs simple Gaussian smearing (GS) of the sim hit position
 * per module. If y resolution < 0 (default) the module length/sqrt(12) is taken as uncertainty.
 */

class TrackingRecHitStripGSPlugin:
    public TrackingRecHitAlgorithm
{
    private:
        double _resolutionX;
        double _resolutionX2;

        double _resolutionY;
        double _resolutionY2;

        constexpr static double INV12 = 1.0/12.0;

    public:
        TrackingRecHitStripGSPlugin(
            const std::string& name,
            const edm::ParameterSet& config,
            edm::ConsumesCollector& consumesCollector
        ):
            TrackingRecHitAlgorithm(name,config,consumesCollector),
            _resolutionX(0.001),
            _resolutionX2(_resolutionX*_resolutionX),
            _resolutionY(-1),
            _resolutionY2(_resolutionY*_resolutionY)

        {
            if (config.exists("resolutionX"))
            {
                _resolutionX = config.getParameter<double>("resolutionX");
                _resolutionX2=_resolutionX*_resolutionX;
            }
            if (config.exists("resolutionY"))
            {
                _resolutionY = config.getParameter<double>("resolutionY");
                _resolutionY2=_resolutionY*_resolutionY;
            }
        }

        TrackingRecHitProductPtr process(TrackingRecHitProductPtr product) const override
        {
            for (const std::pair<unsigned int,const PSimHit*>& simHitIdPair: product->getSimHitIdPairs())
            {
                const PSimHit* simHit = simHitIdPair.second;
                const Local3DPoint& simHitPosition = simHit->localPosition();

                const GeomDet* geomDet = this->getTrackerGeometry().idToDetUnit(product->getDetId());
                const Plane& plane = geomDet->surface();
                const Bounds& bounds = plane.bounds();
                const double boundY = bounds.length();

                Local3DPoint recHitPosition;
                
                unsigned int retry = 0;
                do
                {
                    recHitPosition = Local3DPoint(
                        simHitPosition.x()+this->getRandomEngine().gaussShoot(0.0,_resolutionX),
                        0.0, 0.0 //fix y & z coordinates to center of module
                    );
                    
                    // If we tried to generate thePosition, and it's out of the bounds
                    // for 10 times, then take and return the simHit's location.
                    ++retry;
                    if (retry>10)
                    {
                        recHitPosition = Local3DPoint(
                            simHitPosition.x(),
                            0.0, 0.0 //fix y & z coordinates to center of module
                        );
                        break;
                    }
                }
                while (not bounds.inside(recHitPosition));

                LocalError error(
                    //xx (variance)
                    _resolutionX2,
                     //xy (covariance)
                    0.0,
                    //take here the provided y resolution or (lenght/sqrt(12))^2
                    _resolutionY<0.0 ? boundY*boundY*INV12 : _resolutionY2
                );

                FastSingleTrackerRecHit recHit(
                    recHitPosition,
                    error,
                    *geomDet,
                    fastTrackerRecHitType::siStrip1D
                );
                product->addRecHit(recHit,{simHitIdPair});
            }
            return product;
        }
};

DEFINE_EDM_PLUGIN(
    TrackingRecHitAlgorithmFactory,
    TrackingRecHitStripGSPlugin,
    "TrackingRecHitStripGSPlugin"
);

