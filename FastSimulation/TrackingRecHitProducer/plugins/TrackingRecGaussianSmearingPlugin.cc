#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithmFactory.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitProduct.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastSingleTrackerRecHit.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <string>
#include <iostream>

class TrackingRecGaussianSmearingPlugin:
    public TrackingRecHitAlgorithm
{
    private:
        double _resolutionX;
        double _resolutionX2;
        
        double _resolutionY;
        double _resolutionY2;
        
        constexpr static double INV12 = 1.0/12.0;
        
    public:
        TrackingRecGaussianSmearingPlugin(
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
                _resolutionX2*=_resolutionX;
            }
            if (config.exists("resolutionY"))
            {
                _resolutionY = config.getParameter<double>("resolutionY");
                _resolutionY2*=_resolutionY;
            }
        }

        virtual TrackingRecHitProductPtr process(TrackingRecHitProductPtr product) const
        {
            for (const std::pair<unsigned int,const PSimHit*>& simHitIdPair: product->getSimHitIdPairs())
            {
                const PSimHit* simHit = simHitIdPair.second;
                const Local3DPoint& simHitPosition = simHit->localPosition();
                
                const GeomDet* geomDet = this->getTrackerGeometry().idToDetUnit(product->getDetId());
                const BoundPlane& plane = geomDet->surface();
                const Bounds& bounds = plane.bounds();
                const double boundX = bounds.width()/2.;
                const double boundY = bounds.length()/2.;
                
                
                
                Local3DPoint recHitPosition;
                do
                {
                    recHitPosition = Local3DPoint(this->getRandomEngine().gaussShoot(simHitPosition.x(),_resolutionX),0.0,0.0);
                }
                while (fabs(recHitPosition.x()) > boundX);
                
                
                LocalError error(
                    _resolutionX2, //xx (variance)
                    0.0, //xy (covariance)
                    _resolutionY<0 ? boundY*boundY*INV12: _resolutionY2 //take here the provided y resolution or (lenght/sqrt(12))^2
                );
                FastSingleTrackerRecHit recHit(
                    recHitPosition,   //const LocalPoint &
                    error,      //const LocalError &
                    *geomDet,    //GeomDet const &idet
		            fastTrackerRecHitType::siPixel
	            );
                product->addRecHit(recHit,{simHitIdPair});
            }
            return product;
        }
};

DEFINE_EDM_PLUGIN(
    TrackingRecHitAlgorithmFactory,
    TrackingRecGaussianSmearingPlugin,
    "TrackingRecGaussianSmearingPlugin"
);

