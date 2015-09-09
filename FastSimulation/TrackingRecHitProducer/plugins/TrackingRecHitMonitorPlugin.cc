#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithm.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitAlgorithmFactory.h"
#include "FastSimulation/TrackingRecHitProducer/interface/TrackingRecHitProduct.h"

#include "DataFormats/DetId/interface/DetId.h"
#include "SimDataFormats/TrackingHit/interface/PSimHit.h"
#include "DataFormats/TrackerRecHit2D/interface/FastSingleTrackerRecHit.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/ServiceRegistry/interface/Service.h"
#include "CommonTools/UtilAlgos/interface/TFileService.h"

#include "TH1F.h"
#include "TH2F.h"

#include <string>
#include <iostream>

class TrackingRecHitMonitorPlugin:
    public TrackingRecHitAlgorithm
{
    private:
        edm::Service<TFileService> _fs;
        TH2F* _hist;
    public:
        TrackingRecHitMonitorPlugin(
            const std::string& name,
            const edm::ParameterSet& config,
            edm::ConsumesCollector& consumesCollector
        ):
            TrackingRecHitAlgorithm(name,config,consumesCollector)
        {
            double xmax = config.getParameter<double>( "xmax" );
            double ymax = config.getParameter<double>( "ymax" );

            _hist = _fs->make<TH2F>((name+"_xy").c_str(), ";dx/#sigma x;dy/#sigma y", 100,  -xmax, xmax,100,  -ymax, ymax);
        }

        virtual TrackingRecHitProductPtr process(TrackingRecHitProductPtr product) const
        {
            //std::cout<<getTrackerTopology()->print(product->getDetId())<<std::endl;

            for (unsigned int irechit = 0; irechit<product->numberOfRecHits(); ++irechit)
            {
                
                const FastSingleTrackerRecHit & recHit = product->getRecHit(irechit);
                const Local3DPoint& recHitPosition = recHit.localPosition();
                const LocalError& recHitError = recHit.localPositionError();
                double simHitXmean = 0;
                double simHitYmean = 0;
                for (const PSimHit* simHit: product->getSimHitsFromRecHit(irechit))
                {
                    const Local3DPoint& simHitPosition = simHit->localPosition();
                    simHitXmean+=simHitPosition.x();
                    simHitYmean+=simHitPosition.y();
                }
                simHitXmean/=product->getSimHitsFromRecHit(irechit).size();
                simHitYmean/=product->getSimHitsFromRecHit(irechit).size();

                _hist->Fill((simHitXmean-recHitPosition.x())/sqrt(recHitError.xx()),(simHitYmean-recHitPosition.y())/sqrt(recHitError.yy()));
            }
            return product;
        }
};

DEFINE_EDM_PLUGIN(
    TrackingRecHitAlgorithmFactory,
    TrackingRecHitMonitorPlugin,
    "TrackingRecHitMonitorPlugin"
);

