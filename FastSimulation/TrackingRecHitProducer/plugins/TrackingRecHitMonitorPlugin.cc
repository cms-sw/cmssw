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
            double xmax = config.getParameter<double>( "dxmax" );
            double ymax = config.getParameter<double>( "dymax" );

            _hist = _fs->make<TH2F>((name+"_xy").c_str(), ";dx;dy", 50,  -xmax, xmax,50,  -ymax, ymax);
        }

        virtual TrackingRecHitProductPtr process(TrackingRecHitProductPtr product) const
        {
            //std::cout<<getTrackerTopology()->print(product->getDetId())<<std::endl;
            
            for (const TrackingRecHitProduct::RecHitToSimHitIdPairs& recHitToSimHitIdPair: product->getRecHitToSimHitIdPairs())
            {
                
                const FastSingleTrackerRecHit& recHit = recHitToSimHitIdPair.first;
                const Local3DPoint& recHitPosition = recHit.localPosition();
                double simHitXmean = 0;
                double simHitYmean = 0;
                for (const TrackingRecHitProduct::SimHitIdPair& simHitIdPair: recHitToSimHitIdPair.second)
                {
                    const PSimHit* simHit = simHitIdPair.second;
                    const Local3DPoint& simHitPosition = simHit->localPosition();
                    simHitXmean+=simHitPosition.x();
                    simHitYmean+=simHitPosition.y();
                }
                simHitXmean/= recHitToSimHitIdPair.second.size();
                simHitYmean/= recHitToSimHitIdPair.second.size();
                
                _hist->Fill((simHitXmean-recHitPosition.x()),(simHitYmean-recHitPosition.y()));
            }
            
            return product;
        }
};

DEFINE_EDM_PLUGIN(
    TrackingRecHitAlgorithmFactory,
    TrackingRecHitMonitorPlugin,
    "TrackingRecHitMonitorPlugin"
);

