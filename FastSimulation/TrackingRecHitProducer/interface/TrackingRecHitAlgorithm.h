#ifndef FastSimulation_Tracking_RecHitAlgorithm_H
#define FastSimulation_Tracking_RecHitAlgorithm_H

#include <string>

namespace edm
{
    class ParameterSet;
    class ConsumesCollector;
}

class TrackingRecHitAlgorithm
{
    public:
        TrackingRecHitAlgorithm(
            const std::string&,
            const edm::ParameterSet&,
            edm::ConsumesCollector&
        )
        {
        }

        virtual ~TrackingRecHitAlgorithm()
        {
        }

};


#endif
