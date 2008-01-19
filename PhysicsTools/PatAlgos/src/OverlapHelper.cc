#include "PhysicsTools/PatAlgos/interface/OverlapHelper.h"

using pat::helper::OverlapHelper;

pat::helper::OverlapHelper::OverlapHelper(const std::vector<edm::ParameterSet> &psets) 
{
    typedef std::vector<edm::ParameterSet>::const_iterator VPI;
    workers_.reserve(psets.size());
    for (VPI it = psets.begin(), ed = psets.end(); it != ed; ++it) {
        workers_.push_back(Worker(*it));
    }
}


