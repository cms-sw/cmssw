#ifndef __L1Trigger_L1THGCal_HGCalMulticlusteringImpl_h__
#define __L1Trigger_L1THGCal_HGCalMulticlusteringImpl_h__

#include <stdint.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"


class HGCalMulticlusteringImpl{

public:

    HGCalMulticlusteringImpl(const edm::ParameterSet &conf);    
    void clusterizeMultiple(std::unique_ptr<l1t::HGCalClusterBxCollection> &, std::unique_ptr<l1t::HGCalMulticlusterBxCollection> &);

private:
    
    double dR_forC3d_;
};

#endif
