#ifndef __L1Trigger_L1THGCal_HGCalClusteringImpl_h__
#define __L1Trigger_L1THGCal_HGCalClusteringImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"


class HGCalClusteringImpl{

public:
  
    HGCalClusteringImpl(const edm::ParameterSet &conf);    
    void clusterizeBase(const l1t::HGCalTriggerCellBxCollection&, l1t::HGCalClusterBxCollection&);

private:
    
    double seedThr_;
    double tcThr_;

};

#endif
