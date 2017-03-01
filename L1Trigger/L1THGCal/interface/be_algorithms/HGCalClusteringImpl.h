#ifndef __L1Trigger_L1THGCal_HGCalClusteringImpl_h__
#define __L1Trigger_L1THGCal_HGCalClusteringImpl_h__

#include <stdint.h>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"


class HGCalClusteringImpl{

public:
  
    HGCalClusteringImpl(const edm::ParameterSet &conf);    
    void clusterizeBase(std::unique_ptr<l1t::HGCalTriggerCellBxCollection> &, std::unique_ptr<l1t::HGCalClusterBxCollection>  &);

private:
    
    double seed_CUT_;
    double tc_CUT_;
};

#endif
