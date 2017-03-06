
#ifndef __L1Trigger_L1THGCal_HGCalClusteringImpl_h__
#define __L1Trigger_L1THGCal_HGCalClusteringImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class HGCalClusteringImpl{

public:
  
    HGCalClusteringImpl( const edm::ParameterSet & conf);    
    void clusterize( const l1t::HGCalTriggerCellBxCollection & trgcells, 
                     l1t::HGCalClusterBxCollection & clusters 
        );

private:
    
    double seedThreshold_;
    double triggerCellThreshold_;
    double dr_;

};

#endif
