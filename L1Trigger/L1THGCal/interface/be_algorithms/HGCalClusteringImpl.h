
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

    bool isPertinent( const l1t::HGCalTriggerCell &tc, l1t::HGCalCluster &clu, double distXY) const;

private:
    
    double seedThreshold_;
    double triggerCellThreshold_;
    double dr_;

};

#endif
