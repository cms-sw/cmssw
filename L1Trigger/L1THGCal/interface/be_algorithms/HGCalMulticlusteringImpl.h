#ifndef __L1Trigger_L1THGCal_HGCalMulticlusteringImpl_h__
#define __L1Trigger_L1THGCal_HGCalMulticlusteringImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class HGCalMulticlusteringImpl{

public:

    HGCalMulticlusteringImpl( const edm::ParameterSet &conf);    
    void clusterise( const l1t::HGCalClusterBxCollection & clusters_, 
                     l1t::HGCalMulticlusterBxCollection & multiclusters_);
 

private:
    
    double dr_;
    BXVector<edm::PtrVector<l1t::HGCalTriggerCell>> clustersTCcollections_; 

};

#endif
