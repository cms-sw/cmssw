#ifndef __L1Trigger_L1THGCal_HGCalMulticlusteringImpl_h__
#define __L1Trigger_L1THGCal_HGCalMulticlusteringImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class HGCalMulticlusteringImpl{

public:

    HGCalMulticlusteringImpl( const edm::ParameterSet &conf);    

    void clusterize( const edm::OrphanHandle<l1t::HGCalClusterBxCollection> clustersHandle, 
                     l1t::HGCalMulticlusterBxCollection & multiclusters);
 
    bool isPertinent( const l1t::HGCalCluster & clu, const l1t::HGCalMulticluster & mclu, double dR ) const;

private:
    
    double dr_;

};

#endif
