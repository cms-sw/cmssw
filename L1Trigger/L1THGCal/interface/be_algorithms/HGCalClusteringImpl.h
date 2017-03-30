#ifndef __L1Trigger_L1THGCal_HGCalClusteringImpl_h__
#define __L1Trigger_L1THGCal_HGCalClusteringImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class HGCalClusteringImpl{

public:
  
    HGCalClusteringImpl( const edm::ParameterSet & conf);    

    /* dR-algorithms */
    bool isPertinent( const l1t::HGCalTriggerCell & tc, 
                      const l1t::HGCalCluster & clu, 
                      double distXY) const;

    void clusterize( const edm::PtrVector<l1t::HGCalTriggerCell> & triggerCellsPtrs,
                     l1t::HGCalClusterBxCollection & clusters 
        );

    /* NN-algorithms */
    bool isPertinent( const l1t::HGCalTriggerCell & tc, 
                      const l1t::HGCalCluster & clu, 
                      edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry ) const;
    
    void clusterize( const edm::PtrVector<l1t::HGCalTriggerCell> & triggerCellsPtrs,
                       l1t::HGCalClusterBxCollection & clusters,
                       edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry 
        );

private:
    
    double seedThreshold_;
    double triggerCellThreshold_;
    double dr_;

};

#endif
