#ifndef __L1Trigger_L1THGCal_HGCalMulticlusteringImpl_h__
#define __L1Trigger_L1THGCal_HGCalMulticlusteringImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalShowerShape.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"

class HGCalMulticlusteringImpl{

public:

    HGCalMulticlusteringImpl( const edm::ParameterSet &conf);    

    void eventSetup(const edm::EventSetup& es) 
    {
        triggerTools_.eventSetup(es);
        shape_.eventSetup(es);
    }

    bool isPertinent( const l1t::HGCalCluster & clu, 
                      const l1t::HGCalMulticluster & mclu, 
                      double dR ) const;

    void clusterizeDR( const edm::PtrVector<l1t::HGCalCluster> & clustersPtr, 
                     l1t::HGCalMulticlusterBxCollection & multiclusters,
                     const HGCalTriggerGeometryBase & triggerGeometry
                     );

    void clusterizeDBSCAN( const edm::PtrVector<l1t::HGCalCluster> & clustersPtr, 
                     l1t::HGCalMulticlusterBxCollection & multiclusters,
                     const HGCalTriggerGeometryBase & triggerGeometry
                     );

private:

    void findNeighbor( const std::vector<std::pair<unsigned int,double>>&  rankedList,
                       unsigned int searchInd,
                       const edm::PtrVector<l1t::HGCalCluster> & clustersPtr, 
                       std::vector<unsigned int>& neigbors);
    
    double dr_;
    double ptC3dThreshold_;
    double calibSF_;
    std::string multiclusterAlgoType_;
    double distDbscan_ = 0.005;
    unsigned minNDbscan_ = 3;
    std::vector<double> layerWeights_;
    bool applyLayerWeights_;

    HGCalShowerShape shape_;
    HGCalTriggerTools triggerTools_;

};

#endif
