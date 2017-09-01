#ifndef __L1Trigger_L1THGCal_HGCalMulticlusteringImpl_h__
#define __L1Trigger_L1THGCal_HGCalMulticlusteringImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalShowerShape.h"

class HGCalMulticlusteringImpl{

public:

    HGCalMulticlusteringImpl( const edm::ParameterSet &conf);    

    bool isPertinent( const l1t::HGCalCluster & clu, 
                      const l1t::HGCalMulticluster & mclu, 
                      double dR ) const;

    void clusterizeDR( const edm::PtrVector<l1t::HGCalCluster> & clustersPtr, 
                     l1t::HGCalMulticlusterBxCollection & multiclusters);

    void clusterizeDBSCAN( const edm::PtrVector<l1t::HGCalCluster> & clustersPtr, 
                     l1t::HGCalMulticlusterBxCollection & multiclusters);

private:

    void findNeighbor( const edm::PtrVector<l1t::HGCalCluster> & clustersPtrs, 
		       const l1t::HGCalCluster & cluster,
		       std::vector<int> & neighborList);

    bool isNeighbor( const l1t::HGCalCluster & clu1, 
		     const l1t::HGCalCluster & clu2) const;
    
    double dr_;
    double ptC3dThreshold_;
    double calibSF_;
    string multiclusterAlgoType_;
    double distDbscan_ = 0.03;
    unsigned minNDbscan_ = 3;

    HGCalShowerShape shape_;

};

#endif
