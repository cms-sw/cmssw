#ifndef __L1Trigger_L1THGCal_HGCalMulticlusteringHistoImpl_h__
#define __L1Trigger_L1THGCal_HGCalMulticlusteringHistoImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalShowerShape.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerTools.h"
#include "L1Trigger/L1THGCal/interface/backend/HGCalTriggerClusterIdentificationBase.h"


class HGCalMulticlusteringHistoImpl{

public:

    HGCalMulticlusteringHistoImpl( const edm::ParameterSet &conf);    

    void eventSetup(const edm::EventSetup& es) 
    {
        triggerTools_.eventSetup(es);
        shape_.eventSetup(es);
    }

    float dR( const l1t::HGCalCluster & clu,
	      const GlobalPoint & seed ) const;

    void clusterizeHisto( const std::vector<edm::Ptr<l1t::HGCalCluster>> & clustersPtr,
			  l1t::HGCalMulticlusterBxCollection & multiclusters,
			  const HGCalTriggerGeometryBase & triggerGeometry
			  );


private:

    typedef std::map<std::array<int,3>,float> Histogram;

    Histogram fillHistoClusters( const std::vector<edm::Ptr<l1t::HGCalCluster>> & clustersPtrs );

    Histogram fillSmoothPhiHistoClusters( const Histogram & histoClusters,
					  const vector<unsigned> & binSums );

    Histogram fillSmoothRPhiHistoClusters( const Histogram & histoClusters );

    std::vector<GlobalPoint> computeMaxSeeds( const Histogram & histoClusters );

    std::vector<GlobalPoint> computeThresholdSeeds( const Histogram & histoClusters );

    std::vector<l1t::HGCalMulticluster> clusterSeedMulticluster(const std::vector<edm::Ptr<l1t::HGCalCluster>> & clustersPtrs,
								const std::vector<GlobalPoint> & seeds);

    void finalizeClusters(std::vector<l1t::HGCalMulticluster>&,
            l1t::HGCalMulticlusterBxCollection&,
            const HGCalTriggerGeometryBase&);
    
    double dr_;
    double ptC3dThreshold_;
    std::string multiclusterAlgoType_;
    unsigned nBinsRHisto_ = 36;
    unsigned nBinsPhiHisto_ = 216;
    std::vector<unsigned> binsSumsHisto_;
    double histoThreshold_ = 20.;

    HGCalShowerShape shape_;
    HGCalTriggerTools triggerTools_;
    std::unique_ptr<HGCalTriggerClusterIdentificationBase> id_;

    static constexpr double kROverZMin_ = 0.09;
    static constexpr double kROverZMax_ = 0.52;

};

#endif
