#ifndef __L1Trigger_L1THGCal_HGCalClusteringImpl_h__
#define __L1Trigger_L1THGCal_HGCalClusteringImpl_h__

#include <array> 
#include <string>
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "L1Trigger/L1THGCal/interface/HGCalTriggerGeometryBase.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class HGCalClusteringImpl{

private:
    static const unsigned kNSides_ = 2;
    // FIXME: currently there is no access to the HGCal DDDConstants
    // So cannot retrieve the following constants.
    static const unsigned kLayersEE_ = 28;
    static const unsigned kLayersFH_ = 12;
    static const unsigned kLayersBH_ = 12;
    static const unsigned kLayers_ = kLayersEE_+kLayersFH_+kLayersBH_;

public:
  
    HGCalClusteringImpl( const edm::ParameterSet & conf);    


    /* dR-algorithms */
    bool isPertinent( const l1t::HGCalTriggerCell & tc, 
                      const l1t::HGCalCluster & clu, 
                      double distXY) const;

    void clusterizeDR( const edm::PtrVector<l1t::HGCalTriggerCell> & triggerCellsPtrs,
                     l1t::HGCalClusterBxCollection & clusters 
        );

    /* NN-algorithms */    
    void mergeClusters( l1t::HGCalCluster & main_cluster, 
                        const l1t::HGCalCluster & secondary_cluster ) const;
    
    void NNKernel( const std::vector<edm::Ptr<l1t::HGCalTriggerCell>> &reshuffledTriggerCells,
                   l1t::HGCalClusterBxCollection & clusters,
                   const HGCalTriggerGeometryBase & triggerGeometry
        );
        
    void clusterizeNN( const edm::PtrVector<l1t::HGCalTriggerCell> & triggerCellsPtrs,
                     l1t::HGCalClusterBxCollection & clusters,
                     const HGCalTriggerGeometryBase & triggerGeometry 
        );

    //void showerShape2D( const edm::PtrVector<l1t::HGCalTriggerCell> & triggerCellsPtrs);


private:
    
    double siliconSeedThreshold_;
    double siliconTriggerCellThreshold_;
    double scintillatorSeedThreshold_;
    double scintillatorTriggerCellThreshold_;
    double dr_;
    std::string clusteringAlgorithmType_;
    void triggerCellReshuffling( const edm::PtrVector<l1t::HGCalTriggerCell> & triggerCellsPtrs, 
                                 std::array<std::array<std::vector<edm::Ptr<l1t::HGCalTriggerCell>>, kLayers_>, kNSides_> & reshuffledTriggerCells );
    
   // double sigmaEtaEta_;
   // double sigmaPhiPhi_;

};

#endif
