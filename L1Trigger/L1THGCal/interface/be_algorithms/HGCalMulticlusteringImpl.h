#ifndef __L1Trigger_L1THGCal_HGCalMulticlusteringImpl_h__
#define __L1Trigger_L1THGCal_HGCalMulticlusteringImpl_h__

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/L1THGCal/interface/HGCalTriggerCell.h"
#include "DataFormats/L1THGCal/interface/HGCalCluster.h"
#include "DataFormats/L1THGCal/interface/HGCalMulticluster.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

class HGCalMulticlusteringImpl{

public:

    HGCalMulticlusteringImpl( const edm::ParameterSet &conf);    

    bool isPertinent( const l1t::HGCalCluster & clu, 
                      const l1t::HGCalMulticluster & mclu, 
                      double dR ) const;

    void clusterize( const edm::PtrVector<l1t::HGCalCluster> & clustersPtr, 
                     l1t::HGCalMulticlusterBxCollection & multiclusters);

    void showerShape3D(const edm::PtrVector<l1t::HGCalCluster> & clustersPtr);

private:
    
    double dr_;
    double ptC3dThreshold_;
    double calibSF_;
    double Nlayers_;
    double EMax_; //Maximum energy deposited in a layer
    double SeeTot_; //SigmaEtaEta considering all TC in 3DC
    double SeeMax_; //Maximum SigmaEtaEta in a layer
    double SppTot_; //same but for SigmaPhiPhi
    double SppMax_;


};

#endif
