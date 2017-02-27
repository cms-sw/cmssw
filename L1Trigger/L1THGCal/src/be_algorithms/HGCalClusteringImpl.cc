#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalClusteringImpl.h"


HGCalClusteringImpl::HGCalClusteringImpl(const edm::ParameterSet & beCodecConfig){    
    
    seedThr_ = beCodecConfig.getParameter<double>("seeding_threshold");
    tcThr_   = beCodecConfig.getParameter<double>("clustering_threshold");
    dr_      = beCodecConfig.getParameter<double>("dR_cluster");

}


void HGCalClusteringImpl::clusterise( const l1t::HGCalTriggerCellBxCollection & trgcells_, 
                                      l1t::HGCalClusterBxCollection & clusters_
    ){
    
    edm::LogInfo("HGCclusterParameters") << "C2d seeding Thr: " << seedThr_ ; 
    edm::LogInfo("HGCclusterParameters") << "C2d clustering Thr: " << tcThr_ ; 
    
    bool isSeed[trgcells_.size()];

    /* seeding the TCs */
    int itc=0;
    for( l1t::HGCalTriggerCellBxCollection::const_iterator tc = trgcells_.begin(); tc != trgcells_.end(); ++tc,++itc )
        isSeed[itc] = (tc->hwPt() > seedThr_) ? true : false;

    itc=0;
    /* clustering the TCs */
    for( l1t::HGCalTriggerCellBxCollection::const_iterator tc = trgcells_.begin(); tc != trgcells_.end(); ++tc,++itc ){

        int iclu=0;
        vector<int> tcPertinentClusters; 
        /* searching for TC near the center of the cluster  */
        for( l1t::HGCalClusterBxCollection::const_iterator clu = clusters_.begin(); clu != clusters_.end(); ++clu,++iclu )
            if( clu->isPertinent(*tc, dr_) )
                tcPertinentClusters.push_back(iclu);

        if( tcPertinentClusters.size() == 0 && isSeed[itc] ){
            l1t::HGCalCluster obj( *tc );
            clusters_.push_back( 0, obj );
        }
        else{
            uint minDist = 1;
            uint targetClu = 0; 
            for( std::vector<int>::const_iterator iclu = tcPertinentClusters.begin(); iclu != tcPertinentClusters.end(); ++iclu ){
                double d = clusters_.at(0, *iclu).dist(*tc);
                if( d < minDist ){
                    minDist = d;
                    targetClu = *iclu;
                }
            } 

            clusters_.at(0, targetClu).addTC( *tc );
            
        }

    }

}

