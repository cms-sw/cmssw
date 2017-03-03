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
        isSeed[itc] = ( tc->hwPt() > seedThr_) ? true : false;

    /* clustering the TCs */
    std::vector<l1t::HGCalCluster> clustersTmp;
    itc=0;
    for( l1t::HGCalTriggerCellBxCollection::const_iterator tc = trgcells_.begin(); tc != trgcells_.end(); ++tc,++itc ){

        if( tc->hwPt() < tcThr_ )
            continue;

        /* searching for TC near the center of the cluster  */
        int iclu=0;
        vector<int> tcPertinentClusters; 
        for( std::vector<l1t::HGCalCluster>::iterator clu = clustersTmp.begin(); clu != clustersTmp.end(); ++clu,++iclu )
            if( clu->isPertinent(*tc, dr_) )
                tcPertinentClusters.push_back(iclu);

        if( tcPertinentClusters.size() == 0 && isSeed[itc] ){
            //edm::PtrVector<l1t::HGCalTriggerCell> coll;
            //clustersTCcollections_.push_back( coll );
            l1t::HGCalCluster obj( *tc );
            clustersTmp.push_back( obj );
        }
        else if ( tcPertinentClusters.size() > 0 ){
         
            uint minDist = 300;
            uint targetClu = 0;
                        
            for( std::vector<int>::const_iterator iclu = tcPertinentClusters.begin(); iclu != tcPertinentClusters.end(); ++iclu ){
                double d = clustersTmp.at( *iclu).dist(*tc);
                if( d < minDist ){
                    minDist = d;
                    targetClu = *iclu;
                }
            } 

            clustersTmp.at( targetClu).addTC( *tc );
            
        }

    }

    /* making the collection of clusters */
    for( unsigned i(0); i<clustersTmp.size(); ++i )
        clusters_.push_back( 0, clustersTmp.at(i) );

}


