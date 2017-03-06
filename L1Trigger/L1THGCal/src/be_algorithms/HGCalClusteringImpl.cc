#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalClusteringImpl.h"

//class constructor
HGCalClusteringImpl::HGCalClusteringImpl(const edm::ParameterSet & conf):
    seedThreshold_(conf.getParameter<double>("seeding_threshold")),
    triggerCellThreshold_(conf.getParameter<double>("clustering_threshold")),
    dr_(conf.getParameter<double>("dR_cluster"))
{    
    
}


void HGCalClusteringImpl::clusterize( const l1t::HGCalTriggerCellBxCollection & trgcells, 
                                      l1t::HGCalClusterBxCollection & clusters
    ){

    edm::LogInfo("HGCclusterParameters") << "C2d seeding Thr: " << seedThreshold_ ; 
    edm::LogInfo("HGCclusterParameters") << "C2d clustering Thr: " << triggerCellThreshold_ ; 

    bool isSeed[trgcells.size()];

    /* search for cluster seeds */
    int itc=0;
    for( l1t::HGCalTriggerCellBxCollection::const_iterator tc = trgcells.begin(); tc != trgcells.end(); ++tc,++itc ){
        isSeed[itc] = ( tc->hwPt() > seedThreshold_) ? true : false;
    }
    /* clustering the TCs */
    std::vector<l1t::HGCalCluster> clustersTmp;

    itc=0;
    for( l1t::HGCalTriggerCellBxCollection::const_iterator tc = trgcells.begin(); tc != trgcells.end(); ++tc,++itc ){

        if( tc->hwPt() < triggerCellThreshold_ ){
            continue;
        }
        
        /* searching for TC near the center of the cluster  */
        int iclu=0;
        vector<int> tcPertinentClusters; 
        for( std::vector<l1t::HGCalCluster>::iterator clu = clustersTmp.begin(); clu != clustersTmp.end(); ++clu,++iclu ){
            if( clu->isPertinent(*tc, dr_) ){
                tcPertinentClusters.push_back(iclu);
            }
        }
        if( tcPertinentClusters.size() == 0 && isSeed[itc] ){
            l1t::HGCalCluster obj( *tc );
            clustersTmp.emplace_back( obj );
        }
        else if ( tcPertinentClusters.size() > 0 ){
         
            unsigned minDist = 300;
            unsigned targetClu = 0;
                        
            for( std::vector<int>::const_iterator iclu = tcPertinentClusters.begin(); iclu != tcPertinentClusters.end(); ++iclu ){
                double d = clustersTmp.at( *iclu).distance(*tc);
                if( d < minDist ){
                    minDist = d;
                    targetClu = *iclu;
                }
            } 

            clustersTmp.at( targetClu).addTriggerCell( *tc );
            
        }

    }

    /* making the collection of clusters */
    for( unsigned i(0); i<clustersTmp.size(); ++i ){
        clusters.push_back( 0, clustersTmp.at(i) );
    }
}


