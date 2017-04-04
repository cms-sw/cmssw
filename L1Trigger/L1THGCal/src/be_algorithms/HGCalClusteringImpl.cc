#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalClusteringImpl.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

//class constructor
HGCalClusteringImpl::HGCalClusteringImpl(const edm::ParameterSet & conf):
    seedThreshold_(conf.getParameter<double>("seeding_threshold")),
    triggerCellThreshold_(conf.getParameter<double>("clustering_threshold")),
    dr_(conf.getParameter<double>("dR_cluster"))
{    

    edm::LogInfo("HGCalClusterParameters") << "C2d seeding Thr: " << seedThreshold_ ; 
    edm::LogInfo("HGCalClusterParameters") << "C2d clustering Thr: " << triggerCellThreshold_ ; 
 
}


/* dR-algorithms */
bool HGCalClusteringImpl::isPertinent( const l1t::HGCalTriggerCell & tc, 
                                       const l1t::HGCalCluster & clu, 
                                       double distXY ) const 
{

    HGCalDetId tcDetId( tc.detId() );
    HGCalDetId cluDetId( clu.seedDetId() );
    if( (tcDetId.layer() != cluDetId.layer()) ||
        (tcDetId.subdetId() != cluDetId.subdetId()) ||
        (tcDetId.zside() != cluDetId.zside()) ){
        return false;
    }   
    if ( clu.distance((tc)) < distXY ){
        return true;
    }
    return false;

}


/* dR-algorithms */
void HGCalClusteringImpl::clusterize( const edm::PtrVector<l1t::HGCalTriggerCell> & triggerCellsPtrs, 
                                      l1t::HGCalClusterBxCollection & clusters
    ){

    bool isSeed[triggerCellsPtrs.size()];
    
    /* search for cluster seeds */
    int itc=0;
    for( edm::PtrVector<l1t::HGCalTriggerCell>::const_iterator tc = triggerCellsPtrs.begin(); tc != triggerCellsPtrs.end(); ++tc,++itc ){
        isSeed[itc] = ( (*tc)->mipPt() > seedThreshold_) ? true : false;
    }
    
    /* clustering the TCs */
    std::vector<l1t::HGCalCluster> clustersTmp;

    itc=0;
    for( edm::PtrVector<l1t::HGCalTriggerCell>::const_iterator tc = triggerCellsPtrs.begin(); tc != triggerCellsPtrs.end(); ++tc,++itc ){
            
        if( (*tc)->mipPt() < triggerCellThreshold_ ){
            continue;
        }
        
        /* searching for TC near the center of the cluster  */
        int iclu=0;
        vector<int> tcPertinentClusters; 
        for( std::vector<l1t::HGCalCluster>::iterator clu = clustersTmp.begin(); clu != clustersTmp.end(); ++clu,++iclu ){
            if( this->isPertinent(**tc, *clu, dr_) ){
                tcPertinentClusters.push_back(iclu);
            }
        }
        if( tcPertinentClusters.size() == 0 && isSeed[itc] ){
            clustersTmp.emplace_back( *tc );
        }
        else if ( tcPertinentClusters.size() > 0 ){
         
            unsigned minDist = 300;
            unsigned targetClu = 0;
                        
            for( std::vector<int>::const_iterator iclu = tcPertinentClusters.begin(); iclu != tcPertinentClusters.end(); ++iclu ){
                double d = clustersTmp.at(*iclu).distance(**tc);
                if( d < minDist ){
                    minDist = d;
                    targetClu = *iclu;
                }
            } 

            clustersTmp.at(targetClu).addTriggerCell( *tc );                    

        }
    }

    /* store clusters in the persistent collection */
    for( unsigned i(0); i<clustersTmp.size(); ++i ){
        clusters.push_back( 0, clustersTmp.at(i) );
    }
    
}



/* NN-algorithms */

/* tc-tc */
bool HGCalClusteringImpl::isPertinent( const l1t::HGCalTriggerCell & tc1, 
                                       const l1t::HGCalTriggerCell & tc2, 
                                       edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry ) const 
{
    HGCalDetId detId_tc1( tc1.detId() );
    HGCalDetId detId_tc2( tc2.detId() );
    if( (detId_tc1.layer() != detId_tc2.layer()) ||
        (detId_tc1.subdetId() != detId_tc2.subdetId()) ||
        (detId_tc1.zside() != detId_tc2.zside()) ){
        return false;
    }   
    
    const auto neighbors_tc1 = triggerGeometry->getNeighborsFromTriggerCell( detId_tc1 );
    const auto neighbors_tc2 = triggerGeometry->getNeighborsFromTriggerCell( detId_tc2 );

    for(const auto neighbor_tc2 : neighbors_tc2)
    {        
        if( !( neighbors_tc1.find( neighbor_tc2 ) == neighbors_tc1.end() ) ){ 
            return true;
        }
    }            

    return false;
    
}


/* tc-clu */
bool HGCalClusteringImpl::isPertinent( const l1t::HGCalTriggerCell & tc, 
                                       const l1t::HGCalCluster & clu, 
                                       edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry ) const 
{

    HGCalDetId tcDetId( tc.detId() );
    HGCalDetId cluDetId( clu.seedDetId() );
    if( (tcDetId.layer() != cluDetId.layer()) ||
        (tcDetId.subdetId() != cluDetId.subdetId()) ||
        (tcDetId.zside() != cluDetId.zside()) ){
        return false;
    }   
    
    const edm::PtrVector<l1t::HGCalTriggerCell> pertinentTC = clu.triggercells();
    for( edm::PtrVector<l1t::HGCalTriggerCell>::const_iterator it_tc = pertinentTC.begin(); it_tc!= pertinentTC.end(); ++it_tc){
        HGCalDetId tcInCluDetId( (*it_tc)->detId() );
        const auto neighbors = triggerGeometry->getNeighborsFromTriggerCell( tcInCluDetId );
        if( !( neighbors.find(tcDetId) == neighbors.end() ) ){ 
            return true;
        }
    }

    return false;

}


/* clu-clu */
bool HGCalClusteringImpl::isPertinent( const l1t::HGCalCluster & clu1, 
                                       const l1t::HGCalCluster & clu2, 
                                       edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry ) const 
{
    HGCalDetId detId_clu1( clu1.seedDetId() );
    HGCalDetId detId_clu2( clu2.seedDetId() );

    if( (detId_clu1.layer() != detId_clu2.layer()) ||
        (detId_clu1.subdetId() != detId_clu2.subdetId()) ||
        (detId_clu1.zside() != detId_clu2.zside()) ){
        return false;
    }   

    if( clu1.seedDetId() == clu2.seedDetId() ){
        return false;
    }

    const edm::PtrVector<l1t::HGCalTriggerCell> pertinentTC_clu1 = clu1.triggercells();
    const edm::PtrVector<l1t::HGCalTriggerCell> pertinentTC_clu2 = clu2.triggercells();
   
    int Ntc_clu1 = pertinentTC_clu1.size();
    int Ntc_clu2 = pertinentTC_clu2.size();

    /* assume 1.06cm^2 cells */
    double maxTClength = 3.84;
    double r_max = (Ntc_clu1 + Ntc_clu2) * maxTClength;
    double distance = ( clu1.centre() - clu2.centre() ).mag(); 
    
    if( distance < r_max){
        for( edm::PtrVector<l1t::HGCalTriggerCell>::const_iterator tc1 = pertinentTC_clu1.begin();  tc1 != pertinentTC_clu1.end(); ++tc1 ){
            for( edm::PtrVector<l1t::HGCalTriggerCell>::const_iterator tc2 = pertinentTC_clu2.begin();  tc2 != pertinentTC_clu2.end(); ++tc2 ){
                if( this->isPertinent( **tc1, **tc2, triggerGeometry ) ){
                    return true;
                }
            }
        }
    }

    return false;

}


/* merge clusters that have common neighbors */
void HGCalClusteringImpl::mergeClusters( l1t::HGCalCluster & main_cluster, 
                                         l1t::HGCalCluster & secondary_cluster ) const
{

    const edm::PtrVector<l1t::HGCalTriggerCell> pertinentTC = secondary_cluster.triggercells();
    
    for( edm::PtrVector<l1t::HGCalTriggerCell>::iterator tc = pertinentTC.begin(); tc != pertinentTC.end(); ++tc ){
        main_cluster.addTriggerCell(*tc);
    }
    
}


void HGCalClusteringImpl::triggerCellReshuffling_( const edm::PtrVector<l1t::HGCalTriggerCell> & triggerCellsPtrs, 
                                                   std::vector<edm::Ptr<l1t::HGCalTriggerCell>> (&reshuffledTriggerCells)[2][40] 
    ){

    for( edm::PtrVector<l1t::HGCalTriggerCell>::const_iterator tc = triggerCellsPtrs.begin(); tc != triggerCellsPtrs.end(); ++tc){
        int endcap = (*tc)->zside() == -1 ? 0 : 1 ;
        HGCalDetId tcDetId( (*tc)->detId() );
        int subdet = tcDetId.subdetId();
        int layer = -1;
        if(subdet==3){
            layer = (*tc)->layer();
        }
        else if(subdet==4){
            layer = (*tc)->layer() + 28;
        }
        
        reshuffledTriggerCells[endcap][layer-1].emplace_back(*tc);
        
    }

}


void HGCalClusteringImpl::NNKernel( std::vector<edm::Ptr<l1t::HGCalTriggerCell>> (&reshuffledTriggerCells)[2][40],
                                    l1t::HGCalClusterBxCollection & clusters,
                                    edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry,
                                    int endcap, 
                                    int layer                                
    ){
   
    /* declaring the clusters vector */
    std::vector<l1t::HGCalCluster> clustersTmp;

    /* loop over the trigger-cells */
    for( std::vector<edm::Ptr<l1t::HGCalTriggerCell>>::iterator tc=reshuffledTriggerCells[endcap][layer].begin(); tc!=reshuffledTriggerCells[endcap][layer].end(); tc++ ){
                
        if( (*tc)->mipPt() < triggerCellThreshold_ ){
            continue;
        }
        
        bool createNewC2d = true;
        int i_clu=0;
                    
        for(std::vector<l1t::HGCalCluster>::iterator clu = clustersTmp.begin(); clu != clustersTmp.end(); ++clu, ++i_clu ){
            if( this->isPertinent( **tc, *clu, triggerGeometry ) ){
                clustersTmp.at( i_clu ).addTriggerCell( *tc );                    
                createNewC2d = false;
                break;
            }         
        }
                    
        if( createNewC2d ){
            clustersTmp.emplace_back( *tc );
        }
    }
    
    /* declaring the vector with possible clusters merged */

    for(unsigned i_clu1 = 0; i_clu1 < clustersTmp.size(); ++i_clu1){
      
        std::vector<int> idx_cluToRm;
        for(unsigned i_clu2 = 0; i_clu2 < clustersTmp.size(); ++i_clu2){
            if( this->isPertinent( clustersTmp.at(i_clu1), clustersTmp.at(i_clu2), triggerGeometry ) ){
                this->mergeClusters( clustersTmp.at(i_clu1), clustersTmp.at(i_clu2) );
                idx_cluToRm.push_back( i_clu2 );
            }           
        }       
        
        /* erase all the clusters that has been added */
        std::sort( idx_cluToRm.begin(), idx_cluToRm.end() );
        std::reverse( idx_cluToRm.begin(), idx_cluToRm.end() );

        for(unsigned idx = 0; idx < idx_cluToRm.size(); ++idx){
            clustersTmp.erase( clustersTmp.begin() + idx_cluToRm.at(idx) );
        }

    }

    /* store clusters in the persistent collection */
    for( unsigned i(0); i<clustersTmp.size(); ++i ){
        /* threshold in transverse-mip */
        if( clustersTmp.at(i).containsSeed( seedThreshold_ ) ){
            clusters.push_back( 0, clustersTmp.at(i) );
        }
    }
    
    clustersTmp.clear();

}


void HGCalClusteringImpl::clusterize( const edm::PtrVector<l1t::HGCalTriggerCell> & triggerCellsPtrs, 
                                      l1t::HGCalClusterBxCollection & clusters,
                                      edm::ESHandle<HGCalTriggerGeometryBase> triggerGeometry
    ){
    
    std::vector<edm::Ptr<l1t::HGCalTriggerCell>> reshuffledTriggerCells[2][40];
    triggerCellReshuffling_( triggerCellsPtrs, reshuffledTriggerCells );
    
    for(int iec=0; iec<2; ++iec){
        for(int il=0; il<40; ++il){

            NNKernel( reshuffledTriggerCells, clusters, triggerGeometry, iec, il );
            
        }
    }

}

