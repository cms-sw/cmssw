#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalClusteringImpl.h"
#include "DataFormats/Common/interface/PtrVector.h"
#include "DataFormats/Common/interface/OrphanHandle.h"

//class constructor
HGCalClusteringImpl::HGCalClusteringImpl(const edm::ParameterSet & conf):
    seedThreshold_(conf.getParameter<double>("seeding_threshold")),
    triggerCellThreshold_(conf.getParameter<double>("clustering_threshold")),
    dr_(conf.getParameter<double>("dR_cluster")),
    maxTClenght_(conf.getParameter<double>("maxTClength")),
    clusteringAlgorithmType_(conf.getParameter<string>("clusterType"))
{    
    edm::LogInfo("HGCalClusterParameters") << "C2d Clustering Algorithm selected : " << clusteringAlgorithmType_ ; 
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


void HGCalClusteringImpl::clusterize( const edm::PtrVector<l1t::HGCalTriggerCell> & triggerCellsPtrs, 
                                      l1t::HGCalClusterBxCollection & clusters
    ){

    bool isSeed[triggerCellsPtrs.size()];
    
    /* search for cluster seeds */
    int itc(0);
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
         
            unsigned minDist(300);
            unsigned targetClu(0);
                        
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

/* storing trigger cells into vector per layer and per endcap */
void HGCalClusteringImpl::triggerCellReshuffling( const edm::PtrVector<l1t::HGCalTriggerCell> & triggerCellsPtrs, 
                                                  std::array< std::array<std::vector<edm::Ptr<l1t::HGCalTriggerCell>>,40>,2> & reshuffledTriggerCells 
    ){

    for( edm::PtrVector<l1t::HGCalTriggerCell>::const_iterator tc = triggerCellsPtrs.begin(); tc != triggerCellsPtrs.end(); ++tc){
        int endcap = (*tc)->zside() == -1 ? 0 : 1 ;
        HGCalDetId tcDetId( (*tc)->detId() );
        int subdet = tcDetId.subdetId();
        int layer = -1;

        if( subdet == HGCEE ){ 
            layer = (*tc)->layer();
        }
        else if( subdet == HGCHEF ){
            layer = (*tc)->layer() + 28;
        }
        else if( subdet == HGCHEB ){
            edm::LogWarning("DataNotFound") << "WARNING: the BH trgCells are not yet implemented";            
        }
        
        reshuffledTriggerCells[endcap][layer-1].emplace_back(*tc);
        
    }

}


/* merge clusters that have common neighbors */
void HGCalClusteringImpl::mergeClusters( l1t::HGCalCluster & main_cluster, 
                                         const l1t::HGCalCluster & secondary_cluster ) const
{

    const edm::PtrVector<l1t::HGCalTriggerCell> pertinentTC = secondary_cluster.triggercells();
    
    for( edm::PtrVector<l1t::HGCalTriggerCell>::iterator tc = pertinentTC.begin(); tc != pertinentTC.end(); ++tc ){
        main_cluster.addTriggerCell(*tc);
    }
    
}


void HGCalClusteringImpl::NNKernel( std::vector<edm::Ptr<l1t::HGCalTriggerCell>> & reshuffledTriggerCells,
                                    l1t::HGCalClusterBxCollection & clusters,
                                    const HGCalTriggerGeometryBase & triggerGeometry
    ){
   
    /* declaring the clusters vector */
    std::vector<l1t::HGCalCluster> clustersTmp;

    std::unordered_map<uint32_t, int> cluNNmap;
    int i_clu(0);                       

    /* loop over the trigger-cells */
    for( std::vector<edm::Ptr<l1t::HGCalTriggerCell>>::iterator tc=reshuffledTriggerCells.begin(); tc!=reshuffledTriggerCells.end(); tc++ ){
                
        if( (*tc)->mipPt() < triggerCellThreshold_ ){
            continue;
        }
        
        bool createNewC2d(true);
        
        for( auto it : cluNNmap ){
            HGCalDetId nnDetId( it.first );
            const auto neighbors = triggerGeometry.getNeighborsFromTriggerCell( nnDetId );
            
            if( !( neighbors.find( (*tc)->detId() ) == neighbors.end() ) )
            {
                clustersTmp.at( it.second ).addTriggerCell( *tc );                    
                cluNNmap.insert(std::make_pair( (*tc)->detId(), it.second ) );           
                clustersTmp.at( it.second ).setIsComplete(true);
                createNewC2d = false;
                break;
            }            
        }
        
        if( createNewC2d ){
            clustersTmp.emplace_back( *tc );
            cluNNmap.insert(std::make_pair( (*tc)->detId(), i_clu ) );
            clustersTmp.at( i_clu ).setIsComplete(true);
            ++i_clu;
        }
    }
    
    /* declaring the vector with possible clusters merged */
    for(unsigned i_clu1 = 0; i_clu1 < clustersTmp.size(); ++i_clu1){
        std::unordered_set<uint32_t> cluTcSet;

        const edm::PtrVector<l1t::HGCalTriggerCell> pertinentTC_clu1 = clustersTmp.at(i_clu1).triggercells();
        
        for(edm::PtrVector<l1t::HGCalTriggerCell>::const_iterator tc_clu1 = pertinentTC_clu1.begin(); tc_clu1 != pertinentTC_clu1.end(); ++tc_clu1 ){
            cluTcSet.insert( (*tc_clu1)->detId() );
            HGCalDetId tcInClu1DetId( (*tc_clu1)->detId() );
            const auto neighbors = triggerGeometry.getNeighborsFromTriggerCell( tcInClu1DetId );
            for(const auto neighbor : neighbors)
            {
                if( !( cluTcSet.find( neighbor ) == cluTcSet.end() ) )
                {                     
                    cluTcSet.insert( neighbor );
                }
            }
        }        
            
        for(unsigned i_clu2(i_clu1+1); i_clu2 < clustersTmp.size(); ++i_clu2){

            const edm::PtrVector<l1t::HGCalTriggerCell> pertinentTC_clu2 = clustersTmp.at(i_clu2).triggercells();
            for(edm::PtrVector<l1t::HGCalTriggerCell>::const_iterator tc_clu2 = pertinentTC_clu2.begin(); tc_clu2 != pertinentTC_clu2.end(); ++tc_clu2 ){
                
                if( !( cluTcSet.find( (*tc_clu2)->detId() ) == cluTcSet.end() ) && clustersTmp.at(i_clu2).isComplete()==true )
                {                     
                    this->mergeClusters( clustersTmp.at(i_clu1), clustersTmp.at(i_clu2) );
                    clustersTmp.at(i_clu1).setIsComplete(false);
                }                
            }                   
        }

    }

    /* store clusters in the persistent collection */
    for( unsigned i(0); i<clustersTmp.size(); ++i ){
        
        bool saveInCollection(false);
        const edm::PtrVector<l1t::HGCalTriggerCell> pertinentTC = clustersTmp.at(i).triggercells();
        
        for(edm::PtrVector<l1t::HGCalTriggerCell>::const_iterator tc = pertinentTC.begin(); tc != pertinentTC.end(); ++tc ){
            /* threshold in transverse-mip */
            if( (*tc)->mipPt() > seedThreshold_ ){
                saveInCollection = true;
            }
        }
        if(saveInCollection){
            clusters.push_back( 0, clustersTmp.at(i) );
        }
        
    }
    
}


void HGCalClusteringImpl::clusterize( const edm::PtrVector<l1t::HGCalTriggerCell> & triggerCellsPtrs, 
                                      l1t::HGCalClusterBxCollection & clusters,
                                      const HGCalTriggerGeometryBase & triggerGeometry
    ){

    std::array< std::array< std::vector<edm::Ptr<l1t::HGCalTriggerCell> >,40>,2> reshuffledTriggerCells; 
    triggerCellReshuffling( triggerCellsPtrs, reshuffledTriggerCells );

    for(int iec=0; iec<2; ++iec){
        for(int il=0; il<40; ++il){
            std::vector<edm::Ptr<l1t::HGCalTriggerCell>> oneLayerTriggerCell( reshuffledTriggerCells[iec][il] );
            NNKernel( oneLayerTriggerCell, clusters, triggerGeometry );
        }
    }

}

