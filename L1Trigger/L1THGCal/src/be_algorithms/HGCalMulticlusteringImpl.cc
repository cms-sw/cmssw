#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalMulticlusteringImpl.h"
#include "DataFormats/Math/interface/deltaR.h"


HGCalMulticlusteringImpl::HGCalMulticlusteringImpl( const edm::ParameterSet& conf ) :
    dr_(conf.getParameter<double>("dR_multicluster"))
{    
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster dR for Near Neighbour search: " << dr_;  
}


bool HGCalMulticlusteringImpl::isPertinent( const l1t::HGCalCluster & clu, 
                                            const l1t::HGCalMulticluster & mclu, 
                                            double dR ) const
{
    HGCalDetId cluDetId( clu.seedDetId() );
    HGCalDetId firstClusterDetId( mclu.firstClusterDetId() );
    
    if( cluDetId.zside() != firstClusterDetId.zside() ){
        return false;
    }
    if( ( mclu.centreProj() - clu.centreProj() ).mag() < dR ){
        return true;
    }
    return false;

}


void HGCalMulticlusteringImpl::clusterize( const edm::OrphanHandle<l1t::HGCalClusterBxCollection> clustersHandle, 
                                           l1t::HGCalMulticlusterBxCollection & multiclusters)
{
        
    if( clustersHandle.isValid() ){
        std::vector<l1t::HGCalMulticluster> multiclustersTmp;
        int iclu = 0;
        for(l1t::HGCalClusterBxCollection::const_iterator clu = clustersHandle->begin(); clu != clustersHandle->end(); ++clu, ++iclu){
        
            int imclu=0;
            vector<int> tcPertinentMulticlusters;
            for(l1t::HGCalMulticlusterBxCollection::const_iterator mclu = multiclustersTmp.begin(); mclu != multiclustersTmp.end(); ++mclu,++imclu){
                if( this->isPertinent(*clu, *mclu, dr_) ){
                    tcPertinentMulticlusters.push_back(imclu);
                }
            }
            if( tcPertinentMulticlusters.size() == 0 ){
                l1t::HGCalMulticluster obj( *clu );
                multiclustersTmp.emplace_back( obj );
            }
            else{
                unsigned minDist = 1;
                unsigned targetMulticlu = 0; 
                for( std::vector<int>::const_iterator imclu = tcPertinentMulticlusters.begin(); imclu != tcPertinentMulticlusters.end(); ++imclu ){
                    double d = ( multiclustersTmp.at(*imclu).centreProj() - clu->centreProj() ).mag() ;
                    if( d < minDist ){
                        minDist = d;
                        targetMulticlu = *imclu;
                    }
                } 

                multiclustersTmp.at( targetMulticlu ).addCluster( *clu );
                edm::Ptr<l1t::HGCalCluster> p( clustersHandle, iclu, true );
                multiclustersTmp.at( targetMulticlu ).addClusterList( p );       
            
            }
        
        }

        /* making the collection of multiclusters */
        for( unsigned i(0); i<multiclustersTmp.size(); ++i ){
            if( multiclustersTmp.size()>0 ){
                multiclusters.push_back( 0, multiclustersTmp.at(i) );
            }
        }
    }
}
