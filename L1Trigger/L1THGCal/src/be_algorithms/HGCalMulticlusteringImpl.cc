#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalMulticlusteringImpl.h"
#include "DataFormats/Math/interface/deltaR.h"


HGCalMulticlusteringImpl::HGCalMulticlusteringImpl( const edm::ParameterSet& beCodecConfig ) 
{
    
    dR_ = beCodecConfig.getParameter<double>("dR_searchNeighbour");

}


void HGCalMulticlusteringImpl::clusterise( const l1t::HGCalClusterBxCollection & clusters_, 
                                           l1t::HGCalMulticlusterBxCollection & multiclusters_)
{
   
    edm::LogInfo("HGCclusterParameters") << "Multicluster dR for Near Neighbour search: " << dR_;  

    if( clusters_.size() > 0 ){}
        
    for(l1t::HGCalClusterBxCollection::const_iterator clu = clusters_.begin(); clu != clusters_.end(); ++clu){
        
        
        

    }
    
}
 
