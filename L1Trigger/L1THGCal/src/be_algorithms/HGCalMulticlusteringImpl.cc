#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalMulticlusteringImpl.h"
#include "DataFormats/Math/interface/deltaR.h"

<<<<<<< HEAD
<<<<<<< HEAD
//class constructor
HGCalMulticlusteringImpl::HGCalMulticlusteringImpl(const edm::ParameterSet& conf){    
    dR_ = conf.getParameter<double>("dR_searchNeighbour");
}

void HGCalMulticlusteringImpl::clusterise( const l1t::HGCalClusterBxCollection & clusters_, 
                                           l1t::HGCalMulticlusterBxCollection & multiclusters_)
{
   
    edm::LogInfo("HGCclusterParameters") << "Multicluster dR for Near Neighbour search: " << dr_;  
        
    for(l1t::HGCalClusterBxCollection::const_iterator clu = clusters_.begin(); clu != clusters_.end(); ++clu){
        
        int imclu=0;
        vector<int> tcPertinentMulticlusters;
        for(l1t::HGCalMulticlusterBxCollection::const_iterator mclu = multiclusters_.begin(); mclu != multiclusters_.end(); ++mclu,++imclu)
            if( mclu->isPertinent(*clu, dr_) )
                tcPertinentMulticlusters.push_back(imclu);

        if( tcPertinentMulticlusters.size() == 0 ){
            l1t::HGCalMulticluster obj( *clu );
            multiclusters_.push_back( 0, obj );
        }
        else{
            uint minDist = 1;
            uint targetMulticlu = 0; 
            for( std::vector<int>::const_iterator imclu = tcPertinentMulticlusters.begin(); imclu != tcPertinentMulticlusters.end(); ++imclu ){
                double d = ( multiclusters_.at(0, *imclu).centre() - clu->centreNorm() ).Mag2() ;
                if( d < minDist ){
                    minDist = d;
                    targetMulticlu = *imclu;
                }
            } 

            multiclusters_.at(0, targetMulticlu).addClu( *clu );
        
        }            
    }    
}
