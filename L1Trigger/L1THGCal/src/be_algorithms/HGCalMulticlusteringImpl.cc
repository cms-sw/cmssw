#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalMulticlusteringImpl.h"
#include "DataFormats/Math/interface/deltaR.h"


HGCalMulticlusteringImpl::HGCalMulticlusteringImpl( const edm::ParameterSet& beCodecConfig ) 
{
    
    dr_ = beCodecConfig.getParameter<double>("dR_multicluster");

}


void HGCalMulticlusteringImpl::clusterise( const l1t::HGCalClusterBxCollection & clusters_, 
                                           l1t::HGCalMulticlusterBxCollection & multiclusters_)
{
   
    edm::LogInfo("HGCclusterParameters") << "Multicluster dR for Near Neighbour search: " << dr_;  
        
    std::vector<l1t::HGCalMulticluster> multiclustersTmp;
    for(l1t::HGCalClusterBxCollection::const_iterator clu = clusters_.begin(); clu != clusters_.end(); ++clu){
        
        int imclu=0;
        vector<int> tcPertinentMulticlusters;
        for(l1t::HGCalMulticlusterBxCollection::const_iterator mclu = multiclustersTmp.begin(); mclu != multiclustersTmp.end(); ++mclu,++imclu)
            if( mclu->isPertinent(*clu, dr_) )
                tcPertinentMulticlusters.push_back(imclu);

        if( tcPertinentMulticlusters.size() == 0 ){
            l1t::HGCalMulticluster obj( *clu );
            multiclustersTmp.push_back( obj );
        }
        else{
            uint minDist = 1;
            uint targetMulticlu = 0; 
            for( std::vector<int>::const_iterator imclu = tcPertinentMulticlusters.begin(); imclu != tcPertinentMulticlusters.end(); ++imclu ){
                double d = ( multiclustersTmp.at(*imclu).centre() - clu->centreNorm() ).Mag2() ;
                if( d < minDist ){
                    minDist = d;
                    targetMulticlu = *imclu;
                }
            } 

            multiclustersTmp.at(targetMulticlu).addClu( *clu );
        
        }
        
   }

    /* making the collection of multiclusters */
    for( unsigned i(0); i<multiclustersTmp.size(); ++i )
        multiclusters_.push_back( 0, multiclustersTmp.at(i) );

}

