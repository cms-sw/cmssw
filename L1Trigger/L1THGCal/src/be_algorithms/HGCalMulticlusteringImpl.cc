#include "L1Trigger/L1THGCal/interface/be_algorithms/HGCalMulticlusteringImpl.h"
#include "DataFormats/Math/interface/deltaR.h"


HGCalMulticlusteringImpl::HGCalMulticlusteringImpl( const edm::ParameterSet& conf ) :
    dr_(conf.getParameter<double>("dR_multicluster")),
    ptC3dThreshold_(conf.getParameter<double>("minPt_multicluster")),
    calibSF_(conf.getParameter<double>("calibSF_multicluster"))
{    
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster dR for Near Neighbour search: " << dr_;  
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster minimum transverse-momentum: " << ptC3dThreshold_;
    edm::LogInfo("HGCalMulticlusterParameters") << "Multicluster global calibration factor: " << calibSF_;

}


bool HGCalMulticlusteringImpl::isPertinent( const l1t::HGCalCluster & clu, 
                                            const l1t::HGCalMulticluster & mclu, 
                                            double dR ) const
{
    HGCalDetId cluDetId( clu.detId() );
    HGCalDetId firstClusterDetId( mclu.detId() );
    
    if( cluDetId.zside() != firstClusterDetId.zside() ){
        return false;
    }
    if( ( mclu.centreProj() - clu.centreProj() ).mag() < dR ){
        return true;
    }
    return false;

}


void HGCalMulticlusteringImpl::clusterize( const edm::PtrVector<l1t::HGCalCluster> & clustersPtrs, 
                                           l1t::HGCalMulticlusterBxCollection & multiclusters)
{
           
    std::vector<l1t::HGCalMulticluster> multiclustersTmp;

    int iclu = 0;
    for(edm::PtrVector<l1t::HGCalCluster>::const_iterator clu = clustersPtrs.begin(); clu != clustersPtrs.end(); ++clu, ++iclu){
        
        int imclu=0;
        vector<int> tcPertinentMulticlusters;
        for( const auto& mclu : multiclustersTmp ){
            if( this->isPertinent(**clu, mclu, dr_) ){
                tcPertinentMulticlusters.push_back(imclu);
            }
            ++imclu;
        }
        if( tcPertinentMulticlusters.size() == 0 ){
            multiclustersTmp.emplace_back( *clu );
        }
        else{
            unsigned minDist = 1;
            unsigned targetMulticlu = 0; 
            for( int imclu : tcPertinentMulticlusters ){
                double d = ( multiclustersTmp.at(imclu).centreProj() - (*clu)->centreProj() ).mag() ;
                if( d < minDist ){
                    minDist = d;
                    targetMulticlu = imclu;
                }
            } 

            multiclustersTmp.at( targetMulticlu ).addConstituent( *clu );
            
        }        
    }

    /* making the collection of multiclusters */
    for( unsigned i(0); i<multiclustersTmp.size(); ++i ){
        math::PtEtaPhiMLorentzVector calibP4(  multiclustersTmp.at(i).pt() * calibSF_, 
                                               multiclustersTmp.at(i).eta(), 
                                               multiclustersTmp.at(i).phi(), 
                                               0. );
        // overwriting the 4p with the calibrated 4p     
        multiclustersTmp.at(i).setP4( calibP4 );
        if( multiclustersTmp.at(i).pt() > ptC3dThreshold_ ){
            multiclusters.push_back( 0, multiclustersTmp.at(i));  
        }
    }
    
}
